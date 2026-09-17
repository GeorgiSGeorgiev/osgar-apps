#!/usr/bin/python
"""
  Map the events that decide a Robotour run: where Matty left the mapped
  road, and where the sensing that should have stopped it did or did not
  fire.

  map_events.py already maps the mechanical events (joint forced, tilt,
  bumper, route state). This maps the three failure modes the 2026-09-12
  Stromovka analysis turned up, none of which that tool can see because
  none of them are a single stream crossing a threshold:

    OFF ROAD        the robot's GPS position is more than --off-m past the
                    edge of the nearest routable OSM way, for more than
                    --off-sec. Way half-widths come from osm_router's own
                    DEFAULT_HIGHWAY_HALFWIDTH, so "off the road" here means
                    the same thing it means to the router. THIS is the
                    disqualifying event - everything else on the map is
                    only interesting because it leads here.

    GREY BIN PUSH   an invalid (grey) edge depth bin while steering 12+ deg away
                    from it - the 2026-09-14 artefact: open road running into
                    the distance reads as no measurement and pushed Matty off
                    it. The popup gives the road fraction the network saw under
                    that bin.

    GPS TAKEOVER    the router in recovery, or in junction mode with authority
                    0.8+, while Matty steered 12+ deg and the camera saw road
                    straight ahead - GPS overruling a confident road mask.

    GPS BIAS        GPS+map put Matty 2+ m past the edge of the nearest way
                    while the camera saw it squarely on a road, for 3+ fixes -
                    where the map and the receiver disagree with reality.

    DROP-OFF IGNORED  obstdet3d_zones published ground_hazard=True for at
                    least --hazard-frames consecutive frames while the app
                    was driving forward. On 2026-09-12 the app had
                    enable_ground_hazard=false, so every one of these was
                    discarded; the longest run is the rollover at 17:23.

    GROUND AS OBSTACLE  the centre zone read closer than turning_dist while
                    all three zones agreed within --zone-spread and the
                    depth profile was open. Three zones reporting the same
                    moderate distance is a plane, not an object - it is the
                    row window sitting on the pavement. See the
                    ground-plane fit in the analysis notes.

  Usage
  -----
    python map_offroad.py "../../logs/.../2026_09_12_v4_Stromovka1/*.log" \\
           -o offroad.html --map maps/stromovka.json

  Offline like the others: the backdrop is drawn from the local OSM
  extract via map_basemap, never from a tile server.
"""
import argparse
import glob
import json
import math
import os
import sys

import map_basemap
from osgar.logger import LogReader, lookup_stream_names
from osgar.lib.serialize import deserialize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osm_router import (DEFAULT_ALLOWED_HIGHWAY, DEFAULT_HIGHWAY_HALFWIDTH,
                        DEFAULT_HALFWIDTH_M, EXCLUDED_ACCESS, PERMISSIVE_FOOT)

EARTH_R = 6371000.0


def load_segments(map_files, lat0):
    """Every routable way segment, in local metres, with its half-width and
    surface. Same admission rules as osm_router.RoadGraph so that "off the
    road" on this map means what it means to the robot."""
    kx = math.radians(1) * EARTH_R * math.cos(math.radians(lat0))
    ky = math.radians(1) * EARTH_R
    segs = []
    for path in map_files:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        nodes = data['nodes']
        for way in data['ways']:
            tags = way['tags']
            highway = tags.get('highway')
            if highway not in DEFAULT_ALLOWED_HIGHWAY:
                continue
            if tags.get('access') in EXCLUDED_ACCESS and tags.get('foot') not in PERMISSIVE_FOOT:
                continue
            half = DEFAULT_HIGHWAY_HALFWIDTH.get(highway, DEFAULT_HALFWIDTH_M)
            try:
                half = max(half, float(tags['width']) / 2.0)
            except (KeyError, ValueError, TypeError):
                pass
            pts = [(nodes[n][1] * kx, nodes[n][0] * ky) for n in way['nodes'] if n in nodes]
            for a, b in zip(pts, pts[1:]):
                segs.append((a[0], a[1], b[0], b[1], half,
                             tags.get('surface'), highway))
    return segs, kx, ky


def off_road_m(lon, lat, segs, kx, ky):
    """Metres past the edge of the nearest routable way (0 = on one), plus
    that way's surface tag."""
    px, py = lon * kx, lat * ky
    best, best_surf, best_hw = 1e9, None, None
    for ax, ay, bx, by, half, surf, hw in segs:
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 <= 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        d = math.hypot(px - (ax + t * dx), py - (ay + t * dy)) - half
        if d < best:
            best, best_surf, best_hw = d, surf, hw
    return max(0.0, best), best_surf, best_hw


MASK_HFOV = math.radians(69.0)
PROFILE_HFOV = math.radians(66.0)


def _mask_stats(mask, n_bins):
    """road fraction (lower half), centroid, road dead ahead, and road per
    depth-bin bearing band - the same geometry tulak_obstacle uses"""
    h, w = mask.shape
    low = mask[h // 2:, :]
    frac = float(low.mean())
    xs = low.nonzero()[1]
    cx = float(xs.mean()) / w if len(xs) else 0.5
    ahead = float(mask[int(h * .70):int(h * .85), int(w * .40):int(w * .60)].mean())
    band = mask[int(h * .50):int(h * .80), :]
    bins = []
    for i in range(n_bins):
        b0 = -PROFILE_HFOV / 2 + PROFILE_HFOV * i / n_bins
        b1 = -PROFILE_HFOV / 2 + PROFILE_HFOV * (i + 1) / n_bins
        c0 = max(0, min(w - 1, int((math.tan(b0) / math.tan(MASK_HFOV / 2) + 1) / 2 * w)))
        c1 = max(1, min(w, int((math.tan(b1) / math.tan(MASK_HFOV / 2) + 1) / 2 * w)))
        sub = band[:, c0:c1]
        bins.append(float(sub.mean()) if sub.size else 0.0)
    return dict(frac=frac, cx=cx, ahead=ahead, bins=bins)


def scan(logfile, segs, kx, ky, args):
    names = lookup_stream_names(logfile)
    n2i = {n: i + 1 for i, n in enumerate(names)}
    i2n = {v: k for k, v in n2i.items()}
    want = ['gps.nmea_data', 'obstdet3d_zones.ground_hazard', 'app.desired_steering',
            'obstdet3d_zones.obstacle_zones', 'obstdet3d_zones.depth_profile',
            'osm_router.route_hint', 'oak.nn_mask']
    ids = [n2i[w] for w in want if w in n2i]
    if 'gps.nmea_data' not in n2i:
        return [], [], None

    track, events = [], []
    pos = None                    # last (lat, lon)
    speed = 0.0
    hazard_run = 0
    hazard_start = None
    off_run_start = None
    off_peak = 0.0
    zones = [None, None, None]
    profile = []
    ground_run = 0
    ground_start = None
    steer = 0.0
    mask_stats = dict(frac=0.0, cx=0.5, ahead=0.0, bins=[])
    grey_run, grey_start, grey_info = 0, None, None
    take_run, take_start, take_info = 0, None, None
    bias_run, bias_start, bias_peak = 0, None, 0.0

    def here():
        return pos

    with LogReader(logfile, only_stream_id=ids) as log:
        for dt, sid, raw in log:
            name = i2n[sid]
            data = deserialize(raw)
            t = dt.total_seconds()

            if name == 'app.desired_steering':
                speed = data[0] / 1000.0
                steer = data[1] / 100.0

            elif name == 'oak.nn_mask':
                mask_stats = _mask_stats(data, len(profile) or 9)

            elif name == 'osm_router.route_hint':
                st, mode, auth = data.get('state'), data.get('mode'), data.get('authority') or 0.0
                taking = (st == 'recovering' or (mode == 'junction' and auth >= args.gps_authority))
                if taking and speed > 0.05 and abs(steer) >= args.grey_steer and mask_stats['ahead'] >= 0.6:
                    take_run += 1
                    if take_run == 1:
                        take_start, take_info = t, (st, mode, auth, steer)
                else:
                    if take_run >= 5 and pos:
                        events.append(dict(
                            kind='gpstake', pos=[pos[0], pos[1]], t=take_start,
                            text='router %s/%s (authority %.2f) steered %+.0f deg for %.1fs while the '
                                 'camera saw road straight ahead' % (take_info[0], take_info[1], take_info[2],
                                                                     take_info[3], t - take_start)))
                    take_run = 0

            elif name == 'obstdet3d_zones.obstacle_zones':
                zones = data

            elif name == 'obstdet3d_zones.depth_profile':
                profile = data
                n = len(profile)
                side = None
                if n >= 2 and (profile[0] is None) != (profile[-1] is None):
                    side = 'right' if profile[-1] is None else 'left'
                pushing = side is not None and speed > 0.05 and abs(steer) >= args.grey_steer and \
                    ((side == 'right' and steer > 0) or (side == 'left' and steer < 0))
                if pushing:
                    grey_run += 1
                    if grey_run == 1:
                        i = n - 1 if side == 'right' else 0
                        road = mask_stats['bins'][i] if i < len(mask_stats['bins']) else None
                        grey_start, grey_info = t, (side, steer, road)
                else:
                    if grey_run >= 2 and pos:
                        events.append(dict(
                            kind='greybin', pos=[pos[0], pos[1]], t=grey_start,
                            text='grey %s edge bin, steering %+.0f deg away from it for %.1fs; road '
                                 'network saw %s road under that bin' % (
                                     grey_info[0], grey_info[1], t - grey_start,
                                     '?' if grey_info[2] is None else '%.0f%%' % (100 * grey_info[2]))))
                    grey_run = 0

            elif name == 'obstdet3d_zones.ground_hazard':
                hazard = bool(data[0])
                # drop-off seen while still driving forward
                if hazard and speed > 0.05:
                    hazard_run += 1
                    if hazard_run == 1:
                        hazard_start = t
                else:
                    if hazard_run >= args.hazard_frames and pos:
                        events.append(dict(
                            kind='dropoff', pos=[pos[0], pos[1]], t=hazard_start,
                            text='drop-off seen for %.1fs (%d frames) while driving at up to '
                                 '%.2f m/s - ground_hazard was DISABLED in this run'
                                 % ((t - hazard_start), hazard_run, speed)))
                    hazard_run = 0

                # three zones agreeing on a moderate distance = a plane
                left, centre, right = zones
                if None not in (left, centre, right) and centre is not None:
                    spread = max(left, centre, right) - min(left, centre, right)
                    open_bins = sum(1 for d in profile if d is not None and d > 2.0)
                    if (centre < args.turning_dist and spread < args.zone_spread
                            and open_bins >= max(1, len(profile) // 2)):
                        ground_run += 1
                        if ground_run == 1:
                            ground_start = t
                    else:
                        if ground_run >= args.ground_frames and pos:
                            events.append(dict(
                                kind='groundobs', pos=[pos[0], pos[1]], t=ground_start,
                                text='all three depth zones agreed within %.2fm at %.2fm for '
                                     '%d frames while the profile stayed open - the row window '
                                     'was on the ground, not an obstacle'
                                     % (spread, centre, ground_run)))
                        ground_run = 0

            elif name == 'gps.nmea_data':
                if data.get('identifier') != '$GNGGA' or data.get('lat') is None:
                    continue
                lat, lon = data['lat'], data['lon']
                pos = (lat, lon)
                off, surf, hw = off_road_m(lon, lat, segs, kx, ky)
                track.append([lat, lon, round(off, 2)])
                camera_on_road = mask_stats['frac'] > 0.45 and abs(mask_stats['cx'] - 0.5) < 0.08
                if off >= 2.0 and camera_on_road:
                    bias_run += 1
                    if bias_run == 1:
                        bias_start, bias_peak = (t, lat, lon), off
                    bias_peak = max(bias_peak, off)
                else:
                    if bias_run >= 3:
                        events.append(dict(
                            kind='gpsbias', pos=[bias_start[1], bias_start[2]], t=bias_start[0],
                            text='GPS+map put Matty %.1f m past the road edge for %d fixes while the '
                                 'camera saw it squarely on a road' % (bias_peak, bias_run)))
                    bias_run = 0
                if off > args.off_m:
                    if off_run_start is None:
                        off_run_start = (t, lat, lon)
                        off_peak = off
                    off_peak = max(off_peak, off)
                else:
                    if off_run_start is not None and (t - off_run_start[0]) >= args.off_sec:
                        events.append(dict(
                            kind='offroad', pos=[off_run_start[1], off_run_start[2]],
                            t=off_run_start[0],
                            text='OFF ROAD for %.0fs, peak %.1f m past the edge of the nearest '
                                 '%s (%s)' % (t - off_run_start[0], off_peak, hw or 'way',
                                              surf or 'surface untagged')))
                    off_run_start = None
    return track, events, None


HTML_HEAD = """<!doctype html><meta charset="utf-8"><title>%(title)s</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body,#m{height:100%%;margin:0}
 .legend{background:#fff;padding:8px 10px;font:12px system-ui;line-height:1.5;
         box-shadow:0 1px 5px rgba(0,0,0,.4);border-radius:4px;max-width:280px}
 .legend b{display:block;margin-bottom:4px}
 .sw{display:inline-block;width:10px;height:10px;border-radius:50%%;margin-right:6px;
     vertical-align:middle;border:1px solid #fff}
</style><div id="m"></div><script>
var RUNS=%(runs)s, BASE=%(base)s, C=%(centre)s;
var map=L.map('m').setView(C,17);
var COL={offroad:'#e11d48',dropoff:'#f59e0b',groundobs:'#7c3aed',greybin:'#475569',gpstake:'#2563eb',gpsbias:'#0891b2'};
var NAME={offroad:'OFF ROAD (disqualifying)',dropoff:'drop-off seen while driving',
          groundobs:'ground read as obstacle',greybin:'grey depth bin pushed steering',
          gpstake:'GPS took the wheel from a confident road mask',gpsbias:'GPS+map bias (camera on road)'};
BASE.forEach(function(w){L.polyline(w.pts,{color:'#9aa3af',weight:w.major?4:2,opacity:.7}).addTo(map);});
RUNS.forEach(function(r){
  // the track, coloured by how far off the road it is
  for(var i=1;i<r.track.length;i++){
    var off=r.track[i][2];
    var c = off<=0.01?'#16a34a' : off<1?'#84cc16' : off<2?'#f59e0b' : '#e11d48';
    L.polyline([[r.track[i-1][0],r.track[i-1][1]],[r.track[i][0],r.track[i][1]]],
               {color:c,weight:3.5,opacity:.9}).addTo(map);
  }
  if(r.track.length)L.circleMarker([r.track[0][0],r.track[0][1]],
     {radius:5,color:'#000',fillColor:'#fff',fillOpacity:1}).addTo(map).bindPopup('start '+r.name);
  r.events.forEach(function(e){
    L.circleMarker(e.pos,{radius:8,color:'#fff',weight:2,fillColor:COL[e.kind],fillOpacity:.95})
     .addTo(map).bindPopup('<b>'+NAME[e.kind]+'</b><br>'+r.name+' t='+e.t.toFixed(1)+'s<br>'+e.text);
  });
});
var lg=L.control({position:'bottomright'});
lg.onAdd=function(){var d=L.DomUtil.create('div','legend');
 d.innerHTML='<b>%(title)s</b>'+
  '<span class="sw" style="background:#16a34a"></span>on a mapped road<br>'+
  '<span class="sw" style="background:#84cc16"></span>&lt;1 m past the edge<br>'+
  '<span class="sw" style="background:#f59e0b"></span>1-2 m past the edge<br>'+
  '<span class="sw" style="background:#e11d48"></span>&gt;2 m past the edge<br>'+
  '<hr style="margin:6px 0">'+
  '<span class="sw" style="background:#e11d48"></span>OFF ROAD episode<br>'+
  '<span class="sw" style="background:#f59e0b"></span>drop-off ignored<br>'+
  '<span class="sw" style="background:#7c3aed"></span>ground read as obstacle<br>'+
  '<span class="sw" style="background:#475569"></span>grey bin pushed steering<br>'+
  '<span class="sw" style="background:#2563eb"></span>GPS takeover over road mask<br>'+
  '<span class="sw" style="background:#0891b2"></span>GPS+map bias';
 return d;};
lg.addTo(map);
</script>"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('logs', help='glob of log files')
    p.add_argument('-o', '--output', default='offroad.html')
    p.add_argument('--map', action='append', default=None, help='OSM extract(s) to judge against')
    p.add_argument('--off-m', type=float, default=1.0, help='metres past the road edge to count')
    p.add_argument('--off-sec', type=float, default=3.0, help='for how long before it is an event')
    p.add_argument('--hazard-frames', type=int, default=5)
    p.add_argument('--ground-frames', type=int, default=5)
    p.add_argument('--zone-spread', type=float, default=0.45,
                   help='max L/C/R disagreement for the reading to be called a plane')
    p.add_argument('--turning-dist', type=float, default=1.2)
    p.add_argument('--grey-steer', type=float, default=12.0, help='deg of steering that counts as a push')
    p.add_argument('--gps-authority', type=float, default=0.8,
                   help='junction authority that counts as a takeover')
    args = p.parse_args()

    map_files = args.map or ['maps/stromovka.json']
    files = sorted(glob.glob(args.logs))
    if not files:
        print('no logs matched %s' % args.logs)
        return 1

    # latitude for the local projection - from the first fix we can find
    lat0 = None
    for f in files:
        names = lookup_stream_names(f)
        if 'gps.nmea_data' not in names:
            continue
        i = names.index('gps.nmea_data') + 1
        with LogReader(f, only_stream_id=[i]) as log:
            for dt, sid, raw in log:
                d = deserialize(raw)
                if d.get('lat'):
                    lat0 = d['lat']
                    break
        if lat0:
            break
    if lat0 is None:
        print('no GPS fix in any log')
        return 1

    segs, kx, ky = load_segments(map_files, lat0)
    print('map: %d routable segments from %s' % (len(segs), ', '.join(map_files)))

    runs, tally = [], {}
    for f in files:
        track, events, _ = scan(f, segs, kx, ky, args)
        if not track:
            continue
        name = os.path.basename(f)
        kinds = {}
        for e in events:
            kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
            tally[e['kind']] = tally.get(e['kind'], 0) + 1
        offs = [t[2] for t in track]
        print('  %-46s %5d fixes  off_road med=%.2f max=%.2f  %s'
              % (name, len(track), sorted(offs)[len(offs) // 2], max(offs), kinds or ''))
        runs.append(dict(name=name, track=track, events=events))

    lats = [t[0] for r in runs for t in r['track']]
    lons = [t[1] for r in runs for t in r['track']]
    centre = [sum(lats) / len(lats), sum(lons) / len(lons)]
    base = [dict(pts=w['p'], major=(w['k'] == 'road'))
            for w in map_basemap.load_ways(map_files,
                                           map_basemap.bounds_of(list(zip(lats, lons))))]
    print('  basemap: %d local OSM ways drawn (no tile server)' % len(base))

    html = HTML_HEAD % dict(
        title='Matty off-road &amp; sensing events',
        runs=json.dumps(runs), base=json.dumps(base), centre=json.dumps(centre))
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html)
    print('\nwrote %s (%d runs, %s)' % (args.output, len(runs), tally))
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: expandtab sw=4 ts=4
