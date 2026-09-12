#!/usr/bin/python
"""
  Plot the GPS track of one or more recorded runs onto an OpenStreetMap
  background, as a single self-contained .html file you open in a browser.

  What it shows, per run:
    - the raw GPS track, one dot per fix, coloured by HDOP (green = good
      geometry, red = poor). Click any dot for time / HDOP / satellites /
      fix quality.
    - START and END markers.
    - GLITCH markers (big red rings) wherever two consecutive fixes imply
      a speed the robot cannot physically reach (--max-speed, default
      2 m/s against Matty's ~0.5 m/s ceiling). These are the GPS jumps -
      multipath under trees, or a fix briefly degrading - and they are the
      single most useful thing to look at when judging position error,
      because the robot demonstrably did NOT move that way.
    - BLIND segments (thick dark overlay) wherever the depth camera was
      returning nothing, so you can see WHERE in the park the sun blinded
      it rather than just how often. Detected the same way the driver
      does: the centre zone reporting its 0.0 fail-safe with most of the
      depth profile unknown.
    - optionally the odometry track (--odometry), rotated to the run's
      initial GPS heading and drawn from the first fix. Wheel odometry
      drifts over minutes, so this is NOT ground truth - but it IS smooth,
      so where the GPS wanders away from a locally-smooth odometry curve,
      the wander is GPS noise rather than real motion.

  Everything is computed from the log's own streams; nothing here needs
  the robot, the camera, or any Python package beyond numpy/osgar.

  Usage
  -----
    # one run
    python map_runs.py path/to/run.log

    # several runs on one map, custom output name, with odometry overlay
    python map_runs.py logs/2026_08_29_V2_Stromovka0/*.log \
           -o stromovka.html --odometry

    # only runs longer than 60s, and be stricter about what counts as a jump
    python map_runs.py logs/*.log --min-duration 60 --max-speed 1.5

  Then just open the produced .html (default: gps_map.html) in a browser.
  Tiles are fetched from OpenStreetMap, so that first view needs internet;
  the track data itself is embedded in the file.
"""
import argparse
import glob
import json
import math
import os
import sys

from osgar.logger import LogReader, lookup_stream_names
from osgar.lib.serialize import deserialize

import map_basemap


EARTH_R = 6371000
# distinct, colour-blind-friendly-ish track colours, cycled per run
TRACK_COLORS = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
                '#42d4f4', '#f032e6', '#bfef45', '#469990', '#9A6324']


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def read_run(logfile, blind_valid_frac=0.25):
    """Pull everything the map needs out of one log in a single pass."""
    names = lookup_stream_names(logfile)
    name_to_id = {n: i + 1 for i, n in enumerate(names)}
    id_to_name = {v: k for k, v in name_to_id.items()}
    wanted = ['gps.nmea_data', 'platform.pose2d',
              'obstdet3d_zones.obstacle_zones', 'obstdet3d_zones.depth_profile']
    ids = [name_to_id[n] for n in wanted if n in name_to_id]
    if 'gps.nmea_data' not in name_to_id:
        return None

    fixes, pose, blind_flags = [], [], []
    center = None
    with LogReader(logfile, only_stream_id=ids) as log:
        for dt, stream_id, raw in log:
            name = id_to_name[stream_id]
            data = deserialize(raw)
            t = dt.total_seconds()
            if name == 'gps.nmea_data':
                lat, lon = data.get('lat'), data.get('lon')
                if lat is None or lon is None:
                    continue
                if data.get('lat_dir') == 'S':
                    lat = -lat
                if data.get('lon_dir') == 'W':
                    lon = -lon
                fixes.append({'t': round(t, 1), 'lat': lat, 'lon': lon,
                              'hdop': data.get('hdop'), 'sats': data.get('sats'),
                              'q': data.get('quality')})
            elif name == 'platform.pose2d':
                pose.append((t, data[0] / 1000.0, data[1] / 1000.0))
            elif name == 'obstdet3d_zones.obstacle_zones':
                center = data[1]
            elif name == 'obstdet3d_zones.depth_profile':
                # same test the driver uses: centre at its 0.0 fail-safe AND
                # most of the wider profile unknown (see _update_blind_state)
                if data:
                    frac = sum(1 for d in data if d is not None) / len(data)
                else:
                    frac = 0.0
                blind_flags.append((t, center is not None and center <= 0.0
                                    and frac < blind_valid_frac))
    if len(fixes) < 2:
        return None

    # implied speed between consecutive fixes -> GPS jumps
    for i, f in enumerate(fixes):
        if i == 0:
            f['v'] = 0.0
            continue
        p = fixes[i - 1]
        dt_ = max(1e-3, f['t'] - p['t'])
        f['v'] = round(haversine(p['lat'], p['lon'], f['lat'], f['lon']) / dt_, 2)

    # was the depth camera blind at (approximately) each fix?
    bi = 0
    for f in fixes:
        while bi + 1 < len(blind_flags) and blind_flags[bi + 1][0] <= f['t']:
            bi += 1
        f['blind'] = bool(blind_flags[bi][1]) if blind_flags else False

    return {'name': os.path.basename(logfile),
            'fixes': fixes,
            'pose': pose,
            'duration': fixes[-1]['t'] - fixes[0]['t']}


def odometry_latlon(run):
    """Odometry track expressed as lat/lon, anchored at the first GPS fix
    and rotated so its initial direction matches the GPS track's initial
    direction. Purely a visual aid - odometry drifts, so treat it as a
    locally-smooth reference, never as truth."""
    pose, fixes = run['pose'], run['fixes']
    if len(pose) < 2:
        return []
    # GPS direction over the first 10m of real movement
    lat0, lon0 = fixes[0]['lat'], fixes[0]['lon']
    ref = None
    for f in fixes[1:]:
        if haversine(lat0, lon0, f['lat'], f['lon']) > 10.0:
            ref = f
            break
    if ref is None:
        ref = fixes[-1]
    mlat = math.cos(math.radians(lat0))
    gps_ang = math.atan2((ref['lon'] - lon0) * mlat, ref['lat'] - lat0)
    # matching odometry direction over the same span
    ox0, oy0 = pose[0][1], pose[0][2]
    odo_ref = None
    for _, x, y in pose:
        if math.hypot(x - ox0, y - oy0) > 10.0:
            odo_ref = (x, y)
            break
    if odo_ref is None:
        odo_ref = (pose[-1][1], pose[-1][2])
    odo_ang = math.atan2(odo_ref[1] - oy0, odo_ref[0] - ox0)
    rot = gps_ang - odo_ang
    cs, sn = math.cos(rot), math.sin(rot)

    out = []
    for _, x, y in pose[::5]:            # 2Hz is plenty for drawing
        dx, dy = x - ox0, y - oy0
        north = dx * cs - dy * sn
        east = dx * sn + dy * cs
        out.append([lat0 + math.degrees(north / EARTH_R),
                    lon0 + math.degrees(east / (EARTH_R * mlat))])
    return out


HTML = """<!doctype html>
<meta charset="utf-8">
<title>Matty GPS tracks</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body,#map{height:100%;margin:0}
  .legend{background:#fff;padding:8px 10px;border-radius:6px;font:12px/1.45 sans-serif;
          box-shadow:0 1px 5px rgba(0,0,0,.4);max-width:260px}
  .legend b{display:block;margin-bottom:4px}
  .sw{display:inline-block;width:11px;height:11px;margin-right:5px;border:1px solid #666;vertical-align:-1px}
</style>
<div id="map"></div>
<script>
const RUNS = __DATA__;
const WAYS = __WAYS__;
const TILE_URL = __TILE__;
const map = L.map('map');
__BASEMAP__

function hdopColor(h){
  if(h===null||h===undefined) return '#888';
  if(h<0.8) return '#1a9850';
  if(h<1.2) return '#91cf60';
  if(h<2.0) return '#fee08b';
  if(h<5.0) return '#fc8d59';
  return '#d73027';
}
const overlays={}; let bounds=L.latLngBounds([]);

RUNS.forEach((run,i)=>{
  const g=L.layerGroup();
  const pts=run.fixes.map(f=>[f.lat,f.lon]);
  pts.forEach(p=>bounds.extend(p));
  L.polyline(pts,{color:run.color,weight:3,opacity:.85}).addTo(g);

  // blind stretches drawn on top of the track
  let seg=[];
  run.fixes.forEach(f=>{
    if(f.blind){ seg.push([f.lat,f.lon]); }
    else if(seg.length>1){ L.polyline(seg,{color:'#111',weight:7,opacity:.55}).addTo(g); seg=[]; }
    else seg=[];
  });
  if(seg.length>1) L.polyline(seg,{color:'#111',weight:7,opacity:.55}).addTo(g);

  run.fixes.forEach(f=>{
    L.circleMarker([f.lat,f.lon],{radius:3,color:hdopColor(f.hdop),weight:1,
      fillColor:hdopColor(f.hdop),fillOpacity:.9})
     .bindPopup(`<b>${run.name}</b><br>t=${f.t}s<br>hdop=${f.hdop}<br>sats=${f.sats}`+
                `<br>quality=${f.q}<br>implied speed=${f.v} m/s`+
                (f.blind?'<br><b>depth camera BLIND</b>':'')).addTo(g);
    if(f.v>run.maxspeed)
      L.circleMarker([f.lat,f.lon],{radius:11,color:'#d73027',weight:3,fill:false})
       .bindPopup(`<b>GPS JUMP</b><br>${run.name}<br>t=${f.t}s`+
                  `<br>implied ${f.v} m/s (robot cannot exceed ~0.5)`).addTo(g);
  });
  if(run.odometry.length>1)
    L.polyline(run.odometry,{color:run.color,weight:2,opacity:.9,dashArray:'6,6'}).addTo(g);
  L.marker(pts[0]).bindPopup('START '+run.name).addTo(g);
  L.marker(pts[pts.length-1]).bindPopup('END '+run.name).addTo(g);
  g.addTo(map);
  overlays[`${run.name} (${Math.round(run.duration)}s)`]=g;
});

map.fitBounds(bounds.pad(0.1));
L.control.layers(null,overlays,{collapsed:false}).addTo(map);
const lg=L.control({position:'bottomright'});
lg.onAdd=function(){
  const d=L.DomUtil.create('div','legend');
  d.innerHTML='<b>GPS fix quality (HDOP)</b>'+
   '<span class="sw" style="background:#1a9850"></span>&lt;0.8 excellent<br>'+
   '<span class="sw" style="background:#91cf60"></span>0.8–1.2 good<br>'+
   '<span class="sw" style="background:#fee08b"></span>1.2–2.0 fair<br>'+
   '<span class="sw" style="background:#fc8d59"></span>2.0–5.0 poor<br>'+
   '<span class="sw" style="background:#d73027"></span>&gt;5 bad<br>'+
   '<hr style="margin:6px 0"><span class="sw" style="background:#111"></span>depth camera blind<br>'+
   '<span class="sw" style="border:2px solid #d73027;background:#fff"></span>GPS jump (impossible speed)<br>'+
   '<span class="sw" style="background:repeating-linear-gradient(90deg,#333 0 4px,#fff 4px 8px)"></span>odometry (drifts; smooth reference)';
  return d;
};
lg.addTo(map);
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logfile', nargs='+', help='log file(s); wildcards allowed')
    ap.add_argument('-o', '--output', default='gps_map.html', help='output HTML (default gps_map.html)')
    ap.add_argument('--odometry', action='store_true',
                    help='also draw the wheel-odometry track (dashed) as a smooth reference')
    ap.add_argument('--max-speed', type=float, default=2.0,
                    help='implied speed (m/s) above which a fix pair is flagged as a GPS jump '
                         '(default 2.0; Matty tops out near 0.5)')
    ap.add_argument('--map', nargs='*', default=None,
                    help='local OSM extract(s) to draw as the basemap (default: %s)'
                         % ', '.join(map_basemap.DEFAULT_MAPS))
    ap.add_argument('--tile-url', default='',
                    help='optional raster tile URL template. Empty by default: OpenStreetMap '
                         'blocks this kind of traffic to its own servers. Only pass a service '
                         'you are entitled to use.')
    ap.add_argument('--min-duration', type=float, default=0.0,
                    help='skip runs shorter than this many seconds')
    args = ap.parse_args()

    paths = []
    for pattern in args.logfile:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    runs = []
    for i, p in enumerate(paths):
        try:
            run = read_run(p)
        except Exception as e:                      # a truncated log shouldn't kill the map
            print(f'  {os.path.basename(p)}: skipped ({e})')
            continue
        if run is None:
            print(f'  {os.path.basename(p)}: skipped (no GPS fixes)')
            continue
        if run['duration'] < args.min_duration:
            print(f'  {os.path.basename(p)}: skipped (only {run["duration"]:.0f}s)')
            continue
        run['color'] = TRACK_COLORS[len(runs) % len(TRACK_COLORS)]
        run['maxspeed'] = args.max_speed
        run['odometry'] = odometry_latlon(run) if args.odometry else []
        jumps = sum(1 for f in run['fixes'] if f['v'] > args.max_speed)
        blind = sum(1 for f in run['fixes'] if f['blind'])
        hd = [f['hdop'] for f in run['fixes'] if f['hdop'] is not None]
        print(f'  {run["name"][-18:-4]}: {len(run["fixes"]):5d} fixes  {run["duration"]:6.0f}s  '
              f'hdop med={sorted(hd)[len(hd)//2] if hd else "n/a"}  '
              f'jumps={jumps}  blind_fixes={blind} ({100*blind/len(run["fixes"]):.0f}%)')
        run.pop('pose')
        runs.append(run)

    if not runs:
        print('nothing to draw')
        return 1
    pts = [[f['lat'], f['lon']] for run in runs for f in run['fixes']]
    ways, n_ways = map_basemap.ways_json(pts, args.map)
    print('  basemap: %d local OSM ways drawn (no tile server)' % n_ways)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(HTML.replace('__DATA__', json.dumps(runs))
                    .replace('__WAYS__', ways)
                    .replace('__TILE__', json.dumps(args.tile_url))
                    .replace('__BASEMAP__', map_basemap.BASEMAP_JS))
    print(f'\nwrote {args.output} ({len(runs)} run(s)) - open it in a browser')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: expandtab sw=4 ts=4
