#!/usr/bin/python
"""
  Map a run: planned OSM route, the track Matty actually drove, and the
  critical events, on one OpenStreetMap page.

  map_runs.py answers "where did it go and how good was the GPS". This
  answers "where did it go WRONG": every event that matters is placed at
  the position it happened, so a cluster of markers at one corner tells you
  more than any amount of scrolling through timestamps.

  Events detected, all from the log's own streams:

    JOINT FORCED      the articulation pushed past --joint-deg (default
                      55), which is beyond the 45 deg matty.py can even
                      command - so the rear is being twisted by something
                      it has run onto. THIS is what a wheel climbing a
                      pillar looks like; chassis roll barely moves,
                      because on an articulated robot the box holding the
                      IMU can stay level while the other half climbs.
    ROLLOVER / TILT   |roll| above --roll-deg (default 15). The marker
                      carries the peak angle and how long it lasted - a
                      sustained tilt is the robot up ON something, a spike
                      is a kerb.
    BUMPER            front or rear contact, with what the depth zones and
                      the finer depth_profile bins were reporting at that
                      moment - which is how you see that neither of them
                      saw it coming.
    HEADING ERROR     pointing more than --heading-deg (default 55) off the
                      planned route's own direction for over 2 s. This is
                      the "went the wrong way then turned sharply" and the
                      "circled at the start" signature.
    OFF CORRIDOR      cross-track past --cross-m (default 3) for over 3 s -
                      driving parallel to the path rather than on it.
    ROUTE STATE       every recovering / lost / failed / arrived transition.
    RE-PLAN           each new route_plan, with the reason.

  Usage
  -----
    python map_events.py "../../logs/.../2026_09_05_v3_czu_test1/*.log" \\
           -o events.html

    # only the runs that actually contain something interesting
    python map_events.py "logs/.../*.log" -o events.html --only-eventful

    # a different site - draw only that extract behind the track
    python map_events.py "logs/.../*.log" -o events.html --map maps/stromovka.json

  Works entirely offline, page included: the backdrop is drawn as vectors
  from the local OSM extracts (see map_basemap.py), not fetched as tiles.
"""
import argparse
import map_basemap
import glob
import json
import math
import os
import sys

from osgar.logger import LogReader, lookup_stream_names
from osgar.lib.serialize import deserialize

from tulak_obstacle import normalize_angle

WANTED = ('nmea_data', 'route_hint', 'route_plan', 'rpy', 'rotation',
          'bumpers_front', 'bumpers_rear', 'obstacle_zones', 'depth_profile',
          'desired_steering', 'joint_angle')


def _fix(data):
    lat, lon = data.get('lat'), data.get('lon')
    if lat is None or lon is None:
        return None
    if data.get('lat_dir') == 'S':
        lat = -lat
    if data.get('lon_dir') == 'W':
        lon = -lon
    return lat, lon


def scan(logfile, compass_offset, hardiron, hardiron_phase,
         roll_deg, heading_deg, cross_m, joint_deg=55.0):
    """One pass over a log -> (track, planned route, events)."""
    names = lookup_stream_names(logfile)
    ids = {i + 1: n.split('.')[-1] for i, n in enumerate(names)
           if n.split('.')[-1] in WANTED}
    if not ids:
        return None

    track, events, plans = [], [], []
    zones, prof, speed, yaw = (None,) * 3, None, 0.0, None
    roll_run, heading_run, cross_run, joint_run = None, None, None, None
    last_state = None
    # STUCK - avoidance repeating in place. A reversal is the one part of
    # every avoidance cycle that is visible in the raw command stream, so
    # several within stuck_window seconds is the signature of the pole
    # loops (2026-09-11: 4-10 backups per pole, never more than 1m moved).
    reversals, last_stuck = [], None
    stuck_window, stuck_count = 20.0, 3

    def here(t):
        """position at time t - the last fix at or before it"""
        best = None
        for tt, la, lo in track:
            if tt <= t:
                best = (la, lo)
            else:
                break
        return best

    def close(run, kind, t, extra):
        if run is None:
            return None
        start, peak = run
        if t - start >= (0.3 if kind in ('tilt', 'joint') else 2.0):
            events.append({'kind': kind, 't': round(start, 1),
                           'dur': round(t - start, 1), 'peak': round(peak, 1),
                           'text': extra})
        return None

    with LogReader(logfile, only_stream_id=list(ids)) as log:
        for dt, stream_id, raw in log:
            channel = ids[stream_id]
            data = deserialize(raw)
            t = dt.total_seconds()

            if channel == 'nmea_data':
                f = _fix(data)
                if f:
                    track.append((t, f[0], f[1]))
            elif channel == 'desired_steering':
                was_reversing = speed < 0
                speed = data[0] / 1000.0
                if speed < 0 and not was_reversing:
                    reversals.append(t)
                    while reversals and t - reversals[0] > stuck_window:
                        reversals.pop(0)
                    if len(reversals) >= stuck_count and (last_stuck is None or t - last_stuck > stuck_window):
                        events.append({'kind': 'stuck', 't': round(reversals[0], 1),
                                       'dur': round(t - reversals[0], 1), 'peak': len(reversals),
                                       'text': '%d reversals in %.0f s - avoidance repeating in place'
                                               % (len(reversals), t - reversals[0])})
                        last_stuck = t
            elif channel == 'obstacle_zones':
                zones = tuple(data)
            elif channel == 'depth_profile':
                prof = data
            elif channel == 'rotation':
                yaw = math.radians(data[0] / 100.0)
            elif channel in ('bumpers_front', 'bumpers_rear'):
                if data:
                    z = ' / '.join('--' if v is None else '%.2f' % v for v in zones)
                    bins = ('n/a' if not prof else
                            ' '.join('--' if b is None else '%.1f' % b for b in prof))
                    events.append({'kind': 'bumper', 't': round(t, 1), 'dur': 0, 'peak': 0,
                                   'text': '%s bumper at %.2f m/s<br>zones L/C/R: %s'
                                           '<br>bins: %s' % (channel[8:].upper(), speed, z, bins)})
            elif channel == 'joint_angle':
                joint = data[0] / 100.0
                if abs(joint) > joint_deg:
                    joint_run = (t, joint) if joint_run is None else                         (joint_run[0], joint if abs(joint) > abs(joint_run[1]) else joint_run[1])
                else:
                    joint_run = close(joint_run, 'joint', t,
                                       'joint forced to %.0f deg - past the 45 deg the platform '
                                       'can command')
            elif channel == 'rpy':
                roll = data[0] / 100.0
                if abs(roll) >= roll_deg:
                    roll_run = (t, roll) if roll_run is None else \
                        (roll_run[0], roll if abs(roll) > abs(roll_run[1]) else roll_run[1])
                else:
                    roll_run = close(roll_run, 'tilt', t, 'peak roll %.1f deg')
            elif channel == 'route_plan':
                plans.append(data)
                events.append({'kind': 'plan', 't': round(t, 1), 'dur': 0, 'peak': 0,
                               'text': 'planned: %s<br>%.0f m, %d turns'
                                       % (data.get('reason'), data.get('total_m', 0),
                                          data.get('junctions', 0))})
            elif channel == 'route_hint':
                state = data.get('state')
                if state != last_state:
                    if state in ('recovering', 'lost', 'failed', 'arrived', 'gps_lost'):
                        events.append({'kind': 'state', 't': round(t, 1), 'dur': 0, 'peak': 0,
                                       'text': 'route state -> <b>%s</b>' % state})
                    last_state = state
                ct = data.get('cross_track_m')
                if ct is not None and abs(ct) >= cross_m:
                    cross_run = (t, ct) if cross_run is None else \
                        (cross_run[0], ct if abs(ct) > abs(cross_run[1]) else cross_run[1])
                else:
                    cross_run = close(cross_run, 'cross', t, 'peak cross-track %.1f m')
                rb = data.get('road_bearing_deg')
                if rb is not None and yaw is not None:
                    compass = normalize_angle(math.pi / 2 - yaw + compass_offset)
                    if hardiron:
                        compass = normalize_angle(
                            compass + hardiron * math.cos(compass - hardiron_phase))
                    err = math.degrees(normalize_angle(compass - math.radians(rb)))
                    if abs(err) >= heading_deg:
                        heading_run = (t, err) if heading_run is None else \
                            (heading_run[0], err if abs(err) > abs(heading_run[1]) else heading_run[1])
                    else:
                        heading_run = close(heading_run, 'heading', t, 'peak %.0f deg off the route axis')
    close(joint_run, 'joint', track[-1][0] if track else 0,
          'joint forced to %.0f deg - past the 45 deg the platform can command')
    close(roll_run, 'tilt', track[-1][0] if track else 0, 'peak roll %.1f deg')
    close(heading_run, 'heading', track[-1][0] if track else 0, 'peak %.0f deg off the route axis')
    close(cross_run, 'cross', track[-1][0] if track else 0, 'peak cross-track %.1f m')

    for e in events:
        pos = here(e['t'])
        e['pos'] = list(pos) if pos else None
        if '%' in e['text']:
            try:
                e['text'] = e['text'] % e['peak']
            except TypeError:
                pass
    events = [e for e in events if e['pos']]
    route = plans[-1]['points'] if plans else []
    return {'name': os.path.basename(logfile),
            'track': [[round(la, 7), round(lo, 7), round(t, 1)] for t, la, lo in track],
            'route': route, 'events': events}


HTML = """<!doctype html>
<meta charset="utf-8">
<title>Matty run events</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body,#map{height:100%%;margin:0}
  .legend{background:#fff;padding:9px 11px;border-radius:6px;font:12px/1.5 sans-serif;
          box-shadow:0 1px 5px rgba(0,0,0,.4);max-width:300px}
  .legend b{display:block;margin-bottom:5px}
  .k{display:inline-block;width:12px;height:12px;border-radius:50%%;margin-right:6px;
     border:2px solid #fff;box-shadow:0 0 0 1px #666;vertical-align:-2px}
</style>
<div id="map"></div>
<script>
const RUNS = %s;
const WAYS = %s;
const TILE_URL = %s;
const STYLE = {
  joint:   {c:'#000000', label:'JOINT FORCED (climbing / jammed)'},
  tilt:    {c:'#d7191c', label:'tilt'},
  bumper:  {c:'#ff7f00', label:'bumper contact'},
  stuck:   {c:'#e7298a', label:'STUCK - avoidance repeating in place'},
  heading: {c:'#6a3d9a', label:'heading way off route'},
  cross:   {c:'#1f78b4', label:'off corridor (parallel)'},
  state:   {c:'#33a02c', label:'route state change'},
  plan:    {c:'#777777', label:'re-plan'}
};
const map = L.map('map').setView([50.130, 14.379], 17);
__BASEMAP__
const overlays = {}; const all = [];
RUNS.forEach(run => {
  const g = L.layerGroup();
  if (run.route.length > 1) {
    L.polyline(run.route, {color:'#1f4fd8', weight:7, opacity:.35}).addTo(g)
      .bindPopup(run.name + '<br>planned route');
  }
  const pts = run.track.map(p => [p[0], p[1]]);
  if (pts.length > 1) {
    L.polyline(pts, {color:'#111', weight:2.5, opacity:.8}).addTo(g)
      .bindPopup(run.name + '<br>driven track');
    L.circleMarker(pts[0], {radius:5, color:'#000', fillColor:'#fff', fillOpacity:1}).addTo(g)
      .bindPopup(run.name + '<br>START');
  }
  run.events.forEach(e => {
    const s = STYLE[e.kind] || {c:'#000'};
    L.circleMarker(e.pos, {radius: e.kind==='tilt'?9:7, color:'#fff', weight:2,
                           fillColor:s.c, fillOpacity:.95}).addTo(g)
     .bindPopup('<b>' + run.name + '</b><br>t = ' + e.t + ' s'
                + (e.dur ? ' (' + e.dur + ' s)' : '') + '<br>' + e.text);
  });
  g.addTo(map); overlays[run.name + '  (' + run.events.length + ')'] = g;
  all.push(...pts, ...run.route);
});
if (all.length) map.fitBounds(L.polyline(all).getBounds().pad(0.12));
L.control.layers(null, overlays, {collapsed:false}).addTo(map);
const legend = L.control({position:'bottomright'});
legend.onAdd = function(){
  const d = L.DomUtil.create('div','legend');
  d.innerHTML = '<b>Critical events</b>'
    + Object.values(STYLE).map(s => '<span class="k" style="background:'+s.c+'"></span>'+s.label).join('<br>')
    + '<hr style="margin:7px 0">'
    + '<span style="display:inline-block;width:16px;border-top:7px solid #1f4fd8;opacity:.35"></span> planned route<br>'
    + '<span style="display:inline-block;width:16px;border-top:3px solid #111"></span> driven track<br>'
    + '<i>click any marker for detail; toggle runs top-right</i>';
  return d;
};
legend.addTo(map);
</script>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logfile', nargs='+', help='recorded log file(s); globs are expanded')
    parser.add_argument('-o', '--output', default='events.html')
    parser.add_argument('--roll-deg', type=float, default=15.0)
    parser.add_argument('--joint-deg', type=float, default=55.0)
    parser.add_argument('--heading-deg', type=float, default=55.0)
    parser.add_argument('--cross-m', type=float, default=3.0)
    parser.add_argument('--compass-offset-deg', type=float, default=7.0)
    parser.add_argument('--hardiron-deg', type=float, default=10.3)
    parser.add_argument('--hardiron-phase-deg', type=float, default=9.0)
    parser.add_argument('--map', nargs='*', default=None,
                         help='local OSM extract(s) to draw as the basemap '
                              '(default: %s)' % ', '.join(map_basemap.DEFAULT_MAPS))
    parser.add_argument('--tile-url', default='',
                         help='optional raster tile URL template. Empty by default and '
                              'deliberately so: OpenStreetMap blocks this kind of traffic to '
                              'its own servers, and the local extract is the better backdrop '
                              'anyway. Only pass a service you are entitled to use.')
    parser.add_argument('--only-eventful', action='store_true',
                         help='skip runs with no events at all')
    args = parser.parse_args()

    logfiles = []
    for pattern in args.logfile:
        logfiles.extend(sorted(glob.glob(pattern)) or [pattern])

    runs = []
    for logfile in logfiles:
        try:
            run = scan(logfile, math.radians(args.compass_offset_deg),
                        math.radians(args.hardiron_deg), math.radians(args.hardiron_phase_deg),
                        args.roll_deg, args.heading_deg, args.cross_m, args.joint_deg)
        except Exception as e:  # noqa: BLE001 - one bad log must not stop the map
            print('  %-34s skipped (%s: %s)' % (os.path.basename(logfile)[-32:], type(e).__name__, e))
            continue
        if run is None or not run['track']:
            print('  %-34s skipped (no GPS)' % os.path.basename(logfile)[-32:])
            continue
        if args.only_eventful and not run['events']:
            continue
        kinds = {}
        for e in run['events']:
            kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
        print('  %-34s %4d fixes, %2d events %s'
              % (os.path.basename(logfile)[-32:], len(run['track']), len(run['events']),
                 kinds if kinds else ''))
        runs.append(run)

    if not runs:
        print('nothing to map')
        return 1
    pts = []
    for run in runs:
        pts.extend([p[0], p[1]] for p in run['track'])
        pts.extend(run['route'])
    ways, n_ways = map_basemap.ways_json(pts, args.map)
    print('  basemap: %d local OSM ways drawn (no tile server)' % n_ways)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write((HTML % (json.dumps(runs), ways, json.dumps(args.tile_url)))
                .replace('__BASEMAP__', map_basemap.BASEMAP_JS))
    print()
    print('wrote %s (%d runs, %d events) - open it in a browser'
          % (args.output, len(runs), sum(len(r['events']) for r in runs)))
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: expandtab sw=4 ts=4
