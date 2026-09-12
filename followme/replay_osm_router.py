#!/usr/bin/python
"""
  Replay recorded runs through osm_router.py, without the robot.

  The router is the one new piece of Matty's navigation that cannot be
  bench-tested by driving around a room: it needs GPS, a real map, and a
  park. What it CAN be tested against is the 2026-08-29 Stromovka logs,
  which contain exactly the failure this module exists to prevent - long
  stretches of the robot out on the grass, far from any mapped path.

  So: take a recorded run, plan a route from where that run started to
  where it ended, then feed the recorded GPS fixes and odometry through
  the REAL OSMRouter class in recorded order and watch what it would have
  said. Two questions, both answerable offline:

    1. Would the corridor monitor have caught the real excursions?
       (Runs 125548 / 130005 / 134647 are 6-35m off-path for 20-60s.)
    2. Would it have false-alarmed on the runs that stayed on the paths?

  This is NOT a claim that the robot would have driven the planned route -
  the recorded track is whatever the robot did on the day, under a
  different controller. It is a test of the map, the graph filtering, the
  planner, the arclength tracker and the corridor monitor against real GPS
  in the real place, which is everything in osm_router.py except the aim
  point actually steering anything.

  Usage
  -----
    # every Stromovka run, summary table
    python replay_osm_router.py logs/.../2026_08_29_V2_Stromovka0/*.log \
           --map maps/stromovka.json

    # one run, per-fix detail plus a Leaflet map of planned vs actual
    python replay_osm_router.py logs/.../m02-...-260829_125548.log \
           --map maps/stromovka.json --verbose --html route_125548.html

    # plan to a specific coordinate instead of the run's own end point
    python replay_osm_router.py run.log --map maps/stromovka.json \
           --target 50.10583 14.42734

    # no log at all - just check the map/graph and plan one route
    python replay_osm_router.py --map maps/stromovka.json \
           --from 50.10583 14.42734 --target 50.10659 14.41743
"""
import argparse
import datetime
import glob
import json
import math
import os
import statistics
import sys

from osgar.exceptions import EmergencyStopException
from osgar.logger import LogReader, lookup_stream_names
from osgar.lib.serialize import deserialize

from osm_router import OSMRouter, RoadGraph, RouteState

import map_basemap


class FakeBus:
    """Same minimal stand-in view_obstacle.py uses - no threads, no queues,
    just enough for a Node's handlers to run against replayed data."""

    def __init__(self):
        self.published = {}
        self.counts = {}

    def register(self, *outputs):
        pass

    def publish(self, channel, data):
        self.published[channel] = data
        self.counts[channel] = self.counts.get(channel, 0) + 1
        return data

    def sleep(self, secs):
        pass

    def is_alive(self):
        return True


def router_config(map_file, overrides=None):
    config = {'map_file': map_file, 'compass_offset_deg': 6.1, 'log_interval_sec': 1e9}
    config.update(overrides or {})
    return config


def read_streams(logfile, wanted):
    """(timestamp, stream_name, data) for the streams we care about, in
    recorded order. Stream names differ between configs (the old obstacle
    config has no osm_router at all), so this matches on the channel part
    rather than requiring an exact producer name."""
    names = lookup_stream_names(logfile)
    ids = {}
    for i, name in enumerate(names):
        channel = name.split('.')[-1]
        if channel in wanted:
            ids[i + 1] = channel
    if not ids:
        return
    with LogReader(logfile, only_stream_id=list(ids.keys())) as log:
        for dt, stream_id, raw in log:
            yield dt, ids[stream_id], deserialize(raw)


def fix_from_nmea(data):
    lat, lon = data.get('lat'), data.get('lon')
    if lat is None or lon is None:
        return None
    if data.get('lat_dir') == 'S':
        lat = -lat
    if data.get('lon_dir') == 'W':
        lon = -lon
    return lat, lon


def replay(logfile, map_file, target=None, overrides=None, verbose=False):
    """Run one log through a real OSMRouter. Returns a summary dict, or
    None if the log has no usable GPS."""
    events = list(read_streams(logfile, {'nmea_data', 'pose2d', 'rotation'}))
    fixes = [(dt, fix_from_nmea(d)) for dt, ch, d in events if ch == 'nmea_data']
    fixes = [(dt, f) for dt, f in fixes if f is not None]
    if len(fixes) < 5:
        return None

    goal = tuple(target) if target else fixes[-1][1]
    bus = FakeBus()
    router = OSMRouter(router_config(map_file, overrides), bus)

    samples = []          # (t_sec, cross_track, state, authority)
    state_changes = []    # (t_sec, state)
    replans = []
    qr_sent = False
    last_state = None

    for dt, channel, data in events:
        router.time = dt
        if channel == 'rotation':
            router.on_rotation(data)
        elif channel == 'nmea_data':
            if not qr_sent and fix_from_nmea(data) is not None:
                # the QR is "shown" at the first usable fix - the router
                # itself waits for one before it can plan
                router.on_nmea_data(data)
                router.on_qr_code('%.6f, %.6f' % goal)
                qr_sent = True
                continue
            router.on_nmea_data(data)
        elif channel == 'pose2d':
            router.on_pose2d(data)
            hint = bus.published.get('route_hint')
            if hint is not None and router.route is not None:
                samples.append((dt.total_seconds(), router.cross_track,
                                 router.state, hint.get('authority')))
        plan = bus.published.get('route_plan')
        if plan is not None and (not replans or replans[-1][1] != plan['reason']
                                  or replans[-1][2] != plan['total_m']):
            replans.append((dt.total_seconds(), plan['reason'], plan['total_m']))
        if router.state != last_state:
            state_changes.append((dt.total_seconds(), router.state))
            last_state = router.state
            if verbose:
                print('    t=%7.1f  -> %s' % (dt.total_seconds(), router.state))

    if router.route is None and not samples:
        return {'log': os.path.basename(logfile), 'planned': False,
                'states': state_changes, 'replans': replans}

    cross = [abs(c) for _t, c, _s, _a in samples]
    recovering = [t for t, _c, s, _a in samples if s == RouteState.RECOVERING]
    lost = [t for t, _c, s, _a in samples if s == RouteState.LOST]
    authorities = [a for _t, _c, _s, a in samples if a]
    return {
        'log': os.path.basename(logfile),
        'planned': True,
        'route_m': router.route.total if router.route else 0.0,
        'route_points': len(router.route) if router.route else 0,
        'junctions': int(router.route.junction.sum()) if router.route else 0,
        'duration_s': samples[-1][0] - samples[0][0] if samples else 0.0,
        'cross_med': statistics.median(cross) if cross else float('nan'),
        'cross_p90': sorted(cross)[int(0.9 * len(cross))] if cross else float('nan'),
        'cross_max': max(cross) if cross else float('nan'),
        'recovering_s': len(recovering) * 0.1,
        'lost_s': len(lost) * 0.1,
        'authority_med': statistics.median(authorities) if authorities else float('nan'),
        'final_state': router.state,
        'progress_m': router.s,
        'states': state_changes,
        'replans': replans,
        'samples': samples,
        'router': router,
        'fixes': fixes,
        'goal': goal,
    }


HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Matty planned route vs recorded track</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body,#map{height:100%%;margin:0}
  .legend{background:#fff;padding:8px 10px;border-radius:6px;font:12px/1.45 sans-serif;
          box-shadow:0 1px 5px rgba(0,0,0,.4);max-width:280px}
  .legend b{display:block;margin-bottom:4px}
  .sw{display:inline-block;width:11px;height:11px;margin-right:5px;border:1px solid #666;vertical-align:-1px}
</style>
<div id="map"></div>
<script>
const DATA = %s;
const WAYS = %s;
const TILE_URL = %s;
const map = L.map('map').setView(DATA.route[0], 18);
__BASEMAP__

L.polyline(DATA.route, {color:'#1f4fd8', weight:6, opacity:.65}).addTo(map)
  .bindPopup('planned route: ' + DATA.route_m.toFixed(0) + ' m');
DATA.junctions.forEach(j => L.circleMarker(j, {radius:6, color:'#1f4fd8',
  fillColor:'#fff', fillOpacity:1, weight:2}).addTo(map).bindPopup('planned junction'));

function crossColor(c){
  const a = Math.abs(c);
  if (a > DATA.hard) return '#d7191c';
  if (a > DATA.soft) return '#fdae61';
  return '#1a9641';
}
DATA.track.forEach(p => {
  L.circleMarker([p[0], p[1]], {radius:3, color:crossColor(p[2]), weight:1,
    fillColor:crossColor(p[2]), fillOpacity:.85}).addTo(map)
   .bindPopup('t=' + p[3].toFixed(1) + 's<br>cross-track ' + p[2].toFixed(1) + ' m<br>' + p[4]);
});
L.marker(DATA.route[0]).addTo(map).bindPopup('START');
L.marker(DATA.goal).addTo(map).bindPopup('TARGET (from QR)');
// deferred, and with an explicit invalidateSize: a fitBounds issued while
// the container still measures 0x0 (which happens in embedded/snapshot
// renderers, and occasionally on a slow first paint) silently leaves the
// map at world zoom with everything off screen
function fitAll(){
  map.invalidateSize();
  map.fitBounds(L.polyline(DATA.route.concat(DATA.track.map(p => [p[0], p[1]]))).getBounds().pad(0.15),
                {maxZoom: 18});
}
// deliberately NOT called synchronously here: at that point the container
// can still be 0x0, and fitBounds against a zero-size map resolves to
// world zoom and pins the view there. The setView above is the fallback
// if neither of these ever fires.
if (document.readyState === 'complete') { fitAll(); }
else { window.addEventListener('load', fitAll); }
setTimeout(fitAll, 400);

const legend = L.control({position:'bottomright'});
legend.onAdd = function(){
  const d = L.DomUtil.create('div','legend');
  d.innerHTML = '<b>' + DATA.title + '</b>'
    + '<span class="sw" style="background:#1f4fd8"></span>planned route (' + DATA.route_m.toFixed(0) + ' m)<br>'
    + '<span class="sw" style="background:#1a9641"></span>on corridor (&lt;' + DATA.soft + ' m)<br>'
    + '<span class="sw" style="background:#fdae61"></span>drifting<br>'
    + '<span class="sw" style="background:#d7191c"></span>past hard limit (&gt;' + DATA.hard + ' m)<br>'
    + '<hr style="margin:6px 0">' + DATA.note;
  return d;
};
legend.addTo(map);
</script>
"""


def write_html(result, filename, map_files=None):
    """Planned route against what the robot actually did, coloured by
    cross-track. This is the view that makes an excursion obvious - a red
    arc bulging away from the blue line is the robot on the grass."""
    router = result['router']
    route = router.route
    track = []
    # cross-track is tracked per GPS fix; pair each fix with the nearest
    # sample in time so the map shows the same number the monitor saw
    samples = result['samples']
    si = 0
    for dt, (lat, lon) in result['fixes']:
        t = dt.total_seconds()
        while si + 1 < len(samples) and samples[si + 1][0] <= t:
            si += 1
        if not samples:
            break
        _st, cross, state, _a = samples[si]
        track.append([lat, lon, round(cross, 2), round(t, 1), state])
    data = {
        'title': result['log'],
        'route': [[round(la, 7), round(lo, 7)] for la, lo in route.points_ll],
        'junctions': [[round(route.points_ll[i][0], 7), round(route.points_ll[i][1], 7)]
                      for i in range(len(route)) if route.junction[i]],
        'route_m': route.total,
        'goal': list(result['goal']),
        'track': track,
        'soft': router.corridor_soft_m,
        'hard': router.corridor_hard_m,
        'note': ('recovering %.0fs, lost %.0fs, %d re-plans, final state <b>%s</b>'
                  % (result['recovering_s'], result['lost_s'],
                     max(0, len(result['replans']) - 1), result['final_state'])),
    }
    # basemap drawn from the same offline extract the router planned on,
    # not from OSM's tile servers - see map_basemap
    pts = list(data['route']) + [[p[0], p[1]] for p in track]
    ways, n_ways = map_basemap.ways_json(pts, map_files)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write((HTML_TEMPLATE % (json.dumps(data), ways, json.dumps('')))
                .replace('__BASEMAP__', map_basemap.BASEMAP_JS))
    print('  basemap: %d local OSM ways drawn (no tile server)' % n_ways)
    print('wrote %s' % filename)


def replay_with_app(logfile, map_file, target=None, overrides=None):
    """The whole chain, against a real recorded run: OSMRouter plans and
    publishes aim points, the REAL TulakObstacle consumes them alongside
    its own recorded depth/mask/GPS streams, and out come actual
    desired_steering commands.

    This is the integration test for the follower side - the four places
    tulak_obstacle.py reads route state (on_route_hint, the bearing-fade
    distance, the bearing weight, the hold). It runs against logs recorded
    BEFORE the mode existed, by injecting a router the original config did
    not have, which is the only way to exercise it on real data today.

    What it cannot show is whether the robot would have driven the route -
    the recorded depth and mask frames are from wherever the robot
    actually went. It shows that the plumbing works and that authority
    moves the steering the way it is supposed to."""
    from osgar.logger import lookup_config
    from tulak_obstacle import TulakObstacle

    full_cfg = lookup_config(logfile)['robot']
    app_cfg = dict(full_cfg['modules']['app']['init'])
    app_bus, router_bus = FakeBus(), FakeBus()
    app = TulakObstacle(app_cfg, app_bus)
    router = OSMRouter(router_config(map_file, overrides), router_bus)

    # which recorded stream feeds which app handler, from the log's own links
    to_app = {}
    for src, dst in full_cfg['links']:
        module, _, channel = dst.partition('.')
        if module == 'app':
            to_app.setdefault(src, []).append(channel)

    names = lookup_stream_names(logfile)
    id_to_name = {i + 1: n for i, n in enumerate(names)}
    wanted_ids = [i for i, n in id_to_name.items()
                  if n in to_app or n.split('.')[-1] in ('nmea_data', 'pose2d', 'rotation')]

    if target:
        goal = tuple(target)
    else:
        # Default to the point of the run FARTHEST from where it started,
        # not to its last fix: most of these runs loop back to roughly
        # where they began, which would plan a zero-length route and make
        # the router arrive before the robot has moved - a correct result
        # that tests nothing.
        fixes = [f for _dt, ch, d in read_streams(logfile, {'nmea_data'})
                 if ch == 'nmea_data' for f in [fix_from_nmea(d)] if f]
        if not fixes:
            raise RuntimeError('log has no usable GPS fix')
        origin = fixes[0]
        goal = max(fixes, key=lambda f: (f[0] - origin[0]) ** 2 + (f[1] - origin[1]) ** 2)
    qr_sent = False
    stats = {'cycles': 0, 'route_mode': 0, 'holding': 0, 'moving': 0,
             'authority': [], 'speeds': [], 'states': {}, 'emergency_stop_at': None,
             'modes': {}, 'mask_trust': [], 'bias_deg': [], 'steer_limit_deg': [],
             'speed_capped': 0, 'road_axis_err': []}

    with LogReader(logfile, only_stream_id=wanted_ids) as log:
        for dt, stream_id, raw in log:
            name = id_to_name[stream_id]
            data = deserialize(raw)
            channel = name.split('.')[-1]

            # feed the router first, so the aim point the app sees this
            # cycle is the current one - same order the live bus produces
            router.time = dt
            if channel == 'rotation':
                router.on_rotation(data)
            elif channel == 'nmea_data':
                router.on_nmea_data(data)
                if not qr_sent and router.last_fix is not None:
                    router.on_qr_code('%.6f, %.6f' % goal)
                    qr_sent = True
            elif channel == 'pose2d':
                router.on_pose2d(data)
                hint = router_bus.published.get('route_hint')
                if hint is not None:
                    app.time = dt
                    app.on_route_hint(hint)

            for app_channel in to_app.get(name, ()):
                handler = getattr(app, 'on_' + app_channel, None)
                if handler is None:
                    continue
                app.time = dt
                try:
                    handler(data)
                except EmergencyStopException:
                    # the operator's emergency stop, recorded like any
                    # other message - it ends the run here exactly as it
                    # did on the day, and is not a failure of this replay
                    stats['emergency_stop_at'] = dt.total_seconds()
                    return app, router, stats
                if app_channel == 'pose2d':
                    stats['cycles'] += 1
                    stats['route_mode'] += bool(app.route_mode)
                    stats['states'][app.route_state] = stats['states'].get(app.route_state, 0) + 1
                    if app.route_authority:
                        stats['authority'].append(app.route_authority)
                    mode = app.route_guidance_mode
                    if mode:
                        stats['modes'][mode] = stats['modes'].get(mode, 0) + 1
                    herr = app._route_heading_error()
                    if herr is not None:
                        stats['road_axis_err'].append(abs(math.degrees(herr)))
                        stats['mask_trust'].append(app._mask_trust())
                    if mode == 'corridor':
                        stats['bias_deg'].append(math.degrees(app._route_bias))
                    if app.route_steer_limit:
                        stats['steer_limit_deg'].append(math.degrees(app.route_steer_limit))
                    if app.route_speed_limit is not None:
                        stats['speed_capped'] += 1
                    command = app_bus.published.get('desired_steering')
                    if command is not None:
                        speed = command[0] / 1000.0
                        stats['speeds'].append(speed)
                        if abs(speed) < 1e-6:
                            stats['holding'] += 1
                        else:
                            stats['moving'] += 1
    return app, router, stats


def plan_only(map_file, start, goal, overrides=None):
    bus = FakeBus()
    router = OSMRouter(router_config(map_file, overrides), bus)
    router.time = datetime.timedelta(0)
    route = router.graph.plan(tuple(start), tuple(goal), uturn_penalty_m=0)
    if route is None:
        print('NO ROUTE from %s to %s' % (start, goal))
        return 1
    print('route: %.0f m over %d points, %d junctions '
           '(start %.1f m off-path, target %.1f m off-path)'
           % (route.total, len(route), int(route.junction.sum()),
              route.meta['start_snap_dist_m'], route.meta['goal_snap_dist_m']))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logfile', nargs='*', help='recorded log file(s); globs are expanded')
    parser.add_argument('--map', required=True, help='offline OSM map file from osm_fetch.py')
    parser.add_argument('--target', nargs=2, type=float, metavar=('LAT', 'LON'),
                        help='route target; default is where the replayed run ended')
    parser.add_argument('--from', dest='start', nargs=2, type=float, metavar=('LAT', 'LON'),
                        help='with --target and no log: just plan this route and exit')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='override a router config value, e.g. --set corridor_hard_m=6')
    parser.add_argument('--html', help='write a Leaflet map of planned route vs recorded track')
    parser.add_argument('--verbose', '-v', action='store_true', help='print every router state change')
    parser.add_argument('--with-app', action='store_true',
                         help='also drive the real TulakObstacle from the router output and report '
                              'what it commanded - the end-to-end integration check')
    args = parser.parse_args()

    overrides = {}
    for item in args.set:
        key, _, value = item.partition('=')
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value

    if args.start:
        if not args.target:
            parser.error('--from needs --target')
        return plan_only(args.map, args.start, args.target, overrides)

    logfiles = []
    for pattern in args.logfile:
        logfiles.extend(sorted(glob.glob(pattern)) or [pattern])
    if not logfiles:
        parser.error('give at least one log file, or --from/--target to only plan a route')

    if args.with_app:
        for logfile in logfiles:
            print('=== %s ===' % os.path.basename(logfile))
            app, router, stats = replay_with_app(logfile, args.map, target=args.target,
                                                  overrides=overrides)
            cycles = max(1, stats['cycles'])
            print('  pose2d cycles          : %d' % stats['cycles'])
            print('  route_mode engaged     : %.1f%% of cycles' % (100.0 * stats['route_mode'] / cycles))
            print('  router states seen     : %s' % stats['states'])
            print('  commanded stopped/moving: %d / %d' % (stats['holding'], stats['moving']))
            if stats['authority']:
                print('  bearing authority      : min %.2f  median %.2f  max %.2f'
                       % (min(stats['authority']),
                          statistics.median(stats['authority']),
                          max(stats['authority'])))
            if stats['speeds']:
                print('  commanded speed        : median %.2f  max %.2f m/s'
                       % (statistics.median(stats['speeds']), max(stats['speeds'])))
            if stats['modes']:
                print('  guidance modes         : %s' % stats['modes'])
            if stats['mask_trust']:
                mt = stats['mask_trust']
                print('  camera vs road axis    : median %.0fdeg  p90 %.0fdeg  max %.0fdeg'
                       % (statistics.median(stats['road_axis_err']),
                          sorted(stats['road_axis_err'])[int(.9*len(mt))],
                          max(stats['road_axis_err'])))
                print('  mask_trust             : median %.2f  min %.2f  (<1.0 in %.0f%% of cycles)'
                       % (statistics.median(mt), min(mt),
                          100.0*sum(1 for t in mt if t < 0.999)/len(mt)))
            if stats['bias_deg']:
                b = [abs(x) for x in stats['bias_deg']]
                print('  corridor bias |deg|    : median %.1f  p90 %.1f  max %.1f'
                       % (statistics.median(b), sorted(b)[int(.9*len(b))], max(b)))
            if stats['steer_limit_deg']:
                sl = stats['steer_limit_deg']
                print('  route steer limit      : min %.0fdeg  max %.0fdeg  (>25deg in %.0f%%)'
                       % (min(sl), max(sl), 100.0*sum(1 for x in sl if x > 25)/len(sl)))
            print('  speed cap active       : %.0f%% of cycles' % (100.0*stats['speed_capped']/cycles))
            print('  final: app.route_state=%s route_hold=%s remaining=%s'
                   % (app.route_state, app.route_hold, app.route_remaining_m))
            if stats['emergency_stop_at'] is not None:
                print('  (run ended by the recorded emergency stop at t=%.1fs)'
                       % stats['emergency_stop_at'])
        return 0

    header = ('%-22s %7s %6s %4s   %6s %6s %6s   %6s %5s  %s'
               % ('log', 'route_m', 'dur_s', 'jct', 'x_med', 'x_p90', 'x_max',
                  'recov', 'lost', 'final'))
    print(header)
    print('-' * len(header))
    results = []
    for logfile in logfiles:
        try:
            result = replay(logfile, args.map, target=args.target,
                             overrides=overrides, verbose=args.verbose)
        except Exception as e:  # noqa: BLE001 - one bad log must not stop the sweep
            print('%-22s ERROR %s: %s' % (os.path.basename(logfile)[-18:], type(e).__name__, e))
            continue
        if result is None:
            print('%-22s (no usable GPS)' % os.path.basename(logfile)[-18:])
            continue
        if not result['planned']:
            print('%-22s NO ROUTE PLANNED (states: %s)'
                   % (os.path.basename(logfile)[-18:], [s for _t, s in result['states']]))
            continue
        results.append(result)
        print('%-22s %7.0f %6.0f %4d   %6.2f %6.2f %6.2f   %6.1f %5.1f  %s'
               % (result['log'][-18:], result['route_m'], result['duration_s'],
                  result['junctions'], result['cross_med'], result['cross_p90'],
                  result['cross_max'], result['recovering_s'], result['lost_s'],
                  result['final_state']))
        if args.verbose:
            for t, reason, total in result['replans']:
                print('        plan t=%.1f %-40s %.0fm' % (t, reason, total))

    if args.html and results:
        write_html(results[0], args.html, [args.map])
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: expandtab sw=4 ts=4
