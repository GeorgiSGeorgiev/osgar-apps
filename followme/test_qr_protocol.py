#!/usr/bin/python
"""
  QR command protocol - the run-mode state machine, driven end to end
  through the REAL OSMRouter and TulakObstacle classes.

  Needs no log, no robot and no network: it feeds synthetic QR codes,
  GPS fixes and pose2d cycles straight into the two nodes and checks what
  comes out. Run it after touching either file.

      python test_qr_protocol.py

  The protocol it pins down, all matched case-insensitively:

      (release the emergency stop)   -> WAITING, stopped
      QR "start" / "go"              -> FREE: road mask + depth only,
                                        no route, no GPS guidance
      QR <lat, lon>                  -> plan a route and FOLLOW it
      QR "stop" / "abort" / "cancel" -> forget the target, back to WAITING
      (press + release the e-stop)   -> back to WAITING, whatever was
                                        happening before

  Two config gates, both defaulting to the full competition behaviour:

      enable_gps_routing=False   coordinate codes ignored; start/stop still
                                 work. "Just drive on the road" without
                                 rewiring anything.
      wait_for_start_qr=False    no start gate at all - boots straight into
                                 FREE, i.e. the behaviour from before this
                                 protocol existed.
"""
import datetime
import io
import json
import math
import sys

from osm_router import OSMRouter, RouteState
from replay_osm_router import FakeBus, router_config
from tulak_obstacle import TulakObstacle

CONFIG = 'config/matty-tulak-osm.json'
START_FIX = (50.105834, 14.427345)          # a real Stromovka path
TARGET = '50.106589, 14.417428'


class Harness:
    """One router + one app, wired the way the config wires them, with a
    clock that only advances when told to."""

    def __init__(self, app_overrides=None, **router_overrides):
        modules = json.load(io.open(CONFIG, encoding='utf-8'))['robot']['modules']
        router_init = dict(modules['osm_router']['init'], log_interval_sec=1e9)
        router_init.update(router_overrides)
        # terminate_on_stop is forced False here regardless of what the
        # config says, because that is the only setting under which there
        # IS a release to observe - with True the app raises
        # EmergencyStopException on the press and the run ends, which is a
        # perfectly good workflow (you restart, and land in WAITING) but
        # not one this test can drive.
        app_init = dict(modules['app']['init'], terminate_on_stop=False)
        app_init.update(app_overrides or {})
        self.rbus, self.abus = FakeBus(), FakeBus()
        self.router = OSMRouter(router_init, self.rbus)
        self.app = TulakObstacle(app_init, self.abus)
        self.t = 0.0

    def _now(self):
        return datetime.timedelta(seconds=self.t)

    def tick(self, cycles=3):
        """Advance the 10Hz control loop, carrying route_hint across as the
        real bus would."""
        for _ in range(cycles):
            self.t += 0.1
            self.router.time = self.app.time = self._now()
            self.router.on_pose2d([0, 0, 0])
            self.app.on_route_hint(self.rbus.published['route_hint'])

    def fix(self, lat=START_FIX[0], lon=START_FIX[1]):
        self.router.time = self._now()
        self.router.on_nmea_data({'lat': lat, 'lon': lon, 'lat_dir': 'N', 'lon_dir': 'E',
                                   'quality': 1, 'hdop': 0.8, 'sats': 12})

    def qr(self, text):
        self.router.time = self._now()
        self.router.on_qr_code(text)

    def estop(self, engaged):
        self.router.time = self.app.time = self._now()
        self.router.on_emergency_stop(engaged)
        self.app.on_emergency_stop(engaged)

    @property
    def hint(self):
        return self.rbus.published['route_hint']


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print('   ok  %s' % message)


def test_protocol():
    print('== full protocol (defaults) ==')
    h = Harness()
    h.fix()
    h.tick()
    check(h.router.state == RouteState.WAITING and h.hint['hold'] == 'creep',
          'boots into WAITING and holds')

    h.qr('Start')
    h.tick()
    check(h.router.state == RouteState.FREE, '"Start" (mixed case) enters FREE')
    check(h.hint['hold'] is None, 'FREE does not hold - the robot drives')
    check(h.app.target_lat is None and h.app.route_hold is None,
          'FREE gives the follower no target and no hold')
    check(h.hint['cross_track_m'] is None,
          'FREE withholds cross-track, so no stale corridor bias')
    check(abs(h.app._route_corridor_bias()) < 1e-9, 'FREE produces zero route bias')

    plans = h.rbus.counts.get('route_plan', 0)
    h.qr('start')
    h.tick()
    check(h.rbus.counts.get('route_plan', 0) == plans and h.router.state == RouteState.FREE,
          're-reading "start" while already FREE is a no-op')

    h.qr('STOP')
    h.tick()
    check(h.router.state == RouteState.WAITING and h.hint['hold'] == 'creep',
          '"STOP" returns to WAITING and holds')

    h.qr(TARGET)
    h.tick()
    check(h.router.state == RouteState.FOLLOWING, 'coordinates plan and follow')
    check(h.app.target_lat is not None and h.hint['hold'] is None,
          'following hands the follower an aim point and releases the hold')

    h.qr('stop')
    h.tick()
    check(h.router.goal_ll is None and h.app.target_lat is None,
          '"stop" erases the target')

    h.qr(TARGET)
    h.tick()
    check(h.router.state == RouteState.FOLLOWING,
          'the SAME coordinates after a stop re-plan (not deduped away)')

    h.qr('Go')
    h.tick()
    check(h.router.state == RouteState.FREE and h.router.goal_ll is None,
          '"Go" while following drops the route')

    h.estop(True)
    check(h.app.emergency_stop_active,
          'e-stop press is held by the app (terminate_on_stop=False)')
    h.estop(False)
    h.tick()
    check(h.router.state == RouteState.WAITING and not h.app.emergency_stop_active,
          'e-stop release resets to WAITING')

    state = h.router.state
    h.qr('https://robotika.cz')
    h.tick()
    check(h.router.state == state, 'an unrecognised code changes nothing')

    h.estop(False)
    h.tick()
    check(h.router.state == RouteState.WAITING,
          'a release with no press seen (matty.py sends one at boot) is ignored')


def test_estop_terminates_by_default():
    """With terminate_on_stop=True - what config/matty-tulak-osm.json
    currently ships - the press ends the run instead of resetting the
    mode. Both are legitimate; this pins which one you get."""
    print('== terminate_on_stop=True ends the run on the press ==')
    from osgar.exceptions import EmergencyStopException
    h = Harness(app_overrides={'terminate_on_stop': True})
    h.fix()
    h.tick()
    raised = False
    try:
        h.estop(True)
    except EmergencyStopException:
        raised = True
    check(raised, 'the press raises EmergencyStopException, ending the run')
    check(h.app.emergency_stop_active, 'the hold flag is still set before it raises')


def test_gps_disabled():
    print('== enable_gps_routing=False ==')
    h = Harness(enable_gps_routing=False)
    h.fix()
    h.tick()
    h.qr(TARGET)
    h.tick()
    check(h.router.state == RouteState.WAITING and h.router.goal_ll is None,
          'coordinates are ignored')
    h.qr('go')
    h.tick()
    check(h.router.state == RouteState.FREE, 'start/stop control still works')


def test_no_start_gate():
    print('== wait_for_start_qr=False ==')
    h = Harness(wait_for_start_qr=False)
    h.tick()
    check(h.router.state == RouteState.FREE and h.hint['hold'] is None,
          'boots straight into FREE, no QR needed')
    h.fix()
    h.qr(TARGET)
    h.tick()
    check(h.router.state == RouteState.FOLLOWING,
          'a coordinate code still starts a route')
    h.qr('stop')
    h.tick()
    check(h.router.state == RouteState.FREE,
          '"stop" returns to FREE rather than parking, with no start gate')


def test_waiting_never_moves():
    """A QR code held close to the camera IS an obstacle. While waiting,
    Matty must sit still anyway.

    Field-reported 2026-09-04: holding the code up close made it back
    away. The hold was applied as min(speed, 0) on the finished command,
    which does nothing to a negative speed, so the avoidance state machine
    kept running underneath and reversed."""
    print('== WAITING ignores obstacles (but still senses them) ==')
    h = Harness()
    h.fix()
    h.tick()

    speeds, states = [], set()
    for i in range(80):
        h.t += 0.1
        h.app.time = h.router.time = datetime.timedelta(seconds=h.t)
        # something 0.15 m dead ahead and on both flanks - well inside
        # every stop/turn threshold there is
        h.app.on_obstacle_zones([0.15, 0.15, 0.15])
        h.app.on_depth_profile([0.15] * 9)
        h.router.on_pose2d([0, 0, 0])
        h.app.on_route_hint(h.rbus.published['route_hint'])
        h.app.on_pose2d([0, 0, 0])
        cmd = h.abus.published.get('desired_steering')
        if cmd:
            speeds.append(cmd[0] / 1000.0)
        states.add(h.app.state)

    check(speeds and all(abs(v) < 1e-9 for v in speeds),
          'commanded speed is exactly 0 for all %d cycles (min %.3f, max %.3f)'
          % (len(speeds), min(speeds), max(speeds)))
    check(not any(v < 0 for v in speeds), 'never reverses')
    check(states == {__import__('tulak_obstacle').State.DRIVE},
          'the avoidance state machine never leaves DRIVE (states seen: %s)'
          % sorted(s.name for s in states))
    check(h.app.turn_streak > 0 and h.app.stop_streak > 0,
          'but the depth data IS still processed - streaks confirmed '
          '(turn=%d stop=%d), so the reaction is suppressed, not the sensing'
          % (h.app.turn_streak, h.app.stop_streak))

    # ...and the moment it is released, avoidance is live again
    h.qr('start')
    h.tick(1)
    h.app.on_pose2d([0, 0, 0])
    check(h.app.route_hold is None, 'releasing the hold re-arms normal driving')


def test_gps_loss_ladder():
    """What happens when GPS stops. Measured on the recorded runs: 25
    dropouts longer than 2.5 s, median 3.5 s, NONE longer than 4.0 s, with
    odometry tracking GPS displacement to a median ratio of 1.04 over 10 s
    windows - so a short dropout is worth riding out and a long one is
    not."""
    print('== GPS loss degrades in stages ==')
    h = Harness()
    h.fix(50.130032, 14.379960)
    h.qr('50.1290345, 14.3770581')
    h.tick()
    check(h.router.state == RouteState.FOLLOWING, 'following with a fresh fix')
    full_authority = h.hint['authority']

    def advance_to(age):
        while (h.hint.get('gps_age_sec') or 0) < age:
            h.tick(1)
        return h.hint

    hint = advance_to(5)
    check(hint['authority'] == full_authority and hint['cross_track_m'] is not None,
          'a short dropout (5 s) changes nothing - odometry carries the route')

    hint = advance_to(25)
    check(hint['state'] == RouteState.FOLLOWING and hint['authority'] < full_authority,
          'past the dead-reckoning window authority starts fading (%.2f -> %.2f)'
          % (full_authority, hint['authority']))
    check(hint['cross_track_m'] is None,
          'cross-track is withheld, not frozen - a stale bias is a constant '
          'steering error with no feedback')
    check(hint['speed_limit'] is not None, 'and speed is capped while degraded')

    hint = advance_to(50)
    check(h.router.state == RouteState.GPS_LOST, 'past gps_lost_sec it gives up on the route')
    check(hint['lat'] is None and hint['hold'] is None and hint['mode'] == 'free',
          'and degrades to road following + avoidance, NOT to a hold')
    check(h.router.goal_ll is not None, 'the target is kept, not forgotten')

    h.fix(50.130032, 14.379960)
    h.tick()
    check(h.router.state == RouteState.FOLLOWING and h.router.route is not None,
          'a returning fix re-plans from wherever the robot actually is')


def test_articulated_steering_guard():
    """The pole incident: passed on the left, the camera lost it, the right
    zone read clear, Matty turned hard right and the rear-right wheel caught
    it and began to climb - loading the single central joint."""
    print('== articulated steering guard (both sides) ==')
    from tulak_obstacle import TulakObstacle
    app = TulakObstacle({}, FakeBus())

    app.left_dist, app.right_dist = 99.0, 1.0        # ~0.34 m laterally
    inside = app._articulated_steering_limit(-1)      # steering RIGHT, obstacle RIGHT
    check(inside is not None and math.degrees(inside) < 35,
          'turning TOWARD a 1.0 m flank range (0.34 m lateral) is capped '
          '(%.0f deg, not 45)' % math.degrees(inside))
    check(abs(app._lateral_clearance(-1) - 1.0 * math.sin(app.side_zone_bearing)) < 1e-9,
          'flank RANGE is converted to lateral offset, not used raw')

    app.left_dist, app.right_dist = 1.0, 99.0
    outside = app._articulated_steering_limit(-1)     # steering RIGHT, obstacle LEFT
    check(outside is not None, 'turning AWAY from the same flank is also capped')

    mirrored = app._articulated_steering_limit(+1)
    check(abs(mirrored - inside) < 1e-9,
          'the guard is symmetric: same clearance mirrored gives the same limit')

    app.left_dist = app.right_dist = 99.0
    check(app._articulated_steering_limit(+1) is None, 'open ground is not limited at all')

    # the memory is what makes it work - the live zone loses the obstacle
    app.left_dist = app.right_dist = 99.0
    app._flank_history[-1].append((datetime.timedelta(0), (0.0, 0.0), 1.0))
    remembered = app._articulated_steering_limit(-1)
    check(remembered is not None and math.degrees(remembered) < 35,
          'a remembered flank the camera can no longer see still limits the turn (%.0f deg)'
          % math.degrees(remembered))

    import random
    random.seed(11)
    violations = 0
    for _ in range(4000):
        app.left_dist = random.choice([None, random.uniform(0.2, 3.0)])
        app.right_dist = random.choice([None, random.uniform(0.2, 3.0)])
        app._flank_history = {1: [], -1: []}
        wanted = random.uniform(-0.8, 0.8)
        got = app._limit_tail_swing(wanted)
        if abs(got) > abs(wanted) + 1e-9 or (got and math.copysign(1, got) != math.copysign(1, wanted)):
            violations += 1
    check(violations == 0,
          'over 4000 random states it only ever reduces magnitude, never flips sign')


def test_coordinate_formats():
    """Every format a QR code has actually been generated in for this
    robot. The N/E-suffixed decimal one is here because it silently
    returned None in the field (2026-09-04 CZU runs 174623 / 174729 /
    175216): the codes decoded perfectly and were then discarded as "not
    coordinates", so three targets were shown and ignored."""
    print('== coordinate formats ==')
    from tulak_obstacle import parse_gps_qr
    cases = [
        ('50.1290345, 14.3770581', (50.1290345, 14.3770581)),
        ('50.1299558N, 14.3793860E', (50.1299558, 14.3793860)),      # the field failure
        ('N50.1299558, E14.3793860', (50.1299558, 14.3793860)),
        ('14.3793860E, 50.1299558N', (50.1299558, 14.3793860)),      # letters resolve the order
        ('50.1299558S, 14.3793860W', (-50.1299558, -14.3793860)),
        ('-33.8688, 151.2093', (-33.8688, 151.2093)),
        ('https://maps.google.com/?q=50.087451,14.420671', (50.087451, 14.420671)),
        (u'50°5\'14.8"N 14°25\'14.4"E', (50.0874444, 14.4206667)),
        ('start', None), ('stop', None), ('https://robotika.cz', None), ('', None),
        ('991.5, 8888.2', None),                                      # out of range
    ]
    for text, expect in cases:
        got = parse_gps_qr(text)
        if expect is None:
            check(got is None, '%r -> rejected' % text)
        else:
            check(got is not None
                  and abs(got[0] - expect[0]) < 1e-6 and abs(got[1] - expect[1]) < 1e-6,
                  '%r -> %.6f, %.6f' % (text, expect[0], expect[1]))


CZU_FIX = (50.130032, 14.379960)        # first fix of run 260904_173025
CZU_TARGET = '50.1290345, 14.3770581'   # the code shown that day


def test_off_map_refusal():
    """The 2026-09-04 CZU failure, reproduced exactly and then refused.

    The robot ran at CZU Suchdol with maps/stromovka.json loaded. Both its
    own position and the target snapped to the same corner of the
    Stromovka graph 2.5km away, so the route came out 0.0m long and the
    router reported ARRIVED and held - which from the operator's side was
    "it read the coordinates and then refused to move"."""
    print('== off-map planning is refused ==')
    h = Harness(map_file='maps/stromovka.json')
    h.fix(*CZU_FIX)
    h.qr(CZU_TARGET)
    h.tick()
    check(h.router.state == RouteState.FAILED,
          'planning 2.5km outside the map fails instead of reporting arrival')
    check(not h.hint['arrived'] and h.hint['hold'] == 'stop',
          'it holds without ever claiming to have arrived')
    check(h.hint.get('reason') and 'OUTSIDE the loaded map area' in h.hint['reason'],
          'the reason names the real cause, and is in the published hint (so: in the log)')


def test_map_auto_selection():
    print('== map chosen from the first fix ==')
    both = ['maps/stromovka.json', 'maps/czu_suchdol.json']

    h = Harness(map_file=both)
    h.fix(*CZU_FIX)
    h.qr(CZU_TARGET)
    h.tick()
    check('czu' in h.router.map_path, 'at CZU it selects the CZU map')
    check(h.router.state == RouteState.FOLLOWING and h.router.route.total > 50,
          'and the same code that failed before now plans a real route')

    h = Harness(map_file=both)
    h.fix(*START_FIX)
    h.qr(TARGET)
    h.tick()
    check('stromovka' in h.router.map_path, 'at Stromovka it selects the Stromovka map')
    check(h.router.state == RouteState.FOLLOWING, 'and plans there too')

    h = Harness(map_file=both)
    h.fix(49.195, 16.606)          # Brno - covered by neither
    h.qr('49.196, 16.607')
    h.tick()
    check(h.router.state == RouteState.FAILED,
          'somewhere no map covers still refuses rather than guessing')


def main():
    for test in (test_protocol, test_estop_terminates_by_default,
                 test_waiting_never_moves, test_gps_loss_ladder,
                 test_articulated_steering_guard,
                 test_gps_disabled, test_no_start_gate,
                 test_coordinate_formats, test_off_map_refusal, test_map_auto_selection):
        test()
        print()
    print('ALL QR PROTOCOL TESTS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: expandtab sw=4 ts=4
