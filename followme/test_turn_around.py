#!/usr/bin/python
"""
  The about-turn manoeuvre (State.TURN_AROUND), driven CLOSED LOOP.

      python test_turn_around.py

  Why a simulator and not a replay. A log replay feeds the app the pose
  that was actually recorded, so whatever the app commands, the heading
  does what it did on the day - which is precisely the thing a manoeuvre
  has to change. Replaying run 080142 through the new code shows the
  turn-around being entered at t=750, exactly where the real run missed
  its turn, and then "174 deg still to go" for nine legs, because the
  replayed robot never turned. That is a property of the replay, not of
  the manoeuvre.

  So this integrates Matty's own kinematics instead. The firmware's model
  (documented in the Matty driver and in matty_02_rl_sim_parameters.md):

      radius = (axle_distance / 2) / tan(joint / 2),   axle_distance 0.32 m

  which at the 45 deg joint limit is a 0.39 m pivot radius. Speed and
  joint angle are taken from what the app actually publishes on
  desired_steering, with a first-order lag on the joint so the test
  cannot pass by assuming an instant 45 deg deflection.

  Checked here:
    - a 143 deg about-turn (the one run 080142 missed) completes
    - it completes inside the width of a Stromovka footway
    - a 180 deg one completes too
    - the manoeuvre is refused where there is no room, and the robot
      carries on instead of wedging itself
    - an obstacle appearing inside stop_dist aborts it back to avoidance
"""
import datetime
import io
import json
import math
import sys

import numpy as np

from tulak_obstacle import TulakObstacle, State
from replay_osm_router import FakeBus

CONFIG = 'config/matty-tulak-osm.json'
AXLE_M = 0.32
JOINT_LAG = 0.35                # fraction of the remaining error per 0.1 s cycle


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print('   ok  %s' % message)


class Sim:
    """One app, a kinematic Matty, and a road of a given width."""

    def __init__(self, road_halfwidth_m=1.2, obstacle_at=None, app_overrides=None):
        modules = json.load(io.open(CONFIG, encoding='utf-8'))['robot']['modules']
        init = dict(modules['app']['init'], terminate_on_stop=False)
        init.update(app_overrides or {})
        self.bus = FakeBus()
        self.app = TulakObstacle(init, self.bus)
        self.app.verbose = False
        self.t = 0.0
        self.x = self.y = 0.0
        self.heading = 0.0          # odometry convention: 0 = +x, anticlockwise
        self.joint = 0.0
        self.half = road_halfwidth_m
        self.obstacle_at = obstacle_at
        self.max_off = 0.0          # how far off the road centre line it ever got
        self.travel = 0.0

    # --- what the robot is told about the world ---------------------
    def _zones(self):
        """left/centre/right clearance. The road is a straight corridor of
        half-width `half` along +x; anything past the edge is a wall."""
        out = []
        for bearing in (self.app.side_zone_bearing, 0.0, -self.app.side_zone_bearing):
            a = self.heading + bearing
            dist = 15.0
            for edge in (self.half, -self.half):
                dy = math.sin(a)
                if abs(dy) > 1e-6:
                    d = (edge - self.y) / dy
                    if 0 < d < dist:
                        dist = d
            if self.obstacle_at is not None:
                dx, dy = self.obstacle_at[0] - self.x, self.obstacle_at[1] - self.y
                r = math.hypot(dx, dy)
                if abs(math.atan2(dy, dx) - a) < 0.4 and r < dist:
                    dist = r
            out.append(round(min(dist, 15.0), 3))
        return out

    def _mask(self):
        """A mask showing the corridor, projected the way the camera would
        see it - so road_ahead_frac and the ground projection both get a
        consistent picture."""
        import road_geometry
        proj = road_geometry.Projection(320, 240)
        m = np.zeros((240, 320), np.uint8)
        for row in range(146, 240):
            x = proj.forward_of(row)
            if x > 25:
                continue
            scale = proj.lateral_scale(row)
            # the corridor in the ROBOT frame
            left = self.half - self.y
            right = -self.half - self.y
            ca, sa = math.cos(-self.heading), math.sin(-self.heading)
            # rotate the corridor edges into the robot frame at distance x
            yl = (left * ca - 0 * sa) - x * math.tan(self.heading)
            yr = (right * ca - 0 * sa) - x * math.tan(self.heading)
            c0 = int(round(proj.cx - max(yl, yr) / scale))
            c1 = int(round(proj.cx - min(yl, yr) / scale))
            m[row, max(0, c0):min(320, c1 + 1)] = 1
        return m

    def step(self, hint=None):
        app = self.app
        app.time = datetime.timedelta(seconds=self.t)
        app.on_nn_mask(self._mask())
        app.on_obstacle_zones(self._zones())
        if hint is not None:
            app.on_route_hint(hint)
        app.on_pose2d([int(self.x * 1000), int(self.y * 1000),
                       int(math.degrees(self.heading) * 100)])
        cmd = self.bus.published.get('desired_steering') or (0, 0)
        speed, joint_cmd = cmd[0] / 1000.0, math.radians(cmd[1] / 100.0)
        joint_cmd = max(-math.radians(45), min(math.radians(45), joint_cmd))
        self.joint += JOINT_LAG * (joint_cmd - self.joint)
        dt = 0.1
        ds = speed * dt
        if abs(self.joint) > 1e-4:
            radius = (AXLE_M / 2) / math.tan(self.joint / 2)
            dtheta = ds / radius
        else:
            dtheta = 0.0
        self.x += ds * math.cos(self.heading + dtheta / 2)
        self.y += ds * math.sin(self.heading + dtheta / 2)
        self.heading += dtheta
        self.travel += abs(ds)
        self.max_off = max(self.max_off, abs(self.y))
        self.t += dt
        return speed, self.joint


def hint_for(road_bearing_deg):
    """A route_hint whose road runs at that compass bearing."""
    return {'state': 'following', 'mode': 'corridor', 'authority': 0.45,
            'cross_track_m': 0.0, 'off_road_m': 0.0, 'road_halfwidth_m': 1.2,
            'road_bearing_deg': road_bearing_deg, 'lat': None, 'lon': None,
            'hold': None, 'arrived': False, 'off_route': False, 'junction_m': 200.0,
            'remaining_m': 200.0, 'gps_age_sec': 0.1, 'speed_limit': None}


def drive(sim, want_deg, seconds=90.0, halfwidth=None):
    """Run the app with a route that runs `want_deg` away from the robot's
    initial heading, and report what happened."""
    # the app's heading comes from the odometry+GPS estimator; feed it a
    # consistent GPS track so _current_heading returns 'odo+gps'
    start = sim.heading
    entered = False
    done_at = None
    hint = hint_for((90.0 - math.degrees(start) - want_deg) % 360.0)
    while sim.t < seconds:
        _feed_gps(sim)
        sim.step(hint)
        if sim.app.state == State.TURN_AROUND:
            entered = True
        elif entered and done_at is None:
            done_at = sim.t
            break
    turned = math.degrees(sim.heading - start)
    return dict(entered=entered, done_at=done_at, turned=turned,
                max_off=sim.max_off, state=sim.app.state, t=sim.t)


def _feed_gps(sim):
    """Keep the heading estimator fed: a fix that agrees with odometry, so
    _current_heading() reports 'odo+gps' with zero offset."""
    app = sim.app
    if app.heading_est is None:
        return
    lat = 50.1 + sim.y / 111320.0
    lon = 14.42 + sim.x / (111320.0 * math.cos(math.radians(50.1)))
    # odometry x is east here, so the estimator learns offset = pi/2
    app.heading_est.update_fix(sim.t, lat, lon)


def test_completes_the_turn_080142_missed():
    print('== the 143 deg turn run 080142 armed and abandoned ==')
    sim = Sim(road_halfwidth_m=1.2)
    for _ in range(30):                       # drive straight, so the estimator learns
        _feed_gps(sim)
        sim.step(hint_for(90.0))
    r = drive(sim, 143.0)
    check(r['entered'], 'the about-turn is entered')
    check(r['done_at'] is not None, 'and it finishes (%.0f deg turned in %.1f s)'
          % (r['turned'], r['t']))
    check(abs(abs(r['turned']) - 143.0) < 30.0,
          'it ends within 30 deg of the 143 deg asked for (%.0f deg)' % r['turned'])
    check(r['max_off'] <= 1.2, 'and never leaves the 2.4 m path (worst %.2f m from the centre)'
          % r['max_off'])


def test_completes_a_full_about_turn():
    print('== 180 deg ==')
    sim = Sim(road_halfwidth_m=1.2)
    for _ in range(30):
        _feed_gps(sim)
        sim.step(hint_for(90.0))
    r = drive(sim, 178.0)
    check(r['entered'] and r['done_at'] is not None,
          'a full about-turn finishes too (%.0f deg in %.1f s)' % (r['turned'], r['t']))
    check(r['max_off'] <= 1.2, 'inside the path (worst %.2f m)' % r['max_off'])


def test_refuses_without_room():
    print('== no room ==')
    sim = Sim(road_halfwidth_m=0.45)          # a 0.9 m track
    for _ in range(30):
        _feed_gps(sim)
        sim.step(hint_for(90.0))
    r = drive(sim, 170.0, seconds=25.0)
    check(not r['entered'], 'on a 0.9 m track the manoeuvre is refused')
    check(r['max_off'] <= 0.45, 'and the robot stays on it (worst %.2f m)' % r['max_off'])


def test_obstacle_aborts():
    print('== obstacle during the turn ==')
    sim = Sim(road_halfwidth_m=1.5)
    for _ in range(30):
        _feed_gps(sim)
        sim.step(hint_for(90.0))
    hint = hint_for((90.0 - math.degrees(sim.heading) - 170.0) % 360.0)
    entered = False
    for _ in range(600):
        _feed_gps(sim)
        sim.step(hint)
        if sim.app.state == State.TURN_AROUND and not entered:
            entered = True
            sim.obstacle_at = (sim.x + 0.35 * math.cos(sim.heading),
                               sim.y + 0.35 * math.sin(sim.heading))
        if entered and sim.app.state != State.TURN_AROUND:
            break
    check(entered, 'the manoeuvre starts')
    check(sim.app.state != State.TURN_AROUND,
          'and something inside stop_dist takes it back to avoidance (%s)' % sim.app.state)


def main():
    for test in (test_completes_the_turn_080142_missed, test_completes_a_full_about_turn,
                 test_refuses_without_room, test_obstacle_aborts):
        test()
    print('\nALL TURN-AROUND TESTS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: expandtab sw=4 ts=4
