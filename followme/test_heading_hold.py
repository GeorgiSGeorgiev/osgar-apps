#!/usr/bin/python
"""
  Heading hold and post-avoidance heading return on open paving, CLOSED LOOP.

      python test_heading_hold.py

  The field cases (2026-09-28, CZU test10): on a plaza (145851) and in a
  parking lot (150145) the mask covered the whole picture, so once something
  turned Matty away from the road nothing turned it back - 25-35 deg off the
  mapped road for 25-40 s, 5-8 m across into a dead end. A log replay cannot
  test a fix for that (the replayed pose ignores the command), so this drives
  Matty's own kinematics - the simulator of test_turn_around.py - on paving
  20 m wide, with people as round obstacles the depth profile sees.

  Checked here:
    - after swerving round a person, the robot turns back to the heading it
      had before: a lane change, not a diagonal into the forecourt
    - out of a turn that left it 25 deg off the mapped road, it leans back
    - neither makes it pass closer to anybody than without it, and a second
      person in the way of the return is still avoided
    - with both switched off the old behaviour is untouched
"""
import math

from test_turn_around import Sim, hint_for, _feed_gps, check

PERSON_R = 0.25                  # m, a standing person as the depth camera sees one
FOV_DEG = 66.0                   # polar_profile_hfov_deg in the config
BINS = 9                         # free_space_bins


class OpenSim(Sim):
    """Sim on wide paving, with round obstacles and the 9-bin depth profile."""

    def __init__(self, people=(), app_overrides=None):
        super().__init__(road_halfwidth_m=20.0, app_overrides=app_overrides)
        self.people = list(people)
        self.closest = float('inf')

    def _ray(self, bearing):
        a = self.heading + bearing
        dx, dy = math.cos(a), math.sin(a)
        best = 15.0
        for px, py in self.people:
            ox, oy = px - self.x, py - self.y
            along = ox * dx + oy * dy
            if along <= 0:
                continue
            perp2 = ox * ox + oy * oy - along * along
            if perp2 < PERSON_R ** 2:
                best = min(best, along - math.sqrt(PERSON_R ** 2 - perp2))
        return round(best, 3)

    def _zones(self):
        return [self._ray(b) for b in (self.app.side_zone_bearing, 0.0, -self.app.side_zone_bearing)]

    def _profile(self):
        half = math.radians(FOV_DEG) / 2
        # bin 0 is the LEFT edge of the picture, i.e. the most anticlockwise bearing
        return [self._ray(half - (i + 0.5) * 2 * half / BINS) for i in range(BINS)]

    def step(self, hint=None, camera=True):
        self.app.time = self.app.time if self.app.time is not None else None
        import datetime
        self.app.time = datetime.timedelta(seconds=self.t)
        self.app.on_depth_profile(self._profile())
        out = super().step(hint, camera)
        for px, py in self.people:
            self.closest = min(self.closest, math.hypot(px - self.x, py - self.y) - PERSON_R)
        return out


OFF = {'heading_hold_enabled': False, 'avoid_return_enabled': False}
ON = {'heading_hold_enabled': True, 'avoid_return_enabled': True}


def run(people, overrides, start_heading_deg=0.0, seconds=40.0, road_deg=90.0):
    sim = OpenSim(people=people, app_overrides=overrides)
    sim.heading = math.radians(start_heading_deg)
    for _ in range(40):                       # learn the heading offset while standing still-ish
        _feed_gps(sim)
        sim.step(hint_for(road_deg))
    trace = []
    while sim.t < seconds:
        _feed_gps(sim)
        sim.step(hint_for(road_deg))
        trace.append((sim.t, sim.x, sim.y, math.degrees(sim.heading)))
    return sim, trace


def heading_after(trace, x_from):
    tail = [h for t, x, y, h in trace if x >= x_from]
    return None if not tail else sum(tail[-20:]) / len(tail[-20:])


def test_lane_change_after_a_person():
    print('== a person on the path, open paving, route straight on ==')
    people = [(6.0, -0.15)]                   # just right of the line: the depth terms swerve left
    off, t_off = run(people, OFF)
    on, t_on = run(people, ON)
    h_off, h_on = heading_after(t_off, 12.0), heading_after(t_on, 12.0)
    print('       off: heading %.1f deg at the end, %.1f m across, closest %.2f m' % (h_off, off.y, off.closest))
    print('       on:  heading %.1f deg at the end, %.1f m across, closest %.2f m' % (h_on, on.y, on.closest))
    check(abs(h_off) > 10.0, 'without it the swerve leaves Matty %.0f deg off the route for good' % h_off)
    check(abs(h_on) < 6.0, 'with it Matty ends parallel to the route again (%.1f deg) - a lane change' % h_on)
    check(abs(on.y) < abs(off.y), 'and ends nearer the line (%.1f m against %.1f m)' % (on.y, off.y))
    check(on.closest >= min(off.closest, 0.3), 'without passing closer to the person (%.2f m, off %.2f m)'
          % (on.closest, off.closest))


def test_leans_back_to_the_mapped_road():
    print('== out of a turn 25 deg off the mapped road, nothing in the way ==')
    off, t_off = run([], OFF, start_heading_deg=25.0, seconds=30.0)
    on, t_on = run([], ON, start_heading_deg=25.0, seconds=30.0)
    h_off, h_on = t_off[-1][3], t_on[-1][3]
    check(abs(h_off) > 20.0, 'without it the heading stays %.0f deg off' % h_off)
    check(abs(h_on) < 8.0, 'with it the heading comes back to %.1f deg of the road' % h_on)
    check(abs(on.y) < abs(off.y), 'and the drift across stops (%.1f m against %.1f m)' % (on.y, off.y))


def test_avoidance_still_wins():
    print('== a second person right where the return would go ==')
    people = [(6.0, -0.15), (11.0, 1.2)]
    off, _ = run(people, OFF)
    on, t_on = run(people, ON)
    print('       closest approach: off %.2f m, on %.2f m' % (off.closest, on.closest))
    check(on.closest > 0.2, 'nobody hit with the return on (closest %.2f m)' % on.closest)
    check(on.closest >= off.closest - 0.1, 'and no closer than with it off (%.2f vs %.2f m)'
          % (on.closest, off.closest))


def test_off_means_off():
    print('== switched off ==')
    a, ta = run([(6.0, -0.15)], OFF)
    b, tb = run([(6.0, -0.15)], {'heading_hold_enabled': False, 'avoid_return_enabled': False,
                                  'heading_hold_gain': 5.0})
    check(abs(a.y - b.y) < 1e-6 and abs(a.heading - b.heading) < 1e-9,
          'with both flags off the other settings change nothing')


def main():
    for test in (test_lane_change_after_a_person, test_leans_back_to_the_mapped_road,
                 test_avoidance_still_wins, test_off_means_off):
        test()
    print('\nALL HEADING HOLD TESTS PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

# vim: expandtab sw=4 ts=4
