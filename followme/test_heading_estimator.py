#!/usr/bin/python
"""
  The odometry+GPS heading estimator against reflected GPS signals.

      python test_heading_estimator.py

  The failure this guards against happened on 2026-09-28 (CZU test10,
  144532 t=499-518): beside a housing block the fixes crept ~5 m sideways
  while the wheels drove dead straight. Every chord still had the right
  LENGTH, so the estimator took them, and the heading ended ~100 deg wrong -
  enough for the app to turn Matty around on an empty road. The receiver's
  Doppler course stayed right throughout.

  Checked here, on a synthetic drive at Matty's 0.5 m/s with 1 Hz fixes:
    - the creep reproduces the failure with the previous settings
    - with the reflected-signal defences on, the heading holds
    - a genuinely wrong offset is still corrected: at once when the
      receiver's course agrees with the chords, by consensus without one
    - clean driving, straight and turning, is unaffected
"""
import math

from heading_estimator import OdoGpsHeading

LAT0, LON0 = 50.1, 14.42
M_LON = 111320.0 * math.cos(math.radians(LAT0))
ROBUST = dict(max_baseline_sec=25.0, gate_deg=25.0, cog_agree_deg=25.0,
              consensus_n=3, consensus_spread_deg=15.0, jump_m=1.5, cog_veto_deg=15.0)


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print('   ok  %s' % message)


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class Drive:
    """A robot whose odometry frame is rotated `frame_deg` from east, driving
    at 0.5 m/s; `bearing_deg` is its true compass heading."""

    def __init__(self, est, bearing_deg=0.0, frame_deg=30.0):
        self.est = est
        self.t = 0.0
        self.e = self.n = 0.0              # true position, east/north
        self.bearing = math.radians(bearing_deg)
        self.frame = math.radians(frame_deg)
        self.gps_err = [0.0, 0.0]          # what the fix gets wrong, east/north

    def odo(self):
        # odometry = the true track in a frame rotated by `frame`
        ang = math.pi / 2 - self.bearing  # math angle of the heading, from east
        x = self.e * math.cos(-self.frame) - self.n * math.sin(-self.frame)
        y = self.e * math.sin(-self.frame) + self.n * math.cos(-self.frame)
        return x, y, ang - self.frame

    def run(self, seconds, turn_deg_s=0.0, creep=None, course=True, course_err_deg=0.0):
        """creep = (east, north) m/s added to the GPS error"""
        steps = int(round(seconds / 0.1))
        for i in range(steps):
            self.bearing += math.radians(turn_deg_s) * 0.1
            self.e += 0.05 * math.sin(self.bearing)
            self.n += 0.05 * math.cos(self.bearing)
            self.t += 0.1
            if creep is not None:
                self.gps_err[0] += creep[0] * 0.1
                self.gps_err[1] += creep[1] * 0.1
            x, y, h = self.odo()
            self.est.update_pose(self.t, x, y, h)
            if i % 10 == 9:                 # 1 Hz fixes
                lat = LAT0 + (self.n + self.gps_err[1]) / 111320.0
                lon = LON0 + (self.e + self.gps_err[0]) / M_LON
                cog = (math.degrees(self.bearing) + course_err_deg) % 360.0 if course else None
                self.est.update_fix(self.t, lat, lon, sog=0.5, cog=cog)

    def error_deg(self):
        h = self.est.heading()
        return None if h is None else wrap_deg(math.degrees(h - self.bearing))


def test_creep_breaks_the_old_settings():
    print('== sideways creep, previous settings ==')
    d = Drive(OdoGpsHeading())
    d.run(30.0)
    check(abs(d.error_deg()) < 3.0, 'learned on a clean straight (%.1f deg)' % d.error_deg())
    d.run(10.0, creep=(0.5, -0.5))          # fixes stall and slide east while the robot drives north
    check(abs(d.error_deg()) > 45.0,
          'the creep drags the heading %.0f deg off - the 144532 failure, reproduced' % d.error_deg())


def test_creep_is_rejected():
    print('== sideways creep, reflected-signal defences on ==')
    for course_err, limit in ((0.0, 10.0), (15.0, 30.0)):
        d = Drive(OdoGpsHeading(**ROBUST))
        d.run(30.0)
        worst = 0.0
        for _ in range(10):
            d.run(1.0, creep=(0.5, -0.5), course_err_deg=course_err)
            worst = max(worst, abs(d.error_deg()))
        # 144532 had the course itself 10-20 deg off near the building; the
        # estimator cannot beat its referee, but it must stay far from the
        # 110 deg an about-turn needs
        check(worst < limit, 'with the course %.0f deg off, the heading holds through 10 s of creep '
                             '(worst %.1f deg, limit %.0f)' % (course_err, worst, limit))
        check(d.est.rejected > 0, '%d chord(s) turned away' % d.est.rejected)
        d.run(20.0)
        check(abs(d.error_deg()) < 5.0, 'and it is right again once the fixes settle (%.1f deg)'
              % d.error_deg())


def test_creep_without_a_course():
    """The honest limit: with no course at all, a creep smooth enough to stay
    under jump_m and inside the gate on every chord looks exactly like a real
    offset change, and it moves the heading. What must not happen then is an
    about-turn on it - the app asks unsettled_deg() first (uturn_settle_*)."""
    print('== sideways creep, no course from the receiver ==')
    d = Drive(OdoGpsHeading(**ROBUST))
    d.run(30.0)
    worst = moved = 0.0
    for _ in range(10):
        d.run(1.0, creep=(0.5, -0.5), course=False)
        worst = max(worst, abs(d.error_deg()))
        moved = max(moved, d.est.unsettled_deg(8.0))
    print('       (the heading does move: worst %.0f deg)' % worst)
    check(worst < 30.0 or moved > 30.0,
          'whenever it moves past 30 deg it reports itself unsettled (%.0f deg over 8 m), '
          'which blocks an about-turn' % moved)
    d.run(30.0)
    check(abs(d.error_deg()) < 5.0, 'and it recovers once the fixes settle (%.1f deg)' % d.error_deg())


def test_wrong_offset_is_corrected_with_a_course():
    print('== an offset that really is wrong, course available ==')
    d = Drive(OdoGpsHeading(**ROBUST))
    d.run(30.0)
    d.est.offset += math.radians(80.0)     # e.g. odometry heading slipped in a pivot
    check(abs(d.error_deg()) > 70.0, 'start %.0f deg wrong' % d.error_deg())
    d.run(6.0)
    check(abs(d.error_deg()) < 10.0, 'the course confirms the chords and it snaps back (%.1f deg)'
          % d.error_deg())


def test_wrong_offset_is_corrected_by_consensus():
    print('== an offset that really is wrong, no course ==')
    d = Drive(OdoGpsHeading(**ROBUST))
    d.run(30.0, course=False)
    d.est.offset += math.radians(80.0)
    d.run(12.0, course=False)
    check(abs(d.error_deg()) < 10.0, 'three agreeing chords re-anchor it (%.1f deg)' % d.error_deg())


def test_clean_driving_unchanged():
    print('== clean driving: straight, then a 90 deg turn ==')
    old, new = Drive(OdoGpsHeading()), Drive(OdoGpsHeading(**ROBUST))
    for d in (old, new):
        d.run(30.0)
        d.run(9.0, turn_deg_s=10.0)
        d.run(20.0)
    check(abs(new.error_deg()) < 3.0 and abs(old.error_deg()) < 3.0,
          'both settings end within 3 deg (previous %.1f, new %.1f)' % (old.error_deg(), new.error_deg()))
    check(new.est.unsettled_deg(8.0) < 5.0, 'and the new one reports a settled heading (%.1f deg)'
          % new.est.unsettled_deg(8.0))


def main():
    for test in (test_creep_breaks_the_old_settings, test_creep_is_rejected,
                 test_creep_without_a_course, test_wrong_offset_is_corrected_with_a_course,
                 test_wrong_offset_is_corrected_by_consensus, test_clean_driving_unchanged):
        test()
    print('\nALL HEADING ESTIMATOR TESTS PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

# vim: expandtab sw=4 ts=4
