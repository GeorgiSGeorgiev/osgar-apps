"""
  Heading without the magnetometer: odometry heading plus an offset learned
  from GPS displacement.

  Why. The ESP32 compass on Matty carries a heading-dependent error far larger
  than its configured hard-iron correction (10.3 deg). Measured against GPS
  course on straight, forward, 2.5-6 m fix baselines:

      2026-09-14 (4 logs, 305 baselines):  error = -5 + 32*cos(course - 39)
                                            |error| p50 20 deg, p90 40 deg
      2026-09-15 (10 logs, 1299 baselines): error = -1 + 44*cos(course - 32)
                                            |error| p50 34 deg, p90 55 deg

  i.e. +47 deg when driving north and -41 deg driving south-west. Every
  consumer of that heading inherited it: the junction arrow's "turn done"
  rule released a right turn 10.8 m BEFORE the junction because the compass
  already read the exit bearing (171614 t=73.5), the road-mask trust was cut
  on 47% of driving cycles, and the router latched "misaligned - treat as a
  junction" and slowed down.

  What survives. Odometry heading is smooth and turns correctly: through the
  two hairpins on 09-15 it turned 116 and 124 deg where GPS course turned 123
  and 129 deg (the compass read 194 and 84). It only drifts slowly (a few
  tenths of a degree per metre), so the rotation between the odometry frame
  and true north can be learned by comparing the GPS displacement over a few
  metres with the odometry displacement over the same interval ("chords").
  The two chords are the same vector in two frames whatever the path shape,
  so this needs no straight driving - the straight-only version learned
  nothing at all in 165145, where Matty weaved 9-27 deg at 0.3 m/s - and a
  constant GPS bias cancels out of a displacement.

  Scored out of sample (each straight baseline compared with the heading the
  estimator reported BEFORE learning from it), alpha 0.5, 4 m chords:
      09-15: |error| p50 1.4 deg, p90 5.2 deg, p99 22 deg (compass p50 33, p90 54)
      09-14: |error| p50 1.4 deg, p90 3.9 deg             (compass p50 19, p90 40)

  Conventions: odometry heading is OSGAR's (radians, 0 = +x, anticlockwise);
  the returned heading is a compass bearing (radians, 0 = north, clockwise),
  the convention of initial_bearing() and the router's road bearings.
"""
import math


def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class OdoGpsHeading:
    def __init__(self, alpha=0.5, min_baseline_m=4.0, max_baseline_sec=15.0, valid_travel_m=60.0):
        # weight of each new chord in the running offset (circular EMA)
        self.alpha = alpha
        # shortest GPS displacement a direction is taken from - below ~3 m a
        # 1 m fix jitter is already 20 deg
        self.min_baseline_m = min_baseline_m
        self.max_baseline_sec = max_baseline_sec
        # odometry travel after the last learned chord beyond which the
        # offset is no longer trusted (drift) - heading() returns None
        self.valid_travel_m = valid_travel_m
        self.offset = None              # compass bearing = offset - odometry angle
        self.samples = 0
        self.last_residual = None       # diagnostics: new chord minus the offset it replaced
        self._pose = []                 # (time, x, y, odo_heading, travel)
        self._fixes = []                # (time, east, north)
        self._travel = 0.0
        self._travel_at_update = None
        self._origin = None
        self._odo_heading = None

    # --- inputs -------------------------------------------------------
    def update_pose(self, time_sec, x, y, odo_heading):
        if self._pose:
            _, px, py, _, _ = self._pose[-1]
            self._travel += math.hypot(x - px, y - py)
        self._odo_heading = odo_heading
        self._pose.append((time_sec, x, y, odo_heading, self._travel))
        while self._pose and time_sec - self._pose[0][0] > self.max_baseline_sec + 2.0:
            self._pose.pop(0)

    def update_fix(self, time_sec, lat, lon):
        if self._origin is None:
            self._origin = (lat, lon)
        east = (lon - self._origin[1]) * 111320.0 * math.cos(math.radians(self._origin[0]))
        north = (lat - self._origin[0]) * 111320.0
        self._fixes.append((time_sec, east, north))
        while self._fixes and time_sec - self._fixes[0][0] > self.max_baseline_sec:
            self._fixes.pop(0)
        self._learn()

    # --- output -------------------------------------------------------
    def heading(self):
        """Compass bearing (radians) or None while the offset is unknown or
        stale."""
        if self.offset is None or self._odo_heading is None:
            return None
        if self._travel_at_update is not None and self._travel - self._travel_at_update > self.valid_travel_m:
            return None
        return normalize_angle(self.offset - self._odo_heading) % (2 * math.pi)

    # --- learning -----------------------------------------------------
    def _learn(self):
        if len(self._fixes) < 2 or len(self._pose) < 4:
            return
        t1, e1, n1 = self._fixes[-1]
        for t0, e0, n0 in reversed(self._fixes[:-1]):
            base = math.hypot(e1 - e0, n1 - n0)
            if base < self.min_baseline_m:
                continue
            poses = [p for p in self._pose if t0 <= p[0] <= t1]
            if len(poses) < 4:
                return
            odx, ody = poses[-1][1] - poses[0][1], poses[-1][2] - poses[0][2]
            chord = math.hypot(odx, ody)
            path = poses[-1][4] - poses[0][4]
            # a chord has a direction only if the path did not fold back on
            # itself, and the GPS chord must match it in length - a fix jump
            # or a wheel spin fails this
            if chord < 0.8 * path or not (0.65 * chord < base < 1.5 * chord):
                return
            sample = normalize_angle(math.atan2(e1 - e0, n1 - n0) + math.atan2(ody, odx))
            if self.offset is None:
                self.last_residual = None
                self.offset = sample
            else:
                self.last_residual = normalize_angle(sample - self.offset)
                self.offset = normalize_angle(self.offset + self.alpha * self.last_residual)
            self.samples += 1
            self._travel_at_update = self._travel
            return

# vim: expandtab sw=4 ts=4
