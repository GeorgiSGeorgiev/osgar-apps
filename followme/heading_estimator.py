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

  Reflected signals (2026-09-28, CZU test10, 144532 t=499-518). Beside the
  student-housing blocks the fixes crept ~5 m north while Matty drove ~5 m
  west on an unchanging wheel heading. Each chord still matched the odometry
  chord in LENGTH, so it passed the check below, and one chord 132 deg off
  the learned offset moved it 66 deg at alpha 0.5; three more followed. The
  estimate ended ~100 deg wrong and the app turned Matty around on an empty
  road (the "FP U-turn"). The receiver's own Doppler course said 269 and 259
  deg - the true heading - throughout.

  Hence, all off by default (the previous behaviour):
    gate_deg        a chord more than this off the learned offset is not
                    averaged in. It is judged instead:
    cog_agree_deg   ... by the receiver's course on the same fix (Doppler,
                    far less exposed to multipath than the position): the
                    chord is ACCEPTED outright if the course agrees with it
                    and REJECTED if the course agrees with the offset;
    consensus_n     ... and without a course, only once this many
                    consecutive out-of-gate chords agree with each other
                    within consensus_spread_deg (the offset then jumps to
                    their mean - the case of a genuinely wrong offset, e.g.
                    comp 135449 t=111, where a +83 deg chord was right);
    jump_m          a fix that disagrees with the odometry-predicted step
                    from the previous fix by more than this is never used
                    as a chord end;
    cog_veto_deg    and a chord INSIDE the gate is still refused when the
                    course on its fix disagrees with it by more than this
                    and sides with the offset - a slow creep otherwise walks
                    the offset through any gate a few degrees at a time.
  Scored against a robust reference (odometry-to-GPS rotation fitted over
  +-12 m of travel with outlier rejection) on 20 475 fixes from Stromovka
  0-3 and CZU 9-10, with max_baseline_sec 15 -> 25 so that a 4 m chord
  exists at all at 0.2-0.3 m/s (CZU9 164311 sat 45-70 deg wrong for 25 s
  without one): errors over 45 deg 12 -> 2 samples, over 90 deg 1 -> 0,
  over 20 deg 199 -> 157, p50 3.4 deg unchanged, p90 9.3 -> 9.2, p99 19.6
  -> 18.0 (competition day alone p90 9.4 -> 10.5, the one site that got
  worse). Not counted: CZU9 161640 t=304-339, where the REFERENCE is wrong -
  the Doppler course there (310 deg) agrees with the estimate (302), not
  with the reference (191).
  What it cannot do: with no course at all, a creep smooth enough to stay
  under jump_m and inside the gate still moves the heading - which is why
  unsettled_deg() exists and the app will not turn around while it is high.
"""
import math


def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class OdoGpsHeading:
    def __init__(self, alpha=0.5, min_baseline_m=4.0, max_baseline_sec=15.0, valid_travel_m=60.0,
                 gate_deg=0.0, cog_agree_deg=25.0, cog_min_sog=0.25, consensus_n=3,
                 consensus_spread_deg=15.0, jump_m=0.0, cog_veto_deg=0.0):
        # weight of each new chord in the running offset (circular EMA)
        self.alpha = alpha
        # reflected-signal defences - see the module docstring; 0 = off
        self.gate = math.radians(gate_deg)
        self.cog_agree = math.radians(cog_agree_deg)
        self.cog_min_sog = cog_min_sog
        self.consensus_n = consensus_n
        self.consensus_spread = math.radians(consensus_spread_deg)
        self.jump_m = jump_m
        self.cog_veto = math.radians(cog_veto_deg)
        self.rejected = 0               # diagnostics: out-of-gate chords not (yet) taken
        self._course = None             # (time, compass rad) of the latest fix with a usable course
        self._pending = []              # out-of-gate chords waiting for consensus
        self._bad = set()               # times of fixes never used as a chord end
        self._history = []              # (travel, offset) after each update, for unsettled_deg()
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

    def update_fix(self, time_sec, lat, lon, sog=None, cog=None):
        """sog (m/s) and cog (compass degrees) are the receiver's own speed
        and course on this fix, when it reports them."""
        if self._origin is None:
            self._origin = (lat, lon)
        east = (lon - self._origin[1]) * 111320.0 * math.cos(math.radians(self._origin[0]))
        north = (lat - self._origin[0]) * 111320.0
        self._course = ((time_sec, math.radians(cog))
                        if cog is not None and sog is not None and sog >= self.cog_min_sog else None)
        if self.jump_m > 0 and self._jumped(time_sec, east, north):
            self._bad.add(time_sec)
        self._fixes.append((time_sec, east, north))
        while self._fixes and time_sec - self._fixes[0][0] > self.max_baseline_sec:
            self._bad.discard(self._fixes[0][0])
            self._fixes.pop(0)
        self._learn()

    def _jumped(self, time_sec, east, north):
        """Does this fix disagree with the step odometry predicts from the
        previous one by more than jump_m?"""
        if self.offset is None or not self._fixes or not self._pose:
            return False
        t_prev, e_prev, n_prev = self._fixes[-1]
        if time_sec - t_prev > 2.5:
            return False
        before = [p for p in self._pose if p[0] <= t_prev]
        if not before:
            return False
        dx = self._pose[-1][1] - before[-1][1]
        dy = self._pose[-1][2] - before[-1][2]
        dist = math.hypot(dx, dy)
        bearing = self.offset - math.atan2(dy, dx)
        return math.hypot(east - e_prev - dist * math.sin(bearing),
                          north - n_prev - dist * math.cos(bearing)) > self.jump_m

    # --- output -------------------------------------------------------
    def heading(self):
        """Compass bearing (radians) or None while the offset is unknown or
        stale."""
        if self.offset is None or self._odo_heading is None:
            return None
        if self._travel_at_update is not None and self._travel - self._travel_at_update > self.valid_travel_m:
            return None
        return normalize_angle(self.offset - self._odo_heading) % (2 * math.pi)

    def course(self, time_sec, max_age_sec=3.0):
        """The receiver's latest usable course (compass radians) if recent,
        else None."""
        if self._course is None or time_sec - self._course[0] > max_age_sec:
            return None
        return self._course[1]

    def unsettled_deg(self, window_m):
        """How far (deg) the offset has moved over the last window_m metres
        of travel - a heading that just jumped is not one to turn around on."""
        if self.offset is None:
            return 0.0
        moved = 0.0
        for travel, offset in reversed(self._history):
            if self._travel - travel > window_m:
                break
            moved = max(moved, abs(normalize_angle(self.offset - offset)))
        return math.degrees(moved)

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
            if self._bad and any(t0 <= t <= t1 for t in self._bad):
                return
            sample = normalize_angle(math.atan2(e1 - e0, n1 - n0) + math.atan2(ody, odx))
            self._take(sample, t1)
            return

    def _take(self, sample, t1):
        if self.offset is None:
            self._set(sample, None)
            return
        residual = normalize_angle(sample - self.offset)
        course = self._course
        implied = None                  # the offset the receiver's course on this fix implies
        if course is not None and abs(course[0] - t1) < 0.5 and self._odo_heading is not None:
            implied = normalize_angle(course[1] + self._odo_heading)
        if self.gate <= 0 or abs(residual) <= self.gate:
            if (self.cog_veto > 0 and implied is not None
                    and abs(normalize_angle(implied - sample)) > self.cog_veto
                    and abs(normalize_angle(implied - self.offset)) < abs(normalize_angle(implied - sample))):
                # inside the gate, but the course sides with the offset: a slow
                # creep walks the offset through any gate a few degrees at a time
                self.rejected += 1
                return
            self._pending = []
            self._set(normalize_angle(self.offset + self.alpha * residual), residual)
            return
        # Out of the gate: a reflection, or an offset that really is wrong?
        if implied is not None:
            if abs(normalize_angle(implied - sample)) <= self.cog_agree:
                self._pending = []
                self._set(sample, residual)
                return
            if abs(normalize_angle(implied - self.offset)) <= self.cog_agree:
                self._pending = []
                self.rejected += 1
                return
        self.rejected += 1
        self._pending.append(sample)
        recent = self._pending[-self.consensus_n:]
        if len(recent) >= self.consensus_n:
            mean = math.atan2(sum(math.sin(v) for v in recent), sum(math.cos(v) for v in recent))
            if max(abs(normalize_angle(v - mean)) for v in recent) <= self.consensus_spread:
                self._pending = []
                self._set(mean, normalize_angle(mean - self.offset))

    def _set(self, offset, residual):
        self.last_residual = residual
        self.offset = offset
        self.samples += 1
        self._travel_at_update = self._travel
        self._history.append((self._travel, offset))
        while len(self._history) > 2 and self._travel - self._history[0][0] > 100.0:
            self._history.pop(0)

# vim: expandtab sw=4 ts=4
