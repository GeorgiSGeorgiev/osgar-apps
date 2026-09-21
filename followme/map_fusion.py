"""
  Where the robot really is on the map: GPS + odometry + the road mask.

  --- The problem, measured ---

  Matty's GPS is smooth from fix to fix and biased over minutes. Scored on
  46 000 frames of the 2026-09-17/18/19 logs, at moments when the camera is
  confidently centred on a road and the map has a road running the same way
  within 10m, the map still put the robot a median of 1.2-3.5m from where
  the camera saw the road, reaching 4m at p90. The bias holds one direction
  for 30-60s and then moves. Some of it is the fix, some is the OSM
  geometry being drawn a couple of metres off the real path - the two are
  indistinguishable from the robot and, usefully, have the same cure.

  That error is why turns get missed: the router asks "how far to the
  junction" of a map that thinks the robot is on a different part of it.

  --- What this does ---

  One correction vector, in metres, added to every fix before the route is
  tracked on it. It is estimated by matching the road the CAMERA sees
  against the road the MAP draws - point-to-line ICP, the classic map
  matching update, at three look-ahead distances per frame:

    camera point   C(L) = p + L*u + y_cam(L)*n_robot
    map point      M(L) = the nearest point of the nearest way to p + L*u
    residual       r(L) = (C(L) - M(L)) . n_way(L)
    update         corr -= gain * r(L) * n_way(L)

  Matching along the WAY's normal rather than the robot's is what makes
  this safe on a straight road: there, every normal points the same way, so
  the correction only ever moves SIDEWAYS and the along-track component -
  which nothing in the picture observes - is never touched. Where the road
  bends, the normal at 3m and the normal at 9m differ, and the along-track
  component becomes observable on its own merits. No curvature has to be
  estimated and nothing has to decide whether the road is straight.

  --- Why it does not drive the robot onto the grass ---

  It cannot steer. It moves where the robot THINKS it is by at most
  max_correction_m; the road mask and the depth zones still decide where
  the wheels go, exactly as before.

  What it could do is talk the router out of noticing a genuine excursion
  ("the camera sees a road, so we must be on one") - and a road mask on a
  lawn does look like a road. Four gates stop that, all of them needed:

    alignment  the mapped way must run within align_deg of the road the
        camera sees. In run 080142 at t=430-480s Matty drove off the
        mapped network onto an unmapped track; the nearest way was 35-50
        deg off its heading throughout, so every frame of that excursion
        was rejected and the correction stayed where it was.
    residual   a disagreement bigger than max_residual_m is not a fix
        bias, it is a different road. Rejected, not clamped: clamping
        would let a wrong association drag the estimate in slowly.
    magnitude  the accumulated correction is capped at max_correction_m
        (3m). Paths in a park are further apart than that, so the cap is
        also what stops the estimate walking onto the neighbouring path.
    decay      with no accepted observation the correction bleeds back
        toward zero with a ~100s time constant, so a correction learned
        from a mistake does not outlive the mistake.

  --- Scored, out of sample ---

  Each accepted observation is scored BEFORE it is used, against the
  correction learned from earlier frames only, so this is a prediction and
  not a fit. Gap between the road the camera sees and the road the map
  draws, median over the whole set:

      competition day  2.07m -> 0.43m     (n=10261 observations)
      Stromovka 09-17  3.18m -> 0.65m     (n=14326)
      Suchdol 09-18    1.19m -> 0.35m     (n=14815)

  and the map's own "how far off the road is the robot" at those moments:

      competition day  median 0.48 -> 0.00, p90 3.78 -> 1.01, mean 1.09 -> 0.23
      Stromovka 09-17  median 1.66 -> 0.00, p90 3.27 -> 0.63, mean 1.67 -> 0.19
      Suchdol 09-18    median 0.00 -> 0.00, p90 1.29 -> 0.00, mean 0.37 -> 0.04

  --- What is deliberately NOT here ---

  Correcting the along-track position from junctions the camera sees. The
  camera's branch sightings do sit closer than the map's junctions by a
  consistent 2.9m (comp) / 3.3m (Stromovka) median, which is the right
  sign for the missed turns - but the spread is 4.1-4.4m, and scored
  against the only unarguable ground truth available (the moment Matty
  physically turns 40+ deg) neither instrument wins: the map's nearest
  junction is a median 6.2m out and the camera's branch 6.8m, on 21
  events. That is not enough to move the robot's position with, so it is
  not used for that. See tulak_obstacle's arrow logic for the branch
  sighting used as a turn TRIGGER instead, which risks nothing but the
  timing of a turn the map already asked for.
"""
import math

import numpy as np


class MapFusion:
    """Estimates one correction vector (east, north) in metres."""

    def __init__(self, gain=0.05, max_correction_m=3.0, align_deg=25.0,
                 max_residual_m=3.0, looks_m=(3.0, 5.0, 7.0), min_rings=8,
                 min_mask_frac=0.25, min_halfwidth_m=0.4, max_halfwidth_m=4.0,
                 decay_per_update=0.001, max_step_m=0.05, max_snap_m=10.0):
        self.gain = gain
        self.max_correction_m = max_correction_m
        self.align = math.radians(align_deg)
        self.max_residual_m = max_residual_m
        self.looks_m = tuple(looks_m)
        self.min_rings = min_rings
        self.min_mask_frac = min_mask_frac
        self.min_halfwidth_m = min_halfwidth_m
        self.max_halfwidth_m = max_halfwidth_m
        self.decay_per_update = decay_per_update
        self.max_step_m = max_step_m
        self.max_snap_m = max_snap_m
        self.reset()

    def reset(self):
        self.corr = np.zeros(2)
        self.accepted = 0
        self.updates = 0
        self.last_residuals = []
        self.last_reason = 'no observation yet'

    # --- use ----------------------------------------------------------
    @property
    def correction(self):
        return float(self.corr[0]), float(self.corr[1])

    @property
    def magnitude(self):
        return float(np.hypot(self.corr[0], self.corr[1]))

    def apply(self, x, y):
        return x + float(self.corr[0]), y + float(self.corr[1])

    # --- estimate -----------------------------------------------------
    def update(self, graph, x, y, heading, axis):
        """One fusion step at map position (x, y) with compass `heading`
        and the camera's road axis `axis` (see road_geometry.trunk):
        dict(y0, slope, half, n, frac). Returns True if it was used."""
        self.updates += 1
        used = self._update(graph, x, y, heading, axis)
        if not used and self.decay_per_update > 0:
            self.corr *= (1.0 - self.decay_per_update)
        return used

    def _update(self, graph, x, y, heading, axis):
        self.last_residuals = []
        if axis is None:
            self.last_reason = 'no road axis'
            return False
        if axis.get('n', 0) < self.min_rings:
            self.last_reason = 'road axis from only %d rings' % axis.get('n', 0)
            return False
        if axis.get('frac', 1.0) < self.min_mask_frac:
            self.last_reason = 'mask only %.0f%% road' % (100 * axis.get('frac', 0))
            return False
        half = axis.get('half', 0.0)
        if not (self.min_halfwidth_m <= half <= self.max_halfwidth_m):
            self.last_reason = 'road half-width %.1f m implausible' % half
            return False
        if heading is None:
            self.last_reason = 'no heading'
            return False

        p = np.array([x, y]) + self.corr
        u = np.array([math.sin(heading), math.cos(heading)])
        n_robot = np.array([-math.cos(heading), math.sin(heading)])
        cam_rel = math.atan(axis['slope'])
        step = np.zeros(2)
        used = 0
        for look in self.looks_m:
            q = _snap_xy(graph, p + look * u)
            if q is None or q['dist'] > self.max_snap_m:
                continue
            way_rel = _normalize(math.atan2(q['tan'][0], q['tan'][1]) - heading)
            if abs(way_rel) > math.pi / 2:
                way_rel = _normalize(way_rel + math.pi)   # ways are undirected
            if abs(_normalize(way_rel - cam_rel)) > self.align:
                continue                                  # a different road
            camera_point = p + look * u + (axis['y0'] + axis['slope'] * look) * n_robot
            residual = float((camera_point - q['M']) @ q['nrm'])
            if abs(residual) > self.max_residual_m:
                continue                                  # not a fix bias
            self.last_residuals.append((look, residual))
            step = step - self.gain * residual * q['nrm']
            used += 1
        if not used:
            self.last_reason = 'no look-ahead matched a mapped way'
            return False
        step /= used
        size = float(np.hypot(*step))
        if self.max_step_m > 0 and size > self.max_step_m:
            step *= self.max_step_m / size
        self.corr = self.corr + step
        size = float(np.hypot(*self.corr))
        if size > self.max_correction_m:
            self.corr *= self.max_correction_m / size
        self.accepted += 1
        self.last_reason = 'ok'
        return True


def _snap_xy(graph, xy):
    lat, lon = graph.to_ll(xy[0], xy[1])
    snap = graph.snap(lat, lon)
    if snap is None:
        return None
    i, _, dist, (cx, cy) = snap
    d = graph._seg_d[i]
    length = math.hypot(d[0], d[1])
    if length <= 0:
        return None
    tan = np.array([d[0] / length, d[1] / length])
    return {'M': np.array([cx, cy]), 'tan': tan,
            'nrm': np.array([-tan[1], tan[0]]), 'dist': dist, 'seg': i}


def _normalize(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

# vim: expandtab sw=4 ts=4
