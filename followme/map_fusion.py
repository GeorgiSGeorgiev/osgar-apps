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

  Matching along the WAY's normal rather than the robot's is what makes
  this safe on a straight road: there, every normal points the same way, so
  the correction only ever moves SIDEWAYS and the along-track component -
  which nothing in the picture observes - is never touched. Where the road
  bends, the normal at 3m and the normal at 9m differ, and the along-track
  component becomes observable on its own merits. No curvature has to be
  estimated and nothing has to decide whether the road is straight.

  --- Sideways and along-track are different problems ---

  Sideways is observed every frame. Along-track is observed only when the
  road turns, and the difference is not a matter of tuning:

    A road looks the same at s and at s+10 on a straight. Nothing in the
    picture distinguishes them, so no filter - Kalman, particle or
    otherwise - can recover the along-track position from the road alone.
    A filter can only decide what to do with the absence of information.

  Three candidate landmarks were measured on 89 logs and all three failed
  or were shown to be unnecessary:

    junctions seen by the camera  do not range. Approaching one, a real
      ground feature comes nearer at one metre per metre driven. Cross-
      correlating the road's side extent at the 3m ring against the 7m ring
      in the odometry-travel domain, over 350 log/side pairs at four sites,
      the peak lag is 0.00 m every time, where a ground feature would give
      4.0 m. What the mask calls "the road opens up" happens at every look-
      ahead at once: it is the robot entering the open area, not an opening
      approaching. See scratchpad/ringtest.py and drift2.py.

    the physical turn as a landmark  is real but far too rare: 1-3 turns of
      35 degrees or more per run, so the anchor would be 50-150 m old.

    dead reckoning from an anchor  is only competitive very locally. Held
      out against the GPS, the along-track error after dead reckoning is
      0.6 m at 10 m, 1.0 m at 25 m, 2.1 m at 50 m - i.e. worse than the
      GPS itself beyond about 30 m, and there is no anchor to start from.

  What DOES work is that the error is ONE 2-D VECTOR. Held out on 13 583
  points in 179 folds across four sites, fitting a single vector to the
  measurements taken on roads of one orientation predicts the measurement
  on roads 40 degrees away with a median residual of 0.48 m against 1.13 m
  uncorrected - 57% of the cross-heading error explained by one shared
  vector. So the along-track component IS recoverable, but only after the
  robot has recently driven something at an angle, and only as well as that
  geometry allows.

  --- Why recursive least squares and not a gradient step ---

  That is what the information matrix is for. Each accepted look-ahead
  gives one linear equation n.c = y in the two components of the
  correction; accumulating

      A <- f*A + n n^T ,   b <- f*b + n y ,   c = (A + ridge*I)^-1 b

  remembers WHICH DIRECTIONS it has been told about. On a straight road A
  is rank one and the ridge holds the unobserved component at zero; at a
  corner the new normal fills in the missing rank and the along-track
  component is solved for at once, from everything in the remembered
  window, instead of being crawled toward at 5 cm a frame.

  The difference is not academic. Scored causally - every observation
  judged against the correction held BEFORE it was used - at the first
  moment a fresh heading exposes the along-track component:

      no correction at all      1.39 m
      scalar gradient step      1.59 m     WORSE than doing nothing
      this filter               1.09 m

  The gradient step is worse than nothing there because its along-track
  component is a random walk: it is driven by whatever the sideways
  residuals happened to be, and points nowhere in particular.

  Sideways, over the same runs, the gradient step scores 0.51 m and this
  filter 0.56 m against 2.66 m uncorrected - a tenth of the sideways
  benefit traded for a third off the along-track error.

  --- Why it does not drive the robot onto the grass ---

  It cannot steer. It moves where the robot THINKS it is by at most
  max_correction_m; the road mask and the depth zones still decide where
  the wheels go, exactly as before.

  What it could do is talk the router out of noticing a genuine excursion
  ("the camera sees a road, so we must be on one") - and a road mask on a
  lawn does look like a road. Five gates stop that, all of them needed:

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
    ridge      an unobserved direction is held at zero rather than left
        free. This is the along-track safety property above.
    forgetting with no accepted observation the information decays with a
        half-life of half_life_s, so a correction learned from a mistake
        does not outlive the mistake.

  --- Scored, out of sample ---

  Each accepted observation is scored BEFORE it is used, against the
  correction learned from earlier frames only, so this is a prediction and
  not a fit. Gap between the road the camera sees and the road the map
  draws, median over the whole set:

      competition day  2.16m -> 0.40m     (n=13054 observations)
      Stromovka 09-17  3.18m -> 0.65m     (n=14326)
      Stromovka 09-12  1.32m -> 0.33m     (n=15757)
      Suchdol 09-18    1.19m -> 0.35m     (n=14815)

  and the map's own "how far off the road is the robot" at those moments:

      competition day  median 0.51 -> 0.00, p90 3.44 -> 0.77, mean 1.04 -> 0.19
      Stromovka 09-17  median 1.66 -> 0.00, p90 3.27 -> 0.63, mean 1.67 -> 0.19
      Suchdol 09-18    median 0.00 -> 0.00, p90 1.29 -> 0.00, mean 0.37 -> 0.04
"""
import math

import numpy as np


class MapFusion:
    """Estimates one correction vector (east, north) in metres."""

    def __init__(self, max_correction_m=3.0, align_deg=25.0, max_residual_m=3.0,
                 looks_m=(3.0, 5.0, 7.0), min_rings=8, min_mask_frac=0.25,
                 min_halfwidth_m=0.4, max_halfwidth_m=4.0, half_life_s=8.0,
                 rate_hz=8.0, ridge=4.0, max_slew_m_s=1.0, sigma_m=4.0,
                 max_snap_m=10.0):
        self.max_correction_m = max_correction_m
        self.align = math.radians(align_deg)
        self.max_residual_m = max_residual_m
        self.looks_m = tuple(looks_m)
        self.min_rings = min_rings
        self.min_mask_frac = min_mask_frac
        self.min_halfwidth_m = min_halfwidth_m
        self.max_halfwidth_m = max_halfwidth_m
        self.max_snap_m = max_snap_m
        # Forgetting factor per update. The bias holds a direction for
        # 30-60s, but a SHORT memory scored better: half-lives of 8, 15 and
        # 30s gave a sideways error of 0.56, 0.62 and 0.70m for the same
        # along-track gain, because a longer window averages a bias that is
        # still moving.
        self.forget = 0.5 ** (1.0 / max(1e-6, half_life_s * rate_hz))
        self.ridge = ridge
        self.max_slew = max_slew_m_s / max(1e-6, rate_hz)
        # The three look-aheads of one frame, and one frame and the next,
        # are nowhere near independent measurements, so the information
        # matrix overstates how much is known by more than an order of
        # magnitude. sigma_m is the empirical factor that makes the
        # published uncertainty match the error actually observed (0.22m
        # claimed against 1.09m measured at a fresh heading). Published for
        # the log and for the follower to gate on; never acted on here.
        self.sigma_m = sigma_m
        self.reset()

    def reset(self):
        self.corr = np.zeros(2)         # what is applied (slew limited)
        self.solution = np.zeros(2)     # what the information currently says
        self.info = np.zeros((2, 2))    # sum of n n^T, with forgetting
        self.rhs = np.zeros(2)          # sum of n y
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

    def sigma_along(self, heading):
        """How well the correction is known ALONG the road the robot is
        facing, in metres. Large on a long straight, where nothing in the
        picture observes it; small again once a corner has been driven."""
        if heading is None:
            return float('inf')
        u = np.array([math.sin(heading), math.cos(heading)])
        try:
            m = np.linalg.inv(self.info + self.ridge * np.eye(2))
        except np.linalg.LinAlgError:
            return float('inf')
        return float(self.sigma_m * math.sqrt(max(0.0, float(u @ m @ u))))

    # --- estimate -----------------------------------------------------
    def update(self, graph, x, y, heading, axis):
        """One fusion step at map position (x, y) with compass `heading`
        and the camera's road axis `axis` (see road_geometry.trunk):
        dict(y0, slope, half, n, frac). Returns True if it was used."""
        self.updates += 1
        self.info *= self.forget
        self.rhs *= self.forget
        used = self._update(graph, x, y, heading, axis)
        # Slew toward the solution rather than jumping to it. When a corner
        # suddenly makes the along-track observable, the solution can move
        # metres in one frame; the route the follower is being given should
        # not.
        delta = self.solution - self.corr
        size = float(np.hypot(*delta))
        if self.max_slew > 0 and size > self.max_slew:
            delta = delta * (self.max_slew / size)
        self.corr = self.corr + delta
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
            # the residual is measured at the CORRECTED position, so
            # n.corr - residual is a direct measurement of n.c*
            n = q['nrm']
            self.info += np.outer(n, n)
            self.rhs += n * (float(n @ self.corr) - residual)
            used += 1
        if not used:
            self.last_reason = 'no look-ahead matched a mapped way'
            return False
        solution = np.linalg.solve(self.info + self.ridge * np.eye(2), self.rhs)
        size = float(np.hypot(*solution))
        if size > self.max_correction_m:
            solution = solution * (self.max_correction_m / size)
        self.solution = solution
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
