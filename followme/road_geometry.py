"""
  Ground geometry of the RedRoad mask: where the road actually is, in metres.

  tulak_obstacle reads the mask as a picture - "the road looks a bit left
  of centre, steer left". That is enough to stay on a path and not enough
  to say anything about a MAP, because a map is metric and a picture is
  not: "the centroid is 40px right" does not say whether the robot is 0.5m
  or 3m off the path, and "a branch in the top-left corner" does not say
  whether that fork is 4m ahead or 12m.

  This module answers both, by projecting the mask onto the ground plane.
  Everything here is pure geometry on one frame; nothing is filtered over
  time and nothing steers. See map_fusion.py for what consumes it.

  --- The calibration, and how it was measured ---

  Camera frame: x forward, y left, z up, origin under the camera on the
  ground. A ground point at forward distance x and lateral offset y lands
  at image (row, col):

      row = horizon_row + K / (x - x0)         K = h*f/cos^2(phi)
      col = cx - y * (row - horizon_row) / (h/cos(phi))

  with h the camera height, phi its upward tilt, f the focal length in
  mask pixels and horizon_row = cy + f*tan(phi).

  Inverting gives the two numbers this module is built on:

      lateral  y = h/cos(phi) * (cx - col) / (row - horizon_row)
      forward  x = K / (row - horizon_row) + x0

  Note which parameters each needs. LATERAL OFFSET IS ESSENTIALLY FREE OF
  THE FOCAL LENGTH - it needs the camera height and where the horizon sits
  in the frame, both measurable off the logs, and f only through sec(phi),
  which is 1.0054 at 69 deg and 1.0022 at 50 deg: a 0.3% difference over
  the whole plausible range. That is
  the single most useful fact in this file, because the lateral offset is
  what corrects a GPS fix sideways, and it means that correction survives
  even if the field of view is not what the datasheet says. Forward
  distance scales linearly with f, so a junction's estimated range is only
  as good as the focal length is - which is why map_fusion trusts the
  sideways correction far more than the along-track one.

  Measured, not assumed:

    camera height 0.325m, tilt 8.7 deg UP - least-squares ground plane
      through oak.depth (640x400, mono VFOV 55 deg -> f=384.2). Rows
      280/300/320/340/360/380 of a typical frame read 5.96/3.26/2.12/
      1.58/1.26/1.04m, which is a straight line in (1/Z, row) with
      intercept 0.1532 = tan(8.71 deg) and slope 0.328 = h/cos(phi).
      Sanity check on the same numbers: obstdet3d_zones' ground window
      (rows 300-345) then looks at 0.6-0.8m of ground, which is the range
      min_ground_dist/max_ground_dist (0.42/1.5) were tuned around.

    horizon_row 144 of 240 - fitted on 600 masks from the competition
      runs by asking which horizon makes a road's projected WIDTH
      independent of how far away it is measured. The answer came out
      143.3-144.4 for every focal length from 180 to 360 px, i.e. it is
      determined by the data and not by the rest of the calibration.
      (The colour camera sits ~3 deg off the mono pair's rectified frame;
      the depth-derived tilt alone would have put the horizon at 155.)

    focal 232.8 px on the 320-wide mask - HFOV 69 deg, the OAK-D Pro
      colour datasheet value for its 4:3 output, and the same value
      tulak_obstacle's _mask_fov_scale and its depth-bin mapping already
      assume. Not independently confirmed here: a closed-loop road
      follower keeps the road centred in frame, which removes the very
      signal (the road sliding sideways when the robot yaws) that would
      measure it. Treat +-20% as the honest uncertainty on FORWARD
      distances, and nothing on lateral ones.

  --- What comes out ---

  road_profile() walks the road up the frame from directly under the
  robot and returns, per look-ahead distance: the centre of the corridor
  the robot is in, its left and right edge, whether each edge is a real
  road boundary or just the edge of the picture, and how much road is
  outside that corridor on either side (which is how a side branch
  announces itself). branches() turns the latter into discrete "a path
  leaves to the left, about 6m ahead" events.
"""
import math

import numpy as np

# --- calibration (see module docstring) ---------------------------------
CAMERA_HEIGHT_M = 0.325
HORIZON_ROW_FRAC = 144.0 / 240.0        # a fraction, so any mask size works
MASK_HFOV_DEG = 69.0

# --- extraction ---------------------------------------------------------
NEAR_M = 1.0            # first look-ahead ring
FAR_M = 10.0            # last one - beyond this a single mask row is >1.5m deep
STEP_M = 0.25


class Projection:
    """Mask pixels <-> ground points, for one mask size."""

    def __init__(self, width, height, camera_height_m=CAMERA_HEIGHT_M,
                 horizon_row_frac=HORIZON_ROW_FRAC, hfov_deg=MASK_HFOV_DEG):
        self.width, self.height = width, height
        self.cx, self.cy = width / 2.0, height / 2.0
        self.h = camera_height_m
        self.horizon_row = horizon_row_frac * height
        self.f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        tphi = (self.horizon_row - self.cy) / self.f
        self.tan_phi = tphi
        self.sec_phi = math.sqrt(1.0 + tphi * tphi)
        self.K = self.h * self.f * (1.0 + tphi * tphi)      # h*f/cos^2(phi)
        self.x0 = self.h * tphi

    def row_of(self, x):
        """Mask row (float) of the ground point x metres ahead."""
        dx = x - self.x0
        if dx <= 1e-3:
            return float(self.height)
        return self.horizon_row + self.K / dx

    def forward_of(self, row):
        dr = row - self.horizon_row
        if dr <= 1e-3:
            return float('inf')
        return self.K / dr + self.x0

    def lateral_scale(self, row):
        """metres of lateral offset per column of pixels, at this row"""
        dr = max(1e-3, row - self.horizon_row)
        return self.h * self.sec_phi / dr

    def col_of(self, x, y):
        row = self.row_of(x)
        return self.cx - y / max(1e-9, self.lateral_scale(row)), row

    def half_width_m(self, x):
        """How much lateral ground the frame can see at this distance."""
        row = self.row_of(x)
        return self.cx * self.lateral_scale(row)


def _runs(line):
    """[(start, end_inclusive), ...] of the nonzero runs of a 1-D 0/1 row."""
    idx = np.flatnonzero(line)
    if len(idx) == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[brk + 1]])
    ends = np.concatenate([idx[brk], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def road_profile(mask, proj=None, near_m=NEAR_M, far_m=FAR_M, step_m=STEP_M,
                 min_run_px=3, max_jump_m=1.5):
    """Follow the corridor the robot is in, up the frame.

    Returns a dict of equal-length lists, one entry per look-ahead ring:
      x[]            look-ahead distance, metres
      centre[]       lateral offset of the corridor centre (+ = left), or None
      left[], right[]  lateral offset of its edges (+ = left, so left > right)
      left_open[], right_open[]  True when that edge is the edge of the
                     PICTURE rather than a road boundary - i.e. the road
                     may well continue and we simply cannot see it
      side_left[], side_right[]  how much road (metres of lateral extent)
                     lies outside the corridor on that side: a fork that
                     the mask sees as a SEPARATE blob
      edge_left[], edge_right[]  the outermost road pixel on that side over
                     all blobs - at a junction the corridor simply widens,
                     the branch is not a separate blob at all, so this is
                     what actually finds one
      frac[]         fraction of that mask row which is road
    """
    h, w = mask.shape
    if proj is None:
        proj = Projection(w, h)
    xs = np.arange(near_m, far_m + 1e-6, step_m)
    keys = ('x', 'centre', 'left', 'right', 'left_open', 'right_open',
            'side_left', 'side_right', 'edge_left', 'edge_right', 'frac')
    out = {k: [] for k in keys}
    col = None
    last_row = None
    for x in xs:
        row = int(round(proj.row_of(x)))
        out['x'].append(float(x))
        if row < 0 or row >= h or row == last_row:
            # off the frame, or the same mask row as the previous ring
            # (far out, one row spans several rings) - repeat it rather
            # than pretend to a resolution the picture does not have
            same = row == last_row
            for k in keys[1:]:
                out[k].append(out[k][-1] if (same and out[k]) else
                              (False if k.endswith('_open') else None))
            continue
        last_row = row
        line = mask[row]
        runs = [r for r in _runs(line) if r[1] - r[0] + 1 >= min_run_px]
        scale = proj.lateral_scale(row)
        out['frac'].append(float(line.mean()))
        if not runs:
            for k in ('centre', 'left', 'right', 'side_left', 'side_right',
                      'edge_left', 'edge_right'):
                out[k].append(None)
            out['left_open'].append(False)
            out['right_open'].append(False)
            continue
        if col is None:
            first = min(runs, key=lambda r: abs((r[0] + r[1]) / 2.0 - proj.cx))
            col = (first[0] + first[1]) // 2
        here = None
        for a, b in runs:
            if a - 1 <= col <= b + 1:
                here = (a, b)
                break
        if here is None:
            # the corridor stopped here: take the nearest run, but only if
            # it is close enough to plausibly be the same road
            a, b = min(runs, key=lambda r: min(abs(r[0] - col), abs(r[1] - col)))
            if min(abs(a - col), abs(b - col)) * scale > max_jump_m:
                for k in ('centre', 'left', 'right', 'side_left', 'side_right',
                          'edge_left', 'edge_right'):
                    out[k].append(None)
                out['left_open'].append(False)
                out['right_open'].append(False)
                continue
            here = (a, b)
        a, b = here
        col = (a + b) // 2
        yl = (proj.cx - a) * scale          # left edge (+ left)
        yr = (proj.cx - b) * scale
        out['centre'].append((yl + yr) / 2.0)
        out['left'].append(yl)
        out['right'].append(yr)
        out['left_open'].append(a <= 1)
        out['right_open'].append(b >= w - 2)
        side_l = sum((min(rb, a - 1) - ra + 1) for ra, rb in runs if ra < a) * scale
        side_r = sum((rb - max(ra, b + 1) + 1) for ra, rb in runs if rb > b) * scale
        out['side_left'].append(max(0.0, side_l))
        out['side_right'].append(max(0.0, side_r))
        out['edge_left'].append((proj.cx - runs[0][0]) * scale)
        out['edge_right'].append((proj.cx - runs[-1][1]) * scale)
    return out


def corridor_offset(profile, x_lo, x_hi, max_halfwidth_m=4.0):
    """Mean lateral offset of the corridor centre over a look-ahead band,
    using only rings where BOTH edges are real road boundaries (otherwise
    the "centre" is the centre of the picture, not of the road).

    Returns (offset_m, halfwidth_m, n_rings) or None."""
    vals, hw = [], []
    for i, x in enumerate(profile['x']):
        if not (x_lo <= x <= x_hi):
            continue
        c = profile['centre'][i]
        if c is None or profile['left_open'][i] or profile['right_open'][i]:
            continue
        half = (profile['left'][i] - profile['right'][i]) / 2.0
        if half <= 0 or half > max_halfwidth_m:
            continue
        vals.append(c)
        hw.append(half)
    if not vals:
        return None
    return float(np.mean(vals)), float(np.mean(hw)), len(vals)


def edge_offset(profile, x_lo, x_hi, max_halfwidth_m=4.0):
    """Like corridor_offset but usable when only ONE edge is visible: the
    distance to whichever edges were seen.

    Returns (left_m or None, right_m or None, n_left, n_right)."""
    ls, rs = [], []
    for i, x in enumerate(profile['x']):
        if not (x_lo <= x <= x_hi):
            continue
        if profile['centre'][i] is None:
            continue
        if not profile['left_open'][i] and profile['left'][i] < max_halfwidth_m:
            ls.append(profile['left'][i])
        if not profile['right_open'][i] and profile['right'][i] > -max_halfwidth_m:
            rs.append(profile['right'][i])
    return (float(np.mean(ls)) if ls else None,
            float(np.mean(rs)) if rs else None, len(ls), len(rs))


def trunk(profile, x_lo=2.0, x_hi=9.0, min_rings=4):
    """The road the robot is ON, as a straight line plus a half-width.

    Only rings whose two edges are both real road boundaries can say
    anything about the width, and at a fork they say something too WIDE -
    so the half-width is the 25th percentile over the band rather than the
    mean. Returns (y0, slope, half_width_m, n) with the axis at
    y = y0 + slope*x, or None."""
    xs, ys, hw = [], [], []
    for i, x in enumerate(profile['x']):
        if not (x_lo <= x <= x_hi) or profile['centre'][i] is None:
            continue
        if profile['left_open'][i] or profile['right_open'][i]:
            continue
        half = (profile['left'][i] - profile['right'][i]) / 2.0
        if half <= 0:
            continue
        xs.append(x)
        ys.append(profile['centre'][i])
        hw.append(half)
    if len(xs) < min_rings:
        return None
    # fit the axis on the NARROW rings only, so a junction square does not
    # drag the road's direction sideways
    half_ref = float(np.percentile(hw, 25))
    keep = [k for k in range(len(xs)) if hw[k] <= half_ref * 1.6 + 0.3]
    if len(keep) < min_rings:
        keep = list(range(len(xs)))
    slope, y0 = np.polyfit([xs[k] for k in keep], [ys[k] for k in keep], 1)
    return float(y0), float(slope), half_ref, len(keep)


def branches(profile, tr=None, min_extent_m=1.0, min_rings=3, x_lo=2.5, x_hi=9.0):
    """Side roads seen in this frame.

    At a real junction the side road is CONTIGUOUS with the road the robot
    is on - one blob, just much wider - so looking for a second blob (what
    side_left/side_right measure) finds almost nothing. What a junction
    does do is push the outermost road pixel on one side well beyond where
    this road's own edge should be. That is what this measures.

    Returns dicts with side, x_near, x_far, x_peak, extent_m (how far past
    the road edge the branch reaches) and rings."""
    if tr is None:
        tr = trunk(profile)
    if tr is None:
        return []
    y0, slope, half, _ = tr
    found = []
    for side, sign in (('left', 1.0), ('right', -1.0)):
        key = 'edge_' + side
        open_key = side + '_open'
        run, ext = [], []
        for i, x in enumerate(profile['x']):
            if not (x_lo <= x <= x_hi):
                continue
            e = profile[key][i]
            if e is None:
                over = None
            else:
                expect = (y0 + slope * x) + sign * half
                over = sign * (e - expect)
                # An edge clipped by the frame needs no special case: the
                # outermost road pixel then sits at the frame edge, so
                # `over` is how far past this road's own edge the picture
                # still shows road - a lower bound on the branch, which is
                # the direction that matters.
            if over is not None and over >= min_extent_m:
                run.append(i)
                ext.append(over)
                continue
            if len(run) >= min_rings:
                found.append(_branch(profile, side, run, ext))
            run, ext = [], []
        if len(run) >= min_rings:
            found.append(_branch(profile, side, run, ext))
    return found


def _branch(profile, side, run, ext):
    k = int(np.argmax(ext))
    return {'side': side,
            'x_near': profile['x'][run[0]],
            'x_far': profile['x'][run[-1]],
            'x_peak': profile['x'][run[k]],
            'extent_m': float(max(ext)),
            'rings': len(run)}


def road_direction(profile, x_lo=2.0, x_hi=7.0):
    """Heading of the corridor relative to the robot, radians, + = left.
    Straight-line fit of the corridor centre against distance over rings
    with two real edges; None if too few. """
    tr = trunk(profile, x_lo, x_hi)
    if tr is None:
        return None
    return float(math.atan(tr[1]))

# vim: expandtab sw=4 ts=4
