#!/usr/bin/python
"""
  Ground projection of the road mask, and the map/camera fusion built on it.

      python test_road_geometry.py

  Synthetic masks only - no log, no map file, no robot. What it pins down:

    - the projection round-trips (ground point -> pixel -> ground point)
    - lateral offsets barely depend on the focal length, which is the
      property map_fusion leans on (see road_geometry's docstring)
    - a road drawn 1 m right of centre reads as 1 m right of centre
    - a road bending 15 deg reads as bending 15 deg
    - a side branch merged into the trunk (what a real junction looks
      like: one blob, suddenly much wider) is found, and does not drag
      the trunk's direction sideways
    - the fusion converges on a constant offset, stops at its ceiling,
      and refuses a road running the wrong way
"""
import math
import sys

import numpy as np

import road_geometry as rg
from map_fusion import MapFusion


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print('   ok  %s' % message)


def synth(proj, halfwidth=1.5, offset=0.0, heading=0.0, branch=None, size=(240, 320)):
    """A road of the given half-width, lateral offset and direction, with
    an optional (distance, side, extra_width) side branch touching it."""
    m = np.zeros(size, np.uint8)
    for row in range(int(proj.horizon_row) + 2, size[0]):
        x = proj.forward_of(row)
        if x > 30:
            continue
        s = proj.lateral_scale(row)
        yc = offset + math.tan(heading) * x
        c0 = int(round(proj.cx - (yc + halfwidth) / s))
        c1 = int(round(proj.cx - (yc - halfwidth) / s))
        m[row, max(0, c0):min(size[1], c1 + 1)] = 1
        if branch is not None and abs(x - branch[0]) < 1.2:
            if branch[1] == 'left':
                b0, b1 = int(round(proj.cx - (yc + halfwidth + branch[2]) / s)), c0
            else:
                b0, b1 = c1, int(round(proj.cx - (yc - halfwidth - branch[2]) / s))
            m[row, max(0, b0):min(size[1], b1 + 1)] = 1
    return m


def test_projection():
    print('== projection ==')
    p = rg.Projection(320, 240)
    for x, y in ((2.0, 0.0), (3.0, 0.5), (5.0, 1.2), (8.0, -2.0)):
        col, row = p.col_of(x, y)
        back_x = p.forward_of(row)
        back_y = (p.cx - col) * p.lateral_scale(row)
        check(abs(back_x - x) < 0.05 and abs(back_y - y) < 0.02,
              'ground (%.1f, %+.1f) m round-trips through the image' % (x, y))
    wide = rg.Projection(320, 240, hfov_deg=50.0)
    col, row = p.col_of(5.0, 1.2)
    y_default = (p.cx - col) * p.lateral_scale(row)
    y_wide = (wide.cx - col) * wide.lateral_scale(row)
    check(abs(y_default - y_wide) < 0.005 * abs(y_default),
          'lateral offset is within 0.5%% for a 69 and a 50 deg field (%.4f vs %.4f m)'
          % (y_default, y_wide))
    check(abs(p.forward_of(row) - wide.forward_of(row)) > 1.0,
          'while forward distance is not (%.1f vs %.1f m)' % (p.forward_of(row), wide.forward_of(row)))


def test_reads_the_road():
    print('== reading a road ==')
    p = rg.Projection(320, 240)
    prof = rg.road_profile(synth(p), p)
    tr = rg.trunk(prof)
    check(tr is not None and abs(tr[0]) < 0.1 and abs(tr[2] - 1.5) < 0.15,
          'a centred 3 m road reads as centred and 3 m wide (%.2f m off, half %.2f m)' % (tr[0], tr[2]))
    check(abs(math.degrees(rg.road_direction(prof))) < 2.0, 'and running straight ahead')

    prof = rg.road_profile(synth(p, offset=-1.0), p)
    tr = rg.trunk(prof)
    check(abs(tr[0] + 1.0) < 0.15, 'a road 1 m to the right reads as %.2f m' % tr[0])
    check(abs(math.degrees(rg.road_direction(prof))) < 3.0,
          'and still straight (%.1f deg)' % math.degrees(rg.road_direction(prof)))

    prof = rg.road_profile(synth(p, heading=math.radians(15)), p)
    check(abs(math.degrees(rg.road_direction(prof)) - 15.0) < 4.0,
          'a road bending 15 deg left reads %.1f deg' % math.degrees(rg.road_direction(prof)))


def test_finds_a_junction():
    print('== junctions ==')
    p = rg.Projection(320, 240)
    prof = rg.road_profile(synth(p, branch=(6.0, 'left', 2.5)), p)
    tr = rg.trunk(prof)
    found = rg.branches(prof, tr)
    check(len(found) == 1 and found[0]['side'] == 'left',
          'a branch merged into the trunk on the left is found as one branch')
    check(abs(found[0]['x_peak'] - 6.0) < 2.0,
          'at about the right distance (%.1f m, drawn at 6.0)' % found[0]['x_peak'])
    check(abs(math.degrees(math.atan(tr[1]))) < 3.0,
          'and the junction does not bend the trunk (%.1f deg)' % math.degrees(math.atan(tr[1])))
    check(not rg.branches(rg.road_profile(synth(p), p)), 'a plain road has no branches')


class FlatGraph:
    """A single straight way running north, for the fusion tests."""

    def __init__(self, east=0.0):
        self.east = east
        self.seg_a = np.array([[east, -200.0]])
        self._seg_d = np.array([[0.0, 400.0]])
        self.seg_halfwidth = np.array([1.5])

    def to_ll(self, x, y):
        return (y, x)

    def snap(self, lat, lon):
        x, y = lon, lat
        return 0, 0.5, abs(x - self.east), (self.east, y)


def test_fusion_converges():
    print('== fusion ==')
    graph = FlatGraph(east=2.0)          # the map draws the road 2 m east
    f = MapFusion()
    axis = {'y0': 0.0, 'slope': 0.0, 'half': 1.5, 'n': 20, 'frac': 0.6}
    for _ in range(400):                 # the robot drives up the real road at x=0
        f.update(graph, 0.0, 0.0, 0.0, axis)
    check(abs(f.corr[0] - 2.0) < 0.2 and abs(f.corr[1]) < 0.2,
          'a 2 m sideways map error is learned (%.2f, %.2f)' % tuple(f.corr))
    check(f.accepted > 300, 'from %d accepted observations' % f.accepted)
    # the road runs north, so nothing here observes the north component -
    # the ridge must hold it at zero rather than let it wander
    check(abs(f.corr[1]) < 0.05,
          'and the unobserved along-track component stays at zero (%.3f m)' % f.corr[1])
    check(f.sigma_along(0.0) > 3 * f.sigma_along(math.pi / 2),
          'which it also reports: along %.2f m against across %.2f m'
          % (f.sigma_along(0.0), f.sigma_along(math.pi / 2)))

    f = MapFusion(max_correction_m=1.0)
    for _ in range(400):
        f.update(graph, 0.0, 0.0, 0.0, axis)
    check(abs(np.hypot(*f.corr) - 1.0) < 0.01,
          'and it stops at max_correction_m (%.2f m)' % np.hypot(*f.corr))

    f = MapFusion()
    turned = {'y0': 0.0, 'slope': math.tan(math.radians(60)), 'half': 1.5, 'n': 20, 'frac': 0.6}
    for _ in range(100):
        f.update(graph, 0.0, 0.0, 0.0, turned)
    check(f.accepted == 0 and np.hypot(*f.corr) < 1e-6,
          'a road running 60 deg across the mapped one is refused (%s)' % f.last_reason)

    f = MapFusion()
    thin = {'y0': 0.0, 'slope': 0.0, 'half': 1.5, 'n': 3, 'frac': 0.6}
    for _ in range(100):
        f.update(graph, 0.0, 0.0, 0.0, thin)
    check(f.accepted == 0, 'so is an axis fitted to only 3 rings (%s)' % f.last_reason)

    far = FlatGraph(east=9.0)
    f = MapFusion()
    for _ in range(100):
        f.update(far, 0.0, 0.0, 0.0, axis)
    check(f.accepted == 0, 'and a 9 m disagreement, which is a different road (%s)' % f.last_reason)


def main():
    for test in (test_projection, test_reads_the_road, test_finds_a_junction,
                 test_fusion_converges):
        test()
    print('\nALL ROAD GEOMETRY / FUSION TESTS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: expandtab sw=4 ts=4
