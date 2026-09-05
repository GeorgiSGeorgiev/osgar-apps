#!/usr/bin/python
"""
  Fit Matty's compass calibration from recorded logs.

  Compares the ESP32's fused yaw against the GPS receiver's own course
  over ground and fits

      error = C + A * cos(heading - phi)

  C is the constant part - magnetic declination plus the fixed rotation of
  the IMU in its mount. That is what compass_offset_deg is for.

  A is HARD IRON: something ferrous bolted to the robot, turning with it,
  so the error depends on which way the robot faces. No single offset can
  remove it, and on the 2026-09-04 CZU logs it was the DOMINANT error -
  10.9 deg of it against a 7.5 deg constant. That is what
  compass_hardiron_deg / compass_hardiron_phase_deg are for.

  Re-run this after anything is added to, removed from or moved on the
  robot. Between 2026-08 and 2026-09-04 the constant stayed put (6.1 ->
  7.5 deg) while the hard-iron term nearly doubled (5.6 -> 10.9 deg).

  Usage
  -----
    python fit_compass.py logs/.../2026_09_04_v3_czu_test0/*.log
    python fit_compass.py "logs/**/*.log" --min-sog 0.3

  Only STRAIGHT, FORWARD, moving samples are used, for the same reason
  tulak_obstacle's own runtime calibration gates on straightness: course
  over ground is the direction of TRAVEL, which equals the direction the
  robot FACES only when it is not turning. Samples where yaw moved more
  than --steady-deg inside the preceding two seconds are dropped.

  What it prints is the config block to paste into the app's init, plus
  the residual you would get with it - so you can see whether the fit is
  worth applying before you apply it.
"""
import argparse
import glob
import math
import os
import statistics
import sys

import numpy as np

from osgar.logger import LogReader, lookup_stream_names
from osgar.lib.serialize import deserialize

from tulak_obstacle import haversine_distance, initial_bearing, normalize_angle


def collect(logfile, min_sog, steady_deg, window_sec=2.0, baseline_m=5.0):
    """(true_heading, error, t) triples, radians, for straight forward
    driving.

    Reference heading comes from the receiver's own course over ground
    where the log has it. Logs recorded before RMC parsing was added carry
    only GGA, so there is no course at all in them - those fall back to the
    bearing between two fixes `baseline_m` apart, which is what the driver
    itself used before. Noisier (it averages over whatever the robot did in
    between, which is why the straightness gate matters even more there),
    but it makes the older logs usable for comparison."""
    names = lookup_stream_names(logfile)
    wanted = {i + 1: n.split('.')[-1] for i, n in enumerate(names)
              if n.split('.')[-1] in ('rotation', 'nmea_data')}
    if not wanted:
        return [], 'none'
    history, samples = [], []
    anchor = None          # (lat, lon, yaw) of the last baseline anchor
    used_cog = False
    with LogReader(logfile, only_stream_id=list(wanted)) as log:
        for dt, stream_id, raw in log:
            channel = wanted[stream_id]
            data = deserialize(raw)
            now = dt.total_seconds()
            if channel == 'rotation':
                history.append((now, math.radians(data[0] / 100.0)))
                history = [h for h in history if now - h[0] <= window_sec]
                continue
            if channel != 'nmea_data' or not history:
                continue
            yaw = history[-1][1]
            spread = math.degrees(max(abs(normalize_angle(y - yaw)) for _t, y in history))
            geometric = normalize_angle(math.pi / 2 - yaw)   # matty.py's convention, no offset

            cog, sog = data.get('cog'), data.get('sog')
            if cog is not None and sog is not None:
                used_cog = True
                if sog < min_sog or spread > steady_deg:
                    continue
                true_heading = math.radians(cog) % (2 * math.pi)
                samples.append((true_heading, normalize_angle(true_heading - geometric), now))
                continue

            lat, lon = data.get('lat'), data.get('lon')
            if lat is None or lon is None:
                continue
            if data.get('lat_dir') == 'S':
                lat = -lat
            if data.get('lon_dir') == 'W':
                lon = -lon
            if anchor is None:
                anchor = (lat, lon, yaw, now)
                continue
            if haversine_distance(anchor[0], anchor[1], lat, lon) < baseline_m:
                continue
            # the whole leg must have been straight, not just the last 2s
            if spread > steady_deg or abs(math.degrees(normalize_angle(yaw - anchor[2]))) > steady_deg:
                anchor = (lat, lon, yaw, now)
                continue
            true_heading = initial_bearing(anchor[0], anchor[1], lat, lon)
            samples.append((true_heading, normalize_angle(true_heading - geometric), now))
            anchor = (lat, lon, yaw, now)
    return samples, ('course-over-ground' if used_cog else 'gps fix-differencing')


def fit(samples):
    """Least squares for error = C + A*cos(h) + B*sin(h)."""
    h = np.array([s[0] for s in samples])
    e = np.array([s[1] for s in samples])
    design = np.column_stack([np.ones_like(h), np.cos(h), np.sin(h)])
    (const, a, b), *_ = np.linalg.lstsq(design, e, rcond=None)
    amplitude = math.hypot(a, b)
    phase = math.atan2(b, a)
    residual = e - (const + a * np.cos(h) + b * np.sin(h))
    return const, amplitude, phase, e, residual


def report(name, degrees_array):
    d = np.abs(np.degrees(degrees_array))
    print('    %-34s median %5.1f   p90 %5.1f   max %5.1f' %
          (name, np.median(d), np.percentile(d, 90), d.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logfile', nargs='+', help='recorded log file(s); globs are expanded')
    parser.add_argument('--min-sog', type=float, default=0.25,
                         help='ignore fixes slower than this (m/s) - course over ground is '
                              'noise at a crawl (default 0.25)')
    parser.add_argument('--steady-deg', type=float, default=8.0,
                         help='max yaw movement in the preceding 2s for a sample to count as '
                              'straight driving (default 8)')
    parser.add_argument('--baseline-m', type=float, default=5.0,
                         help='fix-differencing baseline for logs with no course over ground '
                              '(default 5)')
    parser.add_argument('--current-offset', type=float, default=6.1,
                         help='compass_offset_deg currently in the config, for comparison')
    args = parser.parse_args()

    logfiles = []
    for pattern in args.logfile:
        logfiles.extend(sorted(glob.glob(pattern, recursive=True)) or [pattern])

    samples, per_log, sources = [], [], set()
    for logfile in logfiles:
        try:
            got, source = collect(logfile, args.min_sog, args.steady_deg,
                                   baseline_m=args.baseline_m)
        except Exception as e:  # noqa: BLE001 - one unreadable log must not stop the fit
            print('  %-30s skipped (%s)' % (os.path.basename(logfile)[-28:], type(e).__name__))
            continue
        if got:
            per_log.append((os.path.basename(logfile)[-28:], got, source))
            samples.extend(got)
            sources.add(source)

    print('%-30s %7s %10s   %s' % ('log', 'samples', 'median err', 'reference'))
    for name, got, source in per_log:
        print('%-30s %7d %9.1f deg   %s' %
              (name, len(got), statistics.median(math.degrees(s[1]) for s in got), source))
    print()
    if len(samples) < 30:
        print('Only %d straight-driving samples - not enough to fit. Drive some straight '
               'legs at over %.2f m/s, in as many different directions as you can.'
               % (len(samples), args.min_sog))
        return 1

    const, amplitude, phase, error, residual = fit(samples)
    print('%d straight-driving samples' % len(samples))
    print()
    print('  error = %+.1f deg  +  %.1f deg * cos(heading - %.0f deg)'
           % (math.degrees(const), math.degrees(amplitude), math.degrees(phase) % 360))
    print()
    print('  residual:')
    report('current config (%.1f deg)' % args.current_offset,
           error - math.radians(args.current_offset))
    report('best constant alone (%.1f deg)' % math.degrees(const), error - const)
    report('constant + hard-iron term', residual)
    print()

    coverage = sorted(set(int(math.degrees(s[0])) // 45 for s in samples))
    if len(coverage) < 5:
        print('  WARNING: samples only cover %d of the 8 heading sectors. The hard-iron term'
               % len(coverage))
        print('           needs driving in many directions to be trustworthy - treat the')
        print('           amplitude below as provisional.')
        print()
    print('  paste into the app init of your config:')
    print('      "compass_offset_deg": %.1f,' % math.degrees(const))
    print('      "compass_hardiron_deg": %.1f,' % math.degrees(amplitude))
    print('      "compass_hardiron_phase_deg": %.0f,' % (math.degrees(phase) % 360))
    print()
    print('  (set compass_hardiron_deg to 0 to go back to a single offset)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: expandtab sw=4 ts=4
