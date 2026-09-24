#!/usr/bin/python
"""
  View Matty's obstacle-avoidance run: depth map with the live obstacle/
  ground detection windows overlaid, the color image, and a HUD showing
  what command was sent to the platform and why.

  This does NOT reimplement tulak_obstacle.py / obstdet3d_zones.py /
  osm_router.py logic - it replays the ACTUAL TulakObstacle,
  ObstacleDetector3DZones and (for runs recorded with an OSM route config)
  OSMRouter classes against the inputs recorded in the log
  (obstacle_zones, ground_hazard, rotation, pose2d, nn_mask, route_hint,
  nmea_data, ...), using the log's own
  recorded links (config['robot']['links']) to know which stream feeds
  which handler - so it automatically stays correct if the config is
  rewired later. Because both classes derive all of their timing purely
  from message timestamps (never wall-clock), replaying them in recorded
  order reproduces the exact same state-machine trajectory (state,
  stop_streak/turn_streak, escape mode, GPS blend, depth RoI shift, ...)
  that happened live, including the print()-ed reasoning
  ("obstacle too close, backing up", "stop turning, realigning to ...",
  the depth RoI shift line, etc.) which is captured here and shown as a
  "recent events" feed - that print output is the only thing this log
  does NOT already carry as a bus stream.

  Runs recorded with config/matty-tulak-osm.json get an extra HUD line
  with the route state: progress and distance remaining along the plan,
  cross-track from the planned corridor, distance to the next planned
  junction, and the bearing authority currently handed to the route (see
  osm_router.py). That authority line is the one to watch if the robot
  takes a wrong branch at a fork. Older logs have no osm_router module and
  the line is simply absent - nothing else changes.

  The same runs also get a third window, below the color image and next
  to the text: a heading-up map (forward = up) around Matty's latest GPS fix, drawn
  from the replayed router's own state - the ways it may route on, at the
  widths off_road_m is measured against, the planned route and its turns,
  the trail of raw GPS fixes, where along the route the router thinks the
  robot is, and the aim point it last published. See draw_osm_map.
  --no-map puts the old layout back.

  Controls (same as robotem-rovne/view_mask.py):
    space - pause / step one frame
    s     - save current composite frame to save_frame.jpg
    + / - - zoom the map in / out (does not step a paused replay)
    q/ESC - quit
"""
import argparse
import bisect
import contextlib
import io
import math
import os
import tempfile
from collections import deque

import cv2
import numpy as np

from osgar.logger import LogReader, lookup_stream_names, lookup_config
from osgar.lib.serialize import deserialize
from osgar.exceptions import EmergencyStopException
from osgar.obstdet3d_zones import ObstacleDetector3DZones
from osgar.lib.nn_mask import mask_on_color

from tulak_obstacle import TulakObstacle, mask_center


class FakeBus:
    """Just enough of the Bus API for a Node's on_* handlers to run
    directly against replayed data - no threads, no queues."""
    def __init__(self):
        self.published = {}

    def register(self, *outputs):
        pass

    def publish(self, channel, data):
        self.published[channel] = data
        return data

    def sleep(self, secs):
        pass

    def is_alive(self):
        return True


def build_router(links, module_names):
    """stream name (producer "module.channel") -> [(consumer module, input channel), ...]
    restricted to consumers we are actually replaying."""
    router = {}
    for src, dst in links:
        dst_module, _, dst_channel = dst.partition('.')
        if dst_module in module_names:
            router.setdefault(src, []).append((dst_module, dst_channel))
    return router


def _reason_signature(text):
    """Strip the driver's own embedded self.time prefix (its first token)
    so the same message repeating frame after frame (e.g. obstdet3d_zones's
    unconditional per-frame "ground reading looks bad..." warning) is
    recognized as a repeat instead of spamming the feed with near-identical
    lines that only differ by timestamp."""
    parts = text.split(None, 1)
    return parts[1] if len(parts) > 1 else text


def _append_reason(reason_log, timestamp, source_label, text, sig=None):
    if sig is None:
        sig = _reason_signature(text)
    if reason_log and reason_log[-1][1] == source_label and reason_log[-1][3] == sig:
        _, _, _, _, count = reason_log[-1]
        reason_log[-1] = (timestamp, source_label, text, sig, count + 1)
    else:
        reason_log.append((timestamp, source_label, text, sig, 1))


def call_handler(node, channel, timestamp, data, source_label, reason_log):
    """Run node.on_<channel>(data) as it would run live, capturing whatever
    it prints (the human-readable "why") into reason_log."""
    handler = getattr(node, 'on_' + channel, None)
    if handler is None:
        return
    node.time = timestamp
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            handler(data)
    except EmergencyStopException:
        msg = 'EmergencyStopException raised'
        _append_reason(reason_log, timestamp, source_label, msg, sig=msg)
        return
    for line in buf.getvalue().splitlines():
        if line:
            _append_reason(reason_log, timestamp, source_label, line)


def make_video_reader(tmp_path):
    """Adapted from robotem-rovne/view_mask.py's read_h264_image - decodes
    the growing h264/h265 elementary stream via cv2.VideoCapture, keyed off
    a private temp file instead of a cwd-relative 'tmp.h26x'."""
    def read_image(data, i_frame_only=True):
        is_h264 = data.startswith(bytes.fromhex('00000001 0950')) or data.startswith(bytes.fromhex('00000001 0930'))
        is_h265 = data.startswith(bytes.fromhex('00000001 460150')) or data.startswith(bytes.fromhex('00000001 460130'))
        assert is_h264 or is_h265, data[:20].hex()
        is_keyframe = data.startswith(bytes.fromhex('00000001 0950')) or data.startswith(bytes.fromhex('00000001 460150'))

        if is_keyframe:
            with open(tmp_path, 'wb') as f:
                f.write(data)
        else:
            if i_frame_only:
                return None
            with open(tmp_path, 'ab') as f:
                f.write(data)

        cap = cv2.VideoCapture(tmp_path)
        image = None
        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                image = frame
        cap.release()
        return image
    return read_image


def colorize_depth(depth_mm, max_mm):
    """near = warm/red, far = cool/blue, invalid (0) = black."""
    valid = depth_mm > 0
    clipped = np.clip(depth_mm, 0, max_mm).astype(np.float32)
    inverted = 255 - clipped / max_mm * 255
    color = cv2.applyColorMap(inverted.astype(np.uint8), cv2.COLORMAP_JET)
    color[~valid] = (0, 0, 0)
    return color


def draw_drivable_area(color_img, mask):
    """Overlay of the drivable-area (nn_mask) segmentation on the color
    image - same visualization as robotem-rovne/view_mask.py. Zeroes the
    top half before computing the centroid, mirroring tulak_obstacle's own
    on_nn_mask() sky-masking, so the crosshair drawn here matches what
    actually drove last_dir/steering rather than a naive raw centroid.

    Placed through mask_on_color() rather than stretched, so the overlay
    lands where the network actually looked - see there."""
    h_img, w_img = color_img.shape[:2]
    m = mask.copy()
    mh, mw = m.shape
    m[:mh // 2, :] = 0

    center_y, center_x = mask_center(m)
    mask_resized, (scale, x0, y0) = mask_on_color(m, w_img, h_img)
    center_x = int(center_x * scale + x0)
    center_y = int(center_y * scale + y0)

    colored_mask = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    colored_mask[mask_resized == 1] = (0, 0, 255)
    overlay = cv2.addWeighted(color_img, 1.0, colored_mask, 0.5, 0)

    cross, thickness = 30, 3
    cv2.line(overlay, (center_x - cross, center_y), (center_x + cross, center_y), (0, 255, 0), thickness)
    cv2.line(overlay, (center_x, center_y - cross), (center_x, center_y + cross), (0, 255, 0), thickness)
    return overlay


def draw_zone(img, rows, cols, color, label, scale, thickness=2):
    r0, r1 = rows
    c0, c1 = cols
    pt1 = (c0 * scale, r0 * scale)
    pt2 = (c1 * scale, r1 * scale)
    cv2.rectangle(img, pt1, pt2, color, thickness)
    if label:
        cv2.putText(img, label, (pt1[0] + 3, pt1[1] + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)


def draw_shift_gauge(img, zones, orig_height, scale):
    """Small vertical gauge showing where the pitch/tilt-compensated row
    window currently sits within its possible range - the "how much is the
    horizontal view moving up and down" ask."""
    vfov_rad = math.radians(zones.vertical_fov_deg)
    px_per_rad = orig_height / vfov_rad
    max_dynamic_px = math.radians(zones.max_pitch_shift_deg) * px_per_rad
    static_px = math.radians(zones.camera_tilt_deg) * px_per_rad
    max_extent_px = max_dynamic_px + abs(static_px)
    if max_extent_px <= 0:
        return

    gauge_x = img.shape[1] - 40
    gauge_top, gauge_bottom = 20, img.shape[0] - 20
    gauge_mid = (gauge_top + gauge_bottom) // 2
    half_len = (gauge_bottom - gauge_top) // 2

    cv2.line(img, (gauge_x, gauge_top), (gauge_x, gauge_bottom), (150, 150, 150), 2)
    cv2.line(img, (gauge_x - 8, gauge_mid), (gauge_x + 8, gauge_mid), (150, 150, 150), 1)
    cv2.putText(img, "shift", (gauge_x - 18, gauge_top - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (150, 150, 150), 1, cv2.LINE_AA)

    frac = max(-1.0, min(1.0, zones.applied_shift_px / max_extent_px))
    marker_y = int(gauge_mid + frac * half_len)
    cv2.circle(img, (gauge_x, marker_y), 6, (0, 215, 255), -1)
    cv2.putText(img, f"{zones.applied_shift_px:+.0f}px", (gauge_x - 60, marker_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 215, 255), 1, cv2.LINE_AA)


def draw_free_space_bins(depth_panel, app, zones, scale):
    """Visualizes the same depth_profile bins _free_space_steering() reads
    for its continuous "lean toward whichever side is more open" nudge -
    the guiding signal behind ordinary cruising's minor corrections,
    separate from (and acting well before) the discrete L/C/R avoidance
    zones. Drawn at the bins' ACTUAL sampled region (free_space_cols x
    zones.free_space_rows - its own pitch-compensated row window, see
    obstdet3d_zones.py, normally taller than but NOT the same box as the
    narrow C zone, and deliberately nowhere near the full frame height)
    instead of an arbitrary fixed screen position, so what you see here
    is honestly where the bins are looking, not just their column split.
    Gray outline = the sampled free_space_cols x free_space_rows region;
    colored strip along its bottom edge = the bins themselves. Green =
    counts toward the pull (brighter = more clearance past the
    threshold), dark red = present but not clear enough to count, gray
    fill = no valid data in that bin - same fail_value=None convention as
    the L/R zones. Yellow marker = the resulting weighted aim point;
    absent whenever nothing currently counts (matches
    _free_space_steering() returning 0.0)."""
    c0, c1 = zones.free_space_cols
    r0, r1 = zones.free_space_rows
    n = len(app.depth_profile) if app.depth_profile else zones.free_space_bins
    edges = np.linspace(c0, c1, n + 1)
    free_space_min_dist = app.turning_dist * app.free_space_lead_margin

    x0_all, x1_all = int(c0 * scale), int(c1 * scale)
    y0_all, y1_all = int(r0 * scale), int(r1 * scale)
    cv2.rectangle(depth_panel, (x0_all, y0_all), (x1_all, y1_all), (120, 120, 120), 1)
    cv2.putText(depth_panel, "FS BINS", (x0_all, y0_all - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 200), 1, cv2.LINE_AA)

    bar_top, bar_bottom = y1_all + 4, y1_all + 18
    total_weight, weighted_x = 0.0, 0.0
    half_corridor = (app.corridor_half_width_m + app.corridor_margin_m
                      if getattr(app, 'corridor_check', False) else 0.0)
    for i in range(n):
        x0, x1 = int(edges[i] * scale), int(edges[i + 1] * scale)
        d = app.depth_profile[i] if app.depth_profile else None
        # a white cap marks a bin whose reading lands inside the robot's
        # own swept width - i.e. one _corridor_scan is allowed to stop on
        if half_corridor > 0 and d is not None and app.depth_profile:
            bearing = app._profile_bin_bearing(i, n)
            if abs(d * math.sin(bearing)) <= half_corridor:
                cv2.rectangle(depth_panel, (x0, bar_top - 5), (x1, bar_top - 1), (255, 255, 255), -1)
        if d is None:
            # grey = unknown. A grey bin the app has decided to IGNORE (the
            # road network sees road in that bearing - see edge_grey_mode in
            # tulak_obstacle) is drawn blue-grey instead, so a debug video
            # shows the difference between "no data, still treated as a wall"
            # and "no data, known artefact".
            bin_road = getattr(app, 'profile_bin_road', []) or []
            mode = getattr(app, 'edge_grey_mode', 'legacy')
            ignored = mode == 'open' or (mode == 'mask' and i < len(bin_road)
                                         and bin_road[i] >= app.edge_grey_road_frac)
            color = (120, 95, 45) if ignored else (80, 80, 80)
        elif d <= free_space_min_dist:
            color = (0, 0, 140)
        else:
            # capped the same way _free_space_steering() caps it - see
            # that method's docstring - so this marker/coloring matches
            # what actually steers the robot, not an uncapped stand-in
            weight = min(d - free_space_min_dist, free_space_min_dist)
            frac = min(1.0, weight / free_space_min_dist) if free_space_min_dist > 0 else 1.0
            color = (0, int(90 + 130 * frac), 0)
            total_weight += weight
            weighted_x += weight * (edges[i] + edges[i + 1]) / 2
        cv2.rectangle(depth_panel, (x0, bar_top), (x1, bar_bottom), color, -1)
        cv2.rectangle(depth_panel, (x0, bar_top), (x1, bar_bottom), (200, 200, 200), 1)

    if total_weight > 0:
        aim_x = int((weighted_x / total_weight) * scale)
        pts = np.array([[aim_x - 6, bar_bottom + 10], [aim_x + 6, bar_bottom + 10], [aim_x, bar_bottom + 2]],
                        dtype=np.int32)
        cv2.fillPoly(depth_panel, [pts], (0, 215, 255))


def fmt_dist(x):
    return f"{x:4.2f}m" if x is not None else " None"


N_REASON_LINES = 5
EVENTS_HEADER = "-- recent events --"
HUD_ROW_H = 18
HUD_FONT_SCALE = 0.45


def build_hud_lines(dt, app, zones, reason_log, osm=None):
    """Always returns the same number of lines, so the HUD panel (and the
    overall composite frame, for --create-video) has a constant size."""
    lines = []
    lines.append(f"t={dt}   state={app.state.name}")

    cmd = app.bus.published.get('desired_steering')
    if cmd is not None:
        speed_permille, steer_cdeg = cmd
        lines.append(f"cmd -> speed={speed_permille / 1000:+.2f} m/s   steering={steer_cdeg / 100:+.1f} deg")
    else:
        lines.append("cmd -> (none sent yet)")

    lines.append(
        f"zones: L={fmt_dist(app.left_dist)}  C={app.last_obstacle:4.2f}m  R={fmt_dist(app.right_dist)}"
        f"   (stop_dist={app.stop_dist:.2f}m turning_dist={app.turning_dist:.2f}m)"
        f"   stop_streak={app.stop_streak} turn_streak={app.turn_streak}"
        + ("   ESCAPE MODE" if app.in_escape_mode else "")
    )

    free_space_min_dist = app.turning_dist * app.free_space_lead_margin
    n_open = sum(1 for d in app.depth_profile if d is not None and d > free_space_min_dist)
    lines.append(
        f"free-space bins: {n_open}/{len(app.depth_profile)} open (>{free_space_min_dist:.2f}m)"
        f"   aim={math.degrees(app._free_space_steering()):+.1f}deg"
    )

    lines.append(f"ground_hazard: active={app.ground_hazard_active}  streak={app.ground_hazard_streak}"
                  f"   near-obstacle: active={getattr(app, 'ground_near_active', False)}"
                  f" streak={getattr(app, 'ground_near_streak', 0)}")

    # The head-on channel the L/C/R zones cannot provide - closest thing
    # inside the robot's own swept width, from the profile bins (see
    # tulak_obstacle._corridor_scan). This is the line to watch when the
    # robot drives into something the zones called clear.
    corridor = getattr(app, 'corridor_dist', None)
    if getattr(app, 'corridor_check', False):
        half = app.corridor_half_width_m + app.corridor_margin_m
        verdict = 'clear'
        if corridor is not None:
            verdict = ('STOP' if corridor < app.stop_dist
                        else 'BLOCKED' if corridor < app.turning_dist else 'clear')
        lines.append(f"corridor (+-{half:.2f}m): {fmt_dist(corridor)}  {verdict}")
    else:
        lines.append("corridor: disabled")

    # The continuous avoidance push and the junction hint - the two terms
    # added 2026-09-11. repel is what should move a pole out of the robot's
    # path well before the discrete maneuver; the hint is closed on heading,
    # so it should fall to zero once Matty faces the exit bearing.
    repel = math.degrees(app._corridor_repulsion()) if hasattr(app, '_corridor_repulsion') else 0.0
    # read the hint the last drive cycle actually used - in 'arrow' guidance
    # the hint method arms and releases turns, so calling it from the HUD
    # would change what the robot does
    sd = getattr(app, 'steer_debug', None) or {}
    if 'hint' in sd:
        hint = math.degrees(sd.get('hint', 0.0))
    else:
        hint = math.degrees(app._route_turn_hint()) if hasattr(app, '_route_turn_hint') else 0.0
    eb = getattr(app, 'route_exit_bearing', None)
    lines.append(f"steer terms: repel {repel:+5.1f}deg   turn hint {hint:+5.1f}deg"
                 f"   exit bearing {'--' if eb is None else '%.0fdeg' % (math.degrees(eb) % 360)}"
                 f"   pitch source {getattr(zones, 'pitch_source', 'euler')}")

    pitch_deg = math.degrees(zones.smoothed_pitch)
    lines.append(
        f"depth window shift: {zones.applied_shift_px:+.0f}px "
        f"(pitch {pitch_deg:+.1f}deg + mount tilt {zones.camera_tilt_deg:+.1f}deg)"
        f"   rows={zones.rows}  ground_rows={zones.ground_rows}  free_space_rows={zones.free_space_rows}"
    )

    # raw (unsmoothed, UN-offset-corrected) IMU readout, all three axes
    # together - for (re)calibrating pitch_offset_deg (see obstdet3d_zones
    # module docstring): rest the robot on a genuinely level surface, read
    # pitch off this line, set that as pitch_offset_deg. Also still useful
    # for the axis hand-tilt test. Deliberately not smoothed_pitch/zones.
    # pitch here (those are offset-corrected and EMA-lagged; this line is
    # the true sensor reading so it responds instantly to hand motion).
    error_deg = math.degrees(zones.last_pitch_error_rad) if zones.last_pitch_error_rad is not None else None
    error_str = f"{error_deg:.1f}deg" if error_deg is not None else "n/a"
    lines.append(
        f"IMU raw (uncorrected): roll={math.degrees(zones.last_roll):+.1f}deg  "
        f"pitch={math.degrees(zones.last_raw_pitch):+.1f}deg  yaw={math.degrees(zones.last_yaw):+.1f}deg"
        f"   error={error_str} (max {math.degrees(zones.max_pitch_error_rad):.1f}deg)"
        f"   [pitch_offset_deg={math.degrees(zones.pitch_offset_rad):.1f}]"
    )

    if app.follow_gps_target:
        gps_line = f"gps: {app._following_status()}"
        if app.bearing_to_target is not None:
            gps_line += f"   bearing={math.degrees(app.bearing_to_target):.0f}deg  dist={app.target_dist:.1f}m"
        lines.append(gps_line)
    else:
        lines.append("")

    if app.route_mode:
        # Only the numbers that decide something. progress/remaining say
        # where along the plan the robot thinks it is; cross-track is what
        # the corridor monitor watches; authority is how much of the
        # steering the route is currently allowed to set (see
        # osm_router.py's "division of labour"), and is the line to watch
        # if the robot takes a wrong branch at a fork.
        parts = ['route: %s' % app.route_state]
        if app.route_remaining_m is not None:
            parts.append('remaining=%.0fm' % app.route_remaining_m)
        if app.route_cross_track_m is not None:
            parts.append('cross=%+.1fm' % app.route_cross_track_m)
        if app.route_guidance_mode:
            parts.append(app.route_guidance_mode.upper())
        if app.route_authority is not None:
            parts.append('authority=%.2f' % app.route_authority)
        # THE number that ends a Robotour run: metres past the edge of the
        # nearest mapped road (0 = provably on one). Shown even at 0 so
        # its absence is distinguishable from "on the road".
        off_m = getattr(app, 'route_off_road_m', None)
        parts.append('off_road=%s' % ('n/a' if off_m is None else '%.2fm' % off_m))
        offroad = app._route_offroad_frac() if hasattr(app, '_route_offroad_frac') else 0.0
        if offroad > 0:
            # how much of the steering the map has taken over, and the
            # speed ceiling that came with it - see _drive_steering /
            # _offroad_speed_cap
            parts.append('OFFROAD=%.2f' % offroad)
            parts.append('route_takeover=%.2f' % (offroad * app.route_offroad_authority))
            parts.append('v_cap=%.2f' % app._offroad_speed_cap())
        if getattr(app, 'offroad_hold_active', False):
            parts.append('OFF-ROAD CRAWL')
        herr = app._route_heading_error()
        if herr is not None:
            # the two numbers behind item 29: how far the camera is off the
            # mapped road axis, and how much the mask is worth as a result
            parts.append('road_axis_err=%+.0fdeg' % math.degrees(herr))
            parts.append('mask_trust=%.2f' % app._mask_trust())
        if app.route_guidance_mode == 'corridor':
            parts.append('bias=%+.1fdeg' % math.degrees(app._route_bias))
        if app.route_steer_limit:
            parts.append('steer_lim=%.0fdeg' % math.degrees(app.route_steer_limit))
        if app.route_hold:
            parts.append('HOLD=%s' % app.route_hold)
        arrow = getattr(app, '_arrow', None)
        if arrow is not None:
            if arrow.get('odometry'):
                parts.append('TURN ARMED %+.0fdeg to go, est %.1fm' % (
                    math.degrees(arrow.get('remaining', 0.0)),
                    arrow.get('est') or 0.0))
            else:
                parts.append('TURN ARMED exit=%.0fdeg' % (math.degrees(arrow['exit']) % 360))
        if getattr(app, '_road_retreat', None) is not None:
            parts.append('ROAD-LOST RETREAT')
        elif getattr(app, '_road_hold', False):
            parts.append('ROAD-LOST HOLD')
        if osm is not None and osm.route is not None:
            # same definition as the router's own log line and the follower's
            # turn distance: forks the route goes straight through do not count.
            # With every fork counted the HUD read 13.8 m while the router log
            # said 54 m at the same instant (170349 video 6:13.97).
            parts.append('next_turn=%.0fm' % osm.route.next_junction_dist(osm.s, osm.junction_min_turn))
        lines.append('   '.join(parts))
    else:
        lines.append("")

    lines.append(EVENTS_HEADER)
    recent = list(reason_log)[-N_REASON_LINES:]
    for _ in range(N_REASON_LINES - len(recent)):
        lines.append("")
    for t, src, text, sig, count in recent:
        suffix = f"  (x{count})" if count > 1 else ""
        lines.append(f"[{t}] ({src}) {text}{suffix}")

    return lines


def _wrap_text(text, max_width, font_scale=HUD_FONT_SCALE, indent='    '):
    """Greedy word wrap to max_width pixels as cv2.putText will draw it.
    Splits on single spaces so the HUD's triple-space column gaps survive
    inside a row; continuation rows are indented."""
    def width(s):
        return cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
    if width(text) <= max_width:
        return [text]
    rows, row = [], ''
    for word in text.split(' '):
        if row == indent and not word:
            continue  # no leading gap on a continuation row
        candidate = word if not row else row + ' ' + word
        if row.strip() and width(candidate) > max_width:
            rows.append(row.rstrip())
            row = indent + word
        else:
            row = candidate
    rows.append(row)
    return rows


def render_hud(lines, width, height=None):
    """The text panel. Without a height it is one row per line at the full
    frame width, as it always was. With one (the map sits beside it) the
    lines are wrapped to the narrower width, and if they then do not fit,
    the recent events give up their OLDEST entries first - so the newest
    event is always on screen and the panel never changes size, which
    --create-video needs."""
    if height is None:
        rows = lines
        height = 20 + HUD_ROW_H * len(rows) + 10
    else:
        max_w, n_rows = width - 20, (height - 30) // HUD_ROW_H
        split = lines.index(EVENTS_HEADER) + 1 if EVENTS_HEADER in lines else len(lines)
        rows = [r for line in lines[:split] for r in _wrap_text(line, max_w)]
        events = [_wrap_text(line, max_w) for line in lines[split:]]
        while events and len(rows) + sum(len(e) for e in events) > n_rows:
            events.pop(0)
        rows = (rows + [r for e in events for r in e])[:n_rows]

    hud = np.zeros((height, width, 3), dtype=np.uint8)
    y = 20
    for row in rows:
        cv2.putText(hud, row, (10, y), cv2.FONT_HERSHEY_SIMPLEX, HUD_FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)
        y += HUD_ROW_H
    return hud


MAP_SPAN_M = 60.0          # default: metres shown top to bottom
MAP_TRAIL_COLOR = (0, 170, 220)
MAP_FIX_COLOR = (0, 255, 255)
MAP_ROUTE_COLOR = (255, 0, 255)
MAP_ROUTE_DONE_COLOR = (110, 40, 110)
MAP_POSITION_COLOR = (0, 230, 0)
MAP_AIM_COLOR = (255, 255, 0)
MAP_GOAL_COLOR = (0, 0, 255)
MAP_LEGEND = ("grey: routable ways at their mapped width (lighter = competition area)   "
              "magenta: route, ring = next turn   yellow: GPS fixes   "
              "green: router's position on the route   cyan +: aim point   red x: QR target")
# route states in which the robot has no route any more - the last
# route_plan is history, not the plan in force
MAP_NO_ROUTE_STATES = ('waiting', 'free', 'failed')


def nmea_latlon(data):
    """(lat, lon) of a gps.nmea_data message, signed the way
    OSMRouter.on_nmea_data signs it, or None for a message without a fix."""
    lat, lon = data.get('lat'), data.get('lon')
    if lat is None or lon is None:
        return None
    return (-lat if data.get('lat_dir') == 'S' else lat,
            -lon if data.get('lon_dir') == 'W' else lon)


class MapTrack:
    """The recorded half of the map panel: every route_plan the router
    published and every GPS fix it was fed, read up front in one pass.

    Recorded rather than taken from the replayed router, for two reasons.
    A --start-sec past the QR code leaves the replayed router with no route
    at all (it never saw the target), while the robot was following one.
    And the code that ran on the robot can lag this repo, so the replayed
    router's plan need not be the one Matty drove. The app is fed the
    recorded route_hint for the same reason - so the map, the HUD's route
    line and the robot's behaviour all describe the same plan."""

    def __init__(self, logfile, name_to_id, gps_stream):
        self.plan_t, self.plans = [], []
        self.fix_t, fixes = [], []
        wanted = {name_to_id[n]: n for n in ('osm_router.route_plan', gps_stream) if n in name_to_id}
        if wanted:
            with LogReader(logfile, only_stream_id=list(wanted)) as log:
                for dt, stream_id, raw in log:
                    data = deserialize(raw)
                    if wanted[stream_id] == 'osm_router.route_plan':
                        self.plan_t.append(dt)
                        self.plans.append(data)
                    else:
                        fix = nmea_latlon(data)
                        if fix is not None:
                            self.fix_t.append(dt)
                            fixes.append(fix)
        self.fixes = np.asarray(fixes, dtype=float).reshape(-1, 2)
        self._route_key, self._route = None, None

    def trail_at(self, t):
        """(lat, lon) array of the fixes up to t, and the time of the last one."""
        n = bisect.bisect_right(self.fix_t, t)
        return self.fixes[:n], (self.fix_t[n - 1] if n else None)

    def route_at(self, t, graph):
        """The route_plan in force at t, as an osm_router.Route in graph's
        frame (so its arclength matches route_hint's progress_m), or None."""
        from osm_router import Route
        i = bisect.bisect_right(self.plan_t, t) - 1
        if i < 0:
            return None
        if self._route_key != (i, id(graph)):
            points = [tuple(p) for p in self.plans[i]['points']]
            self._route = None
            if len(points) >= 2:
                self._route = Route(points, [graph.to_xy(*p) for p in points], [False] * len(points),
                                    tuple(self.plans[i]['goal']))
            self._route_key = (i, id(graph))
        return self._route


def _scale_bar_m(ppm, max_px):
    for m in (500, 200, 100, 50, 20, 10, 5, 2, 1):
        if m * ppm <= max_px:
            return m
    return 1


MAP_COURSE_BASELINE_M = 4.0


def _map_view_bearing(osm, trail_xy, route, hint):
    """Compass bearing (0 = north, clockwise) to put up the map, and where
    it came from. The router's odometry+GPS heading first - smooth, and
    the one it steers turns by. Until that has learned its offset, the
    course over the last few metres of GPS fixes, then the route's own
    direction at the robot. Never the compass: +-30-40 deg off, and
    heading-dependent. North-up only when there is nothing at all."""
    bearing = osm._compass_bearing()
    if bearing is not None:
        return bearing, 'odo+gps'
    if trail_xy is not None and len(trail_xy) >= 2:
        d = np.hypot(*(trail_xy - trail_xy[-1]).T)
        far = np.flatnonzero(d >= MAP_COURSE_BASELINE_M)
        if len(far):
            dx, dy = trail_xy[-1] - trail_xy[far[-1]]
            return math.atan2(dx, dy), 'gps course'
    if route is not None and hint.get('progress_m') is not None:
        tx, ty = route.tangent_at(hint['progress_m'])
        return math.atan2(tx, ty), 'route'
    return 0.0, 'north'


def draw_osm_map(w, h, osm, track, hint, span_m, now):
    """Heading-up map around Matty's latest GPS fix, span_m metres from
    top to bottom - up is the way the robot faces (see _map_view_bearing),
    with a compass rose in the corner for where north went.

    The road network and the local frame come from the replayed router
    (osm.graph, osm.graph.to_xy - metres east/north of the extract's
    centre), so the ways are exactly the ones it plans on, at the width it
    measures off_road_m against. The route, the fixes and the router's
    position come from the log as recorded - see MapTrack. `hint` is the
    last recorded route_hint.

      grey          every way the router may route on; lighter ones are
                    the competition-area graph it is planning on now
      magenta       the planned route - dim behind progress_m, the
                    router's position along it - with a ring at the next
                    turn it counts (route_hint junction_m)
      yellow        the raw GPS fixes so far, the latest one large, with
                    the router's heading estimate as a white arrow
      green         the route point at progress_m, joined to the latest
                    fix: the gap is where the router believes Matty is on
                    the plan vs where the GPS says he is
      cyan +        the aim point in route_hint
      red x         the QR target (the route ends at the nearest way)
    """
    img = np.full((h, w, 3), 28, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    graph = osm.graph
    hint = hint or {}
    route = None if hint.get('state') in MAP_NO_ROUTE_STATES else track.route_at(now, graph)
    trail_ll, fix_time = track.trail_at(now)
    if len(trail_ll):
        tx, ty = graph.to_xy(trail_ll[:, 0], trail_ll[:, 1])
        trail_xy = np.column_stack((tx, ty))
        cx, cy = trail_xy[-1]
    elif route is not None:
        trail_xy = None
        cx, cy = route.xy[0]
    else:
        cv2.putText(img, 'OSM map: waiting for the first GPS fix...', (20, h // 2), font,
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return img
    ppm = h / span_m
    view, view_src = _map_view_bearing(osm, trail_xy, route, hint)
    cos_v, sin_v = math.cos(view), math.sin(view)
    # the robot sits below the middle, so more of the panel is what is ahead
    ox, oy = w / 2, h * 0.62

    def px(xy):
        # heading-up: rotate east/north so that `view` (compass bearing)
        # points up the screen
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        dx, dy = xy[:, 0] - cx, xy[:, 1] - cy
        right, ahead = dx * cos_v - dy * sin_v, dx * sin_v + dy * cos_v
        return np.column_stack((np.round(ox + right * ppm), np.round(oy - ahead * ppm))).astype(np.int32)

    def pt(xy):
        return tuple(int(v) for v in px(xy)[0])

    # --- the road network: whole map dim, the area graph on top of it ---
    whole = getattr(osm, '_map_entry', {}).get('graph', graph)
    layers = [(whole, (48, 48, 48), (85, 85, 85))]
    if graph is not whole:
        layers.append((graph, (80, 80, 80), (150, 150, 150)))
    # rotated, so the panel's corners can reach this far in any direction
    reach_x = reach_y = math.hypot(max(ox, w - ox), max(oy, h - oy)) / ppm + 5
    for g, surface, centre in layers:
        a, b = g.seg_a, g.seg_b
        if not len(a):
            continue
        visible = np.flatnonzero((np.minimum(a[:, 0], b[:, 0]) < cx + reach_x)
                                 & (np.maximum(a[:, 0], b[:, 0]) > cx - reach_x)
                                 & (np.minimum(a[:, 1], b[:, 1]) < cy + reach_y)
                                 & (np.maximum(a[:, 1], b[:, 1]) > cy - reach_y))
        if not len(visible):
            continue
        segs = np.stack([px(a[visible]), px(b[visible])], axis=1)
        thick = np.maximum(1, np.round(2 * g.seg_halfwidth[visible] * ppm)).astype(int)
        for t in np.unique(thick):
            cv2.polylines(img, list(segs[thick == t]), False, surface, int(t), cv2.LINE_AA)
        cv2.polylines(img, list(segs), False, centre, 1, cv2.LINE_AA)

    # --- the plan ---
    here = None
    if route is not None:
        s = hint.get('progress_m')
        if s is None:
            cv2.polylines(img, [px(route.xy)], False, MAP_ROUTE_COLOR, 3, cv2.LINE_AA)
        else:
            here = np.asarray(route.xy_at(s))
            i = int(np.searchsorted(route.cum, s, side='right'))
            cv2.polylines(img, [px(np.vstack([route.xy[:i], here]))], False, MAP_ROUTE_DONE_COLOR, 2, cv2.LINE_AA)
            cv2.polylines(img, [px(np.vstack([here, route.xy[i:]]))], False, MAP_ROUTE_COLOR, 3, cv2.LINE_AA)
            junction_m = hint.get('junction_m')
            if junction_m is not None and s + junction_m < route.total - 0.5:
                cv2.circle(img, pt(route.xy_at(s + junction_m)), 9, MAP_ROUTE_COLOR, 2, cv2.LINE_AA)
        cv2.circle(img, pt(route.xy[-1]), 5, MAP_ROUTE_COLOR, -1, cv2.LINE_AA)
        gx, gy = pt(graph.to_xy(*route.goal_ll))
        cv2.line(img, (gx - 7, gy - 7), (gx + 7, gy + 7), MAP_GOAL_COLOR, 2, cv2.LINE_AA)
        cv2.line(img, (gx - 7, gy + 7), (gx + 7, gy - 7), MAP_GOAL_COLOR, 2, cv2.LINE_AA)
    if hint.get('lat') is not None and hint.get('lon') is not None:
        cv2.drawMarker(img, pt(graph.to_xy(hint['lat'], hint['lon'])), MAP_AIM_COLOR,
                       cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

    # --- the GPS ---
    if trail_xy is not None:
        trail = px(trail_xy)
        on_screen = (trail[:, 0] >= 0) & (trail[:, 0] < w) & (trail[:, 1] >= 0) & (trail[:, 1] < h)
        for x, y in trail[:-1][on_screen[:-1]]:
            cv2.circle(img, (int(x), int(y)), 2, MAP_TRAIL_COLOR, -1, cv2.LINE_AA)
        fix = (int(trail[-1][0]), int(trail[-1][1]))
        if here is not None:
            cv2.line(img, fix, pt(here), MAP_POSITION_COLOR, 1, cv2.LINE_AA)
        bearing = osm._compass_bearing()
        if bearing is not None:
            rel = bearing - view    # 0 = straight up whenever the view follows this estimate
            tip = (int(round(fix[0] + 30 * math.sin(rel))), int(round(fix[1] - 30 * math.cos(rel))))
            cv2.arrowedLine(img, fix, tip, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(img, fix, 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, fix, 5, MAP_FIX_COLOR, -1, cv2.LINE_AA)
    if here is not None:
        cv2.circle(img, pt(here), 5, MAP_POSITION_COLOR, -1, cv2.LINE_AA)

    # --- labels, on a backing strip so a road under them stays readable ---
    parts = ['OSM route + GPS', 'up=%.0fdeg (%s)' % (math.degrees(view) % 360, view_src),
             'route=%s' % hint.get('state', '--')]
    if fix_time is not None:
        lat, lon = trail_ll[-1]
        parts.append('fix %.6f,%.6f (%.1fs old)' % (lat, lon, (now - fix_time).total_seconds()))
    if route is not None:
        s = hint.get('progress_m')
        parts.append('s=%s/%.0fm' % ('--' if s is None else '%.0f' % s, route.total))
    if hint.get('cross_track_m') is not None:
        parts.append('cross=%+.1fm' % hint['cross_track_m'])
    if hint.get('off_road_m') is not None:
        parts.append('off_road=%.2fm' % hint['off_road_m'])
    header = _wrap_text('   '.join(parts), w - 60)
    img[:10 + HUD_ROW_H * len(header)] = 28
    for i, row in enumerate(header):
        cv2.putText(img, row, (10, 20 + HUD_ROW_H * i), font, HUD_FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)

    # compass rose: where north is, now that up is forward
    nx, ny = w - 35, 42 + HUD_ROW_H * len(header)
    ux, uy = -math.sin(view), -math.cos(view)
    cv2.circle(img, (nx, ny), 22, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.arrowedLine(img, (int(nx - 16 * ux), int(ny - 16 * uy)), (int(nx + 16 * ux), int(ny + 16 * uy)),
                    (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
    (tw, th), _ = cv2.getTextSize('N', font, 0.5, 1)
    cv2.putText(img, 'N', (int(nx + 32 * ux - tw / 2), int(ny + 32 * uy + th / 2)), font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    legend = _wrap_text(MAP_LEGEND, w - 20, 0.4)
    legend_top = h - 8 - HUD_ROW_H * len(legend)
    img[legend_top:] = 28
    y = legend_top + 14
    for row in legend:
        cv2.putText(img, row, (10, y), font, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
        y += HUD_ROW_H

    bar_m = _scale_bar_m(ppm, w / 6)
    bar_px = int(round(bar_m * ppm))
    y_bar = legend_top - 12
    cv2.line(img, (10, y_bar), (10 + bar_px, y_bar), (255, 255, 255), 2)
    for x in (10, 10 + bar_px):
        cv2.line(img, (x, y_bar - 5), (x, y_bar + 5), (255, 255, 255), 2)
    cv2.putText(img, '%d m' % bar_m, (16 + bar_px, y_bar + 5), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_frame(dt, depth_mm, color_img, mask, app, zones, reason_log, max_depth_mm, depth_scale, color_size,
                  right_panel='color', osm=None, map_track=None, route_hint=None, map_span_m=None):
    """map_span_m None = no map panel (old layout)."""
    h, w = depth_mm.shape
    depth_panel = cv2.resize(colorize_depth(depth_mm, max_depth_mm), (w * depth_scale, h * depth_scale),
                              interpolation=cv2.INTER_NEAREST)

    draw_zone(depth_panel, zones.base_rows, zones.center_cols, (90, 90, 90), '', depth_scale, thickness=1)
    draw_zone(depth_panel, zones.rows, zones.left_cols, (255, 255, 0), 'L', depth_scale)
    draw_zone(depth_panel, zones.rows, zones.center_cols, (0, 255, 0), 'C', depth_scale)
    draw_zone(depth_panel, zones.rows, zones.right_cols, (0, 0, 255), 'R', depth_scale)
    draw_zone(depth_panel, zones.ground_rows, zones.ground_cols, (0, 255, 255), 'GROUND', depth_scale)
    draw_shift_gauge(depth_panel, zones, h, depth_scale)
    draw_free_space_bins(depth_panel, app, zones, depth_scale)

    if color_size is not None:
        cw, ch = color_size
        show_mask = right_panel == 'mask' and mask is not None and color_img is not None
        right_source = draw_drivable_area(color_img, mask) if show_mask else color_img
        if right_source is not None:
            color_panel = cv2.resize(right_source, (cw, ch))
            if right_panel == 'mask' and mask is None:
                cv2.putText(color_panel, 'waiting for nn_mask...', (20, ch - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            color_panel = np.full((ch, cw, 3), 40, dtype=np.uint8)
            cv2.putText(color_panel, 'waiting for color...', (20, ch // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 2, cv2.LINE_AA)
    else:
        color_panel = None

    lines = build_hud_lines(dt, app, zones, reason_log, osm)
    show_map = osm is not None and map_track is not None and map_span_m is not None
    if show_map and color_panel is not None:
        # map under the color image, text under the depth map. Room is left
        # for every event and a few status lines to wrap once at the
        # narrower width - see render_hud.
        bottom_h = 30 + HUD_ROW_H * (len(lines) + N_REASON_LINES + 3)
        map_panel = draw_osm_map(color_panel.shape[1], bottom_h, osm, map_track, route_hint, map_span_m, dt)
        return np.vstack([np.hstack([depth_panel, color_panel]),
                          np.hstack([render_hud(lines, depth_panel.shape[1], bottom_h), map_panel])])
    if show_map:
        # --no-color: the map takes the color image's slot
        map_panel = draw_osm_map(depth_panel.shape[0], depth_panel.shape[0], osm, map_track, route_hint,
                                 map_span_m, dt)
        top = np.hstack([depth_panel, map_panel])
    else:
        top = depth_panel if color_panel is None else np.hstack([depth_panel, color_panel])
    return np.vstack([top, render_hud(lines, top.shape[1])])


MAP_ZOOM_KEYS = {ord('+'): 1 / 1.5, ord('='): 1 / 1.5, ord('-'): 1.5}


def wait_key(delay, map_view, redraw):
    """cv2.waitKey that consumes the map zoom keys itself: the current
    frame is redrawn at the new zoom and the wait goes on, so zooming a
    paused replay does not also step it."""
    while True:
        key = cv2.waitKey(delay) & 0xFF
        if key not in MAP_ZOOM_KEYS or map_view['span_m'] is None:
            return key
        map_view['span_m'] = min(2000.0, max(10.0, map_view['span_m'] * MAP_ZOOM_KEYS[key]))
        cv2.imshow('Matty obstacle viewer', redraw())


def read_logfile(logfile, max_depth_mm=4000, show_color=True, depth_scale=2, start_sec=0.0, writer=None,
                  right_panel='color', verbose=False, map_span_m=MAP_SPAN_M):
    full_cfg = lookup_config(logfile)['robot']
    modules = full_cfg['modules']
    assert modules['app']['driver'] == 'tulak_obstacle:TulakObstacle', \
        f"expected app driver tulak_obstacle:TulakObstacle, got {modules['app']['driver']}"
    assert modules['obstdet3d_zones']['driver'] == 'osgar.obstdet3d_zones:ObstacleDetector3DZones', \
        f"expected obstdet3d_zones driver osgar.obstdet3d_zones:ObstacleDetector3DZones, got {modules['obstdet3d_zones']['driver']}"

    app = TulakObstacle(modules['app']['init'], FakeBus())
    zones = ObstacleDetector3DZones(modules['obstdet3d_zones']['init'], FakeBus())
    # mirrors how `osgar.replay --verbose` sets this - see osgar/node.py.
    # Prints every frame when on (e.g. obstdet3d_zones's unconditional
    # per-frame pitch/rows/zones line) - fine for a short --start-sec
    # window, but at full-log length it'll dominate the small 5-line
    # "recent events" panel above; the dedicated "IMU raw" HUD line does
    # NOT depend on this and is the better fit for the hand-tilt test.
    app.verbose = verbose
    zones.verbose = verbose
    nodes = {'app': app, 'obstdet3d_zones': zones}

    # OSM route planning (osm_router.py) - replayed the same way as the
    # other two nodes when the run was recorded with a config that had it
    # (matty-tulak-osm.json). Old logs simply have no 'osm_router' module
    # and everything below stays None, so this file keeps working on every
    # run recorded before the mode existed.
    osm = None
    if 'osm_router' in modules:
        from osm_router import OSMRouter
        osm = OSMRouter(modules['osm_router']['init'], FakeBus())
        osm.verbose = verbose
        nodes['osm_router'] = osm
    router = build_router(full_cfg['links'], set(nodes.keys()))

    stream_names = lookup_stream_names(logfile)
    name_to_id = {n: i + 1 for i, n in enumerate(stream_names)}

    needed = set(router.keys()) | {'oak.depth'}
    map_track = None
    if osm is not None and map_span_m is not None:
        # the stream feeding the router's GPS input, from the recorded links
        gps_stream = next((src for src, dsts in router.items() if ('osm_router', 'nmea_data') in dsts), None)
        map_track = MapTrack(logfile, name_to_id, gps_stream)
        needed.add('osm_router.route_hint')
    show_color = show_color and 'oak.color' in name_to_id
    if show_color:
        needed.add('oak.color')
    only_ids = [name_to_id[n] for n in needed if n in name_to_id]
    id_to_name = {v: k for k, v in name_to_id.items()}

    tmp_dir = tempfile.mkdtemp(prefix='view_obstacle_')
    read_video_frame = make_video_reader(os.path.join(tmp_dir, 'stream.h26x'))

    reason_log = deque(maxlen=64)
    color_img = None
    color_size = None
    last_mask = None
    paused = False
    last_route_hint = None      # as recorded - see MapTrack
    map_view = {'span_m': map_span_m if map_track is not None else None}

    with LogReader(logfile, only_stream_id=only_ids, clip_start_time_sec=start_sec) as log:
        for dt, stream_id, raw in log:
            name = id_to_name[stream_id]
            data = deserialize(raw)

            if name in router:
                for module_name, channel in router[name]:
                    call_handler(nodes[module_name], channel, dt, data, module_name, reason_log)

            if name == 'oak.nn_mask':
                last_mask = data
            elif name == 'osm_router.route_hint':
                last_route_hint = data

            if name == 'oak.depth':
                if show_color and color_img is not None and color_size is None:
                    ch = data.shape[0] * depth_scale
                    cw = round(ch * color_img.shape[1] / color_img.shape[0])
                    color_size = (cw, ch)
                elif show_color and color_size is None:
                    ch = data.shape[0] * depth_scale
                    color_size = (round(ch * 16 / 9), ch)  # placeholder aspect until first color frame

                def redraw(depth_mm=data, dt=dt):
                    return render_frame(dt, depth_mm, color_img, last_mask, app, zones, reason_log,
                                        max_depth_mm, depth_scale, color_size if show_color else None,
                                        right_panel=right_panel, osm=osm, map_track=map_track,
                                        route_hint=last_route_hint, map_span_m=map_view['span_m'])

                frame = redraw()
                cv2.imshow('Matty obstacle viewer', frame)
                if writer is not None:
                    writer.write(frame)

                key = wait_key(0 if paused else 1, map_view, redraw)
                if key == 0x20:  # space - toggle pause / step
                    paused = not paused
                    if paused:
                        key = wait_key(0, map_view, redraw)
                if key == ord('s'):
                    cv2.imwrite('save_frame.jpg', redraw())  # at the current map zoom
                    print('saved save_frame.jpg')
                if key in (27, ord('q')):
                    break

            elif show_color and name == 'oak.color':
                img = read_video_frame(data, i_frame_only=False)
                if img is not None:
                    color_img = img


def normalize_turn(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class VideoWriter:
    """Lazily opens a cv2.VideoWriter once the first composite frame's size
    is known (it depends on --depth-scale and the color image's aspect).
    cv2.VideoWriter only ever writes video frames - there is no code path
    here that attaches an audio track, so --create-video output is silent
    by construction; there is no sound to disable."""
    def __init__(self, filename, fps):
        self.filename = filename
        self.fps = fps
        self._writer = None

    def write(self, frame):
        if self._writer is None:
            self._writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*"mp4v"),
                                            self.fps, (frame.shape[1], frame.shape[0]))
        self._writer.write(frame)

    def release(self):
        if self._writer is not None:
            self._writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logfile', nargs='+', help='recorded log file(s) from the matty-tulak-obstacle config')
    parser.add_argument('--max-depth-mm', type=int, default=4000, help='depth colormap clip range, in mm')
    parser.add_argument('--depth-scale', type=int, default=2, help='upscale factor for the depth panel')
    parser.add_argument('--no-color', action='store_true', help='skip decoding/showing the color image')
    parser.add_argument('--right-panel', choices=['color', 'mask'], default='color',
                         help="what to show next to the depth map: plain color image, or the drivable-area "
                              "(nn_mask) overlay on the color image, same view as robotem-rovne/view_mask.py "
                              "(default: color)")
    parser.add_argument('--no-map', action='store_true',
                         help='skip the OSM route + GPS map panel (only drawn for runs recorded with osm_router)')
    parser.add_argument('--map-span-m', type=float, default=MAP_SPAN_M,
                         help='metres the map panel shows top to bottom; +/- zoom while viewing '
                              '(default: %(default)s)')
    parser.add_argument('--start-sec', type=float, default=0.0, help='skip ahead to this time in the log')
    parser.add_argument('--verbose', '-v', action='store_true',
                         help="enable each replayed node's own verbose printouts (fed into \"recent events\" "
                              "above, same as osgar.replay --verbose) - noisy at full-log length; the IMU raw "
                              "HUD line does not need this")
    parser.add_argument('--create-video', help='filename of output video (silent - cv2.VideoWriter never '
                                                 'writes an audio track, so there is nothing to mute)')
    parser.add_argument('--fps', type=float, default=10, help='fps for --create-video')
    args = parser.parse_args()

    for logfile in args.logfile:
        writer = VideoWriter(args.create_video, args.fps) if args.create_video is not None else None

        read_logfile(logfile, max_depth_mm=args.max_depth_mm, show_color=not args.no_color,
                     depth_scale=args.depth_scale, start_sec=args.start_sec, writer=writer,
                     right_panel=args.right_panel, verbose=args.verbose,
                     map_span_m=None if args.no_map else args.map_span_m)

        if writer is not None:
            writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# vim: expandtab sw=4 ts=4
