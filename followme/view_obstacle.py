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

  Controls (same as robotem-rovne/view_mask.py):
    space - pause / step one frame
    s     - save current composite frame to save_frame.jpg
    q/ESC - quit
"""
import argparse
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


def colorize_depth(depth_mm, max_mm, far_fill_mm=None):
    """near = warm/red, far = cool/blue, invalid (0) = black, and the
    driver's synthetic "far" fill in flat grey.

    That last part matters and is not cosmetic. The oak driver replaces
    invalid pixels in the upper frame with depth_far_mask_value_mm (15000)
    to say "nothing near here" - see depth_far_mask_* in oak_camera_v3.
    Those pixels are not measurements. Coloured on the same scale as the
    rest they clip to the far end of the JET map and read as a clean,
    confident, uniformly distant background, which is exactly the part of
    this picture that impresses people who ask why their OAK-D looks
    noisier. Over 400 frames of run 133922, 47.5% of the frame was that
    fill, 23.3% was invalid and only 29.3% was a real stereo reading -
    93% fill inside the centre obstacle window. The detector already
    treats the fill with suspicion (far_fill_suspect_m); the viewer
    should not show it as data."""
    valid = depth_mm > 0
    clipped = np.clip(depth_mm, 0, max_mm).astype(np.float32)
    inverted = 255 - clipped / max_mm * 255
    color = cv2.applyColorMap(inverted.astype(np.uint8), cv2.COLORMAP_JET)
    color[~valid] = (0, 0, 0)
    if far_fill_mm:
        color[depth_mm == far_fill_mm] = (70, 70, 70)
    return color


def depth_census(depth_mm, far_fill_mm):
    """(real, fill, invalid) as percentages of the frame."""
    px = float(depth_mm.size)
    fill = float((depth_mm == far_fill_mm).sum()) if far_fill_mm else 0.0
    invalid = float((depth_mm == 0).sum())
    return (100.0 * (px - fill - invalid) / px, 100.0 * fill / px, 100.0 * invalid / px)


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

    lines.append("-- recent events --")
    recent = list(reason_log)[-N_REASON_LINES:]
    for _ in range(N_REASON_LINES - len(recent)):
        lines.append("")
    for t, src, text, sig, count in recent:
        suffix = f"  (x{count})" if count > 1 else ""
        lines.append(f"[{t}] ({src}) {text}{suffix}")

    return lines


def render_frame(dt, depth_mm, color_img, mask, app, zones, reason_log, max_depth_mm, depth_scale, color_size,
                  right_panel='color', osm=None, far_fill_mm=None):
    h, w = depth_mm.shape
    depth_panel = cv2.resize(colorize_depth(depth_mm, max_depth_mm, far_fill_mm),
                              (w * depth_scale, h * depth_scale),
                              interpolation=cv2.INTER_NEAREST)
    real, fill, invalid = depth_census(depth_mm, far_fill_mm)
    cv2.putText(depth_panel, 'depth %.0f%% measured  %.0f%% far-fill  %.0f%% invalid'
                % (real, fill, invalid), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)

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
        top = np.hstack([depth_panel, color_panel])
    else:
        top = depth_panel

    lines = build_hud_lines(dt, app, zones, reason_log, osm)
    hud = np.zeros((20 + 18 * len(lines) + 10, top.shape[1], 3), dtype=np.uint8)
    y = 20
    for line in lines:
        cv2.putText(hud, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18

    return np.vstack([top, hud])


def read_logfile(logfile, max_depth_mm=4000, show_color=True, depth_scale=2, start_sec=0.0, writer=None,
                  far_fill_mm=None,
                  right_panel='color', verbose=False):
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

    with LogReader(logfile, only_stream_id=only_ids, clip_start_time_sec=start_sec) as log:
        for dt, stream_id, raw in log:
            name = id_to_name[stream_id]
            data = deserialize(raw)

            if name in router:
                for module_name, channel in router[name]:
                    call_handler(nodes[module_name], channel, dt, data, module_name, reason_log)

            if name == 'oak.nn_mask':
                last_mask = data

            if name == 'oak.depth':
                if show_color and color_img is not None and color_size is None:
                    ch = data.shape[0] * depth_scale
                    cw = round(ch * color_img.shape[1] / color_img.shape[0])
                    color_size = (cw, ch)
                elif show_color and color_size is None:
                    ch = data.shape[0] * depth_scale
                    color_size = (round(ch * 16 / 9), ch)  # placeholder aspect until first color frame

                frame = render_frame(dt, data, color_img, last_mask, app, zones, reason_log,
                                      max_depth_mm, depth_scale, color_size if show_color else None,
                                      right_panel=right_panel, osm=osm, far_fill_mm=far_fill_mm)
                cv2.imshow('Matty obstacle viewer', frame)
                if writer is not None:
                    writer.write(frame)

                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key == 0x20:  # space - toggle pause / step
                    paused = not paused
                    if paused:
                        key = cv2.waitKey(0) & 0xFF
                if key == ord('s'):
                    cv2.imwrite('save_frame.jpg', frame)
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
    parser.add_argument('--start-sec', type=float, default=0.0, help='skip ahead to this time in the log')
    parser.add_argument('--no-mark-far-fill', action='store_true',
                         help="colour the driver's synthetic far-field fill (depth_far_mask_value_mm) like "
                              "any other depth instead of flat grey - see colorize_depth for why it is "
                              "marked by default")
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

        # what THIS log's oak module was told to fill the far field with,
        # so the viewer can mark it rather than colour it as a measurement
        far_fill = lookup_config(logfile)['robot']['modules'].get('oak', {}).get(
                'init', {}).get('depth_far_mask_value_mm', 15000)
        if args.no_mark_far_fill:
            far_fill = None
        read_logfile(logfile, max_depth_mm=args.max_depth_mm, show_color=not args.no_color,
                     depth_scale=args.depth_scale, start_sec=args.start_sec, writer=writer,
                     right_panel=args.right_panel, verbose=args.verbose, far_fill_mm=far_fill)

        if writer is not None:
            writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# vim: expandtab sw=4 ts=4
