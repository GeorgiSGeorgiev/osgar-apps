#!/usr/bin/python
"""
  View Matty's obstacle-avoidance run: depth map with the live obstacle/
  ground detection windows overlaid, the color image, and a HUD showing
  what command was sent to the platform and why.

  This does NOT reimplement tulak_obstacle.py / obstdet3d_zones.py logic -
  it replays the ACTUAL TulakObstacle and ObstacleDetector3DZones classes
  against the inputs recorded in the log (obstacle_zones, ground_hazard,
  rotation, pose2d, nn_mask, qr_code, nmea_data, ...), using the log's own
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
    actually drove last_dir/steering rather than a naive raw centroid."""
    h_img, w_img = color_img.shape[:2]
    m = mask.copy()
    mh, mw = m.shape
    m[:mh // 2, :] = 0

    center_y, center_x = mask_center(m)
    center_x = int(center_x * w_img / mw)
    center_y = int(center_y * h_img / mh)

    mask_resized = cv2.resize(m, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
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
    for i in range(n):
        x0, x1 = int(edges[i] * scale), int(edges[i + 1] * scale)
        d = app.depth_profile[i] if app.depth_profile else None
        if d is None:
            color = (80, 80, 80)
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


def build_hud_lines(dt, app, zones, reason_log):
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

    lines.append(f"ground_hazard: active={app.ground_hazard_active}  streak={app.ground_hazard_streak}")

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

    lines.append("-- recent events --")
    recent = list(reason_log)[-N_REASON_LINES:]
    for _ in range(N_REASON_LINES - len(recent)):
        lines.append("")
    for t, src, text, sig, count in recent:
        suffix = f"  (x{count})" if count > 1 else ""
        lines.append(f"[{t}] ({src}) {text}{suffix}")

    return lines


def render_frame(dt, depth_mm, color_img, mask, app, zones, reason_log, max_depth_mm, depth_scale, color_size,
                  right_panel='color'):
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
        top = np.hstack([depth_panel, color_panel])
    else:
        top = depth_panel

    lines = build_hud_lines(dt, app, zones, reason_log)
    hud = np.zeros((20 + 18 * len(lines) + 10, top.shape[1], 3), dtype=np.uint8)
    y = 20
    for line in lines:
        cv2.putText(hud, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18

    return np.vstack([top, hud])


def read_logfile(logfile, max_depth_mm=4000, show_color=True, depth_scale=2, start_sec=0.0, writer=None,
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
                                      right_panel=right_panel)
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
                     right_panel=args.right_panel, verbose=args.verbose)

        if writer is not None:
            writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# vim: expandtab sw=4 ts=4
