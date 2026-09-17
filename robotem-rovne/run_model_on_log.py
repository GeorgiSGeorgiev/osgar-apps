#!/usr/bin/python
"""
  Run a road-detection blob over the colour frames of a recorded log and
  compare it with the mask that was recorded at the time.

  A .blob is compiled for the OAK's VPU, so this needs the camera plugged
  in - it just does not need the robot, the road, or daylight. The frames
  come from the log, the inference from the device.

      python run_model_on_log.py <logfile> [--blob models/xxx.blob]

  What it reports: how much of the frame each model calls road, how far
  apart their steering centroids land (in degrees of bearing, the unit
  that actually reaches the controller), and how often they disagree pixel
  by pixel - all measured only where the two models' fields of view
  overlap.

  CAVEAT worth knowing before trusting the numbers. The recorded colour
  stream is 1920x1080, which is itself a CROP of the 4:3 sensor - the top
  and bottom eighths that a 640x480 model sees live are not in the log.
  They are padded here by edge replication and excluded from every
  comparison, but the network still sees them. The middle 75% is faithful;
  the near-ground band at the very bottom is not. For a verdict on the
  near field, record a short run with the new model instead.
"""
import argparse
import os
import tempfile

import cv2
import numpy as np

from osgar.logger import LogReader, lookup_stream_names
from osgar.lib.serialize import deserialize

SENSOR_ASPECT = 4.0 / 3.0

KEY_FRAME_PREFIXES = (bytes.fromhex('00000001 460150'),   # h265 IDR
                      bytes.fromhex('00000001 0950'))     # h264 IDR


def decode_color(logfile, t0, t1, limit):
    """(timestamp, BGR frame) for the colour stream, decoded in one pass.

    Every packet is one encoded frame, so the whole window is written out
    as an elementary stream and decoded sequentially - decoding each frame
    from the start of its GOP instead is quadratic and, over a 400 s log,
    slower than the inference it is feeding."""
    names = lookup_stream_names(logfile)
    nid = {n: i + 1 for i, n in enumerate(names)}
    assert 'oak.color' in nid, 'log has no oak.color stream'
    stamps = []
    path = os.path.join(tempfile.gettempdir(), 'run_model_on_log.h26x')
    with open(path, 'wb') as out:
        with LogReader(logfile, only_stream_id=[nid['oak.color']]) as log:
            for dt, _sid, raw in log:
                t = dt.total_seconds()
                if t < t0:
                    continue
                if t1 and t > t1:
                    break
                data = deserialize(raw)
                if not stamps and not data.startswith(KEY_FRAME_PREFIXES):
                    continue        # a decoder has to start on a key frame
                out.write(data)
                stamps.append(t)
                if limit and len(stamps) >= limit:
                    break
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield (stamps[i] if i < len(stamps) else None), frame
        i += 1
    cap.release()


def read_masks(logfile, t0, t1):
    """(timestamp, mask) as recorded by whatever model was flown."""
    names = lookup_stream_names(logfile)
    nid = {n: i + 1 for i, n in enumerate(names)}
    if 'oak.nn_mask' not in nid:
        return []
    out = []
    with LogReader(logfile, only_stream_id=[nid['oak.nn_mask']]) as log:
        for dt, _sid, raw in log:
            t = dt.total_seconds()
            if t < t0:
                continue
            if t1 and t > t1:
                break
            out.append((t, deserialize(raw)))
    return out


def to_model_input(frame_1080p, width, height):
    """The colour frame as the model would have been fed it live.

    requestOutput() crops to the requested aspect out of the FULL sensor,
    so a 4:3 model input is the whole sensor while the recorded 16:9 frame
    is its middle 75%. The missing bands are replicated rather than left
    black - a black bar is not something the network has ever seen, and it
    reads as an obstacle; replication at least continues sky and ground.

    Returns (input image, rows of it that came from real pixels)."""
    h, w = frame_1080p.shape[:2]
    want = width / float(height)
    sensor_h = int(round(w / SENSOR_ASPECT))
    if want < w / float(h) and sensor_h > h:
        pad = (sensor_h - h) // 2
        frame = cv2.copyMakeBorder(frame_1080p, pad, sensor_h - h - pad, 0, 0,
                                   cv2.BORDER_REPLICATE)
        band = (pad, pad + h)
    else:
        frame, band = frame_1080p, (0, h)
    ch, cw = frame.shape[:2]
    if want > cw / float(ch):                   # too tall - crop rows
        new_h = int(round(cw / want))
        y0 = (ch - new_h) // 2
        frame = frame[y0:y0 + new_h]
        band = (band[0] - y0, band[1] - y0)
    elif want < cw / float(ch):                 # too wide - crop columns
        new_w = int(round(ch * want))
        x0 = (cw - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    scale = height / float(frame.shape[0])
    real = (max(0, int(band[0] * scale)), min(height, int(band[1] * scale)))
    return cv2.resize(frame, (width, height)), real


def mask_to_sensor(mask, size=256):
    """Mask resampled onto a common grid covering the whole 4:3 sensor, so
    masks from models with different crops can be compared at all.

    Returns (grid, valid); valid marks where this model actually looked."""
    mh, mw = mask.shape
    grid_w = size
    grid_h = int(round(size / SENSOR_ASPECT))
    if mw / float(mh) >= SENSOR_ASPECT:
        box_w = grid_w
        box_h = int(round(grid_w * mh / float(mw)))
    else:
        box_h = grid_h
        box_w = int(round(grid_h * mw / float(mh)))
    big = cv2.resize(mask.astype(np.uint8), (box_w, box_h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((grid_h, grid_w), dtype=np.uint8)
    valid = np.zeros((grid_h, grid_w), dtype=bool)
    x0, y0 = (grid_w - box_w) // 2, (grid_h - box_h) // 2
    sx, sy = max(0, x0), max(0, y0)
    bx, by = max(0, -x0), max(0, -y0)
    cw = min(box_w - bx, grid_w - sx)
    ch = min(box_h - by, grid_h - sy)
    out[sy:sy + ch, sx:sx + cw] = big[by:by + ch, bx:bx + cw]
    valid[sy:sy + ch, sx:sx + cw] = True
    return out, valid


def centroid_bearing(mask, valid, hfov_deg):
    """Horizontal centroid of the drivable area in degrees off centre, on
    the common sensor grid - sky half dropped and near rows weighted
    exactly as tulak_obstacle does before it steers on this."""
    m = mask.copy()
    sky = m.shape[0] // 2
    m[:sky, :] = 0
    m[~valid] = 0
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return None
    weights = (ys - sky) / float(max(1, m.shape[0] - sky))
    if weights.sum() <= 0:
        return None
    half = m.shape[1] / 2.0
    cx = (xs * weights).sum() / weights.sum()
    return float(np.degrees(np.arctan((cx - half) / half * np.tan(np.radians(hfov_deg / 2)))))


def summarize(rows):
    def stat(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return 'n/a'
        v = np.array(vals, dtype=float)
        return 'mean %7.3f  median %7.3f  p10 %7.3f  p90 %7.3f' % (
                v.mean(), np.median(v), np.percentile(v, 10), np.percentile(v, 90))

    print()
    print('%d frames' % len(rows))
    print('  road fraction, new model      ', stat('new_frac'))
    print('  road fraction, recorded       ', stat('old_frac'))
    print('  road fraction, shared view new', stat('new_frac_shared'))
    print('  road fraction, shared view old', stat('old_frac_shared'))
    print('  steering centroid, new [deg]  ', stat('new_bearing'))
    print('  steering centroid, old [deg]  ', stat('old_bearing'))
    print('  pixel agreement, shared view  ', stat('agree'))
    diffs = [r['new_bearing'] - r['old_bearing'] for r in rows
             if r.get('new_bearing') is not None and r.get('old_bearing') is not None]
    if diffs:
        d = np.abs(np.array(diffs))
        print('  |centroid difference| [deg]    mean %.2f  median %.2f  p90 %.2f  max %.2f'
              % (d.mean(), np.median(d), np.percentile(d, 90), d.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logfile')
    ap.add_argument('--blob', default='models/redroad-v2-20260826-rdrd-s-640x480-rgb-chw-2shaves.blob')
    ap.add_argument('--input-size', default='640x480', help='WxH the blob expects')
    ap.add_argument('--hfov-deg', type=float, default=69.0,
                    help='colour camera horizontal FOV over the full 4:3 sensor')
    ap.add_argument('--t0', type=float, default=0.0)
    ap.add_argument('--t1', type=float, default=0.0)
    ap.add_argument('--limit', type=int, default=300, help='frames; 0 = the whole log')
    ap.add_argument('--rgb', action='store_true',
                    help='feed RGB instead of BGR - the blob name says rgb while the '
                         'driver feeds BGR888p, so this is worth measuring both ways')
    ap.add_argument('--video', help='write a side-by-side overlay video here')
    args = ap.parse_args()

    import depthai as dai
    assert os.path.exists(args.blob), args.blob
    assert dai.Device.getAllAvailableDevices(), (
            "no OAK device found - a .blob only runs on the camera's VPU, "
            "so this test needs it plugged in")
    W, H = (int(v) for v in args.input_size.split('x'))

    pipeline = dai.Pipeline()
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(args.blob)
    nn.setNumInferenceThreads(2)
    q_in = nn.input.createInputQueue()
    q_out = nn.out.createOutputQueue()
    pipeline.start()

    recorded = read_masks(args.logfile, args.t0, args.t1)
    rec_i = 0
    writer = None
    rows = []
    layer = None
    for t, frame in decode_color(args.logfile, args.t0, args.t1, args.limit):
        model_in, _real_rows = to_model_input(frame, W, H)
        send = cv2.cvtColor(model_in, cv2.COLOR_BGR2RGB) if args.rgb else model_in
        img = dai.ImgFrame()
        img.setCvFrame(send, dai.ImgFrame.Type.BGR888p)
        q_in.send(img)
        packet = q_out.get()
        if layer is None:
            names = list(packet.getAllLayerNames())
            print('blob outputs:', names)
            layer = next((n for n in ('logits', 'redroad_output') if n in names), names[0])
            print('reading:', layer)
        logits = np.array(packet.getTensor(layer)).reshape((H // 2, W // 2))
        new_mask = (logits > 0).astype(np.uint8)

        while rec_i + 1 < len(recorded) and t is not None and recorded[rec_i + 1][0] <= t:
            rec_i += 1
        old_mask = recorded[rec_i][1] if recorded else None

        new_grid, new_valid = mask_to_sensor(new_mask)
        row = dict(t=t, new_frac=float(new_mask.mean()),
                   new_bearing=centroid_bearing(new_grid, new_valid, args.hfov_deg))
        if old_mask is not None:
            old_grid, old_valid = mask_to_sensor(old_mask)
            both = new_valid & old_valid
            both[:both.shape[0] // 2, :] = False   # the half that steers
            row.update(old_frac=float(old_mask.mean()),
                       old_bearing=centroid_bearing(old_grid, old_valid, args.hfov_deg),
                       agree=float((new_grid[both] == old_grid[both]).mean()) if both.any() else None,
                       new_frac_shared=float(new_grid[both].mean()) if both.any() else None,
                       old_frac_shared=float(old_grid[both].mean()) if both.any() else None)
        rows.append(row)

        if args.video:
            from osgar.lib.nn_mask import mask_on_color
            h_img, w_img = frame.shape[:2]
            panels = []
            for m, colour, label in ((old_mask, (0, 0, 255), 'recorded'),
                                     (new_mask, (0, 255, 0), os.path.basename(args.blob)[:28])):
                if m is None:
                    continue
                mm = m.copy()
                mm[:mm.shape[0] // 2, :] = 0
                placed, _ = mask_on_color(mm, w_img, h_img)
                over = np.zeros_like(frame)
                over[placed == 1] = colour
                panel = cv2.addWeighted(frame, 1.0, over, 0.5, 0)
                cv2.putText(panel, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, colour, 4)
                panels.append(panel)
            grid = cv2.resize(np.vstack(panels), (960, 540 * len(panels)))
            if writer is None:
                writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*'mp4v'),
                                         10, (grid.shape[1], grid.shape[0]))
            writer.write(grid)

    if writer is not None:
        writer.release()
        print('wrote', args.video)
    pipeline.stop()
    print()
    print(os.path.basename(args.logfile), '/', os.path.basename(args.blob),
          '(RGB)' if args.rgb else '(BGR)')
    summarize(rows)


if __name__ == '__main__':
    main()
