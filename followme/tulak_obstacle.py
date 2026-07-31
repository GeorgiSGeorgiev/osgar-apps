"""
  Tulak po krasu - RedRoad following combined with 3-zone obstacle
  detection / avoidance (AprilTag following removed).

  Steering direction along the road comes from the "nn_mask" (RedRoad)
  segmentation output, exactly as in tulak.py. Obstacle handling uses
  osgar.obstdet3d_zones:ObstacleDetector3DZones (left/center/right
  distances) instead of per-cone bounding boxes, so it reacts to any
  obstacle, not just cones, and knows which side has more room.

  avoid_obstacles=False disables all of the avoidance behaviour below
  (backing up, turning, side selection, heading hold): the robot will
  still stop before hitting anything (stop_dist always applies), it
  just stops in place and does not try to get around it.

  When avoid_obstacles is True, the sequence for a blocked path is:
    1. back up a bit, to make room to turn
    2. turn - toward whichever side has more depth clearance AND more
       of the road mask still visible, so avoidance does not wander
       onto a different path around the obstacle
    3. once clear, hold the heading recorded just before step 1 for a
       moment (using the IMU-derived 'rotation' when available, since
       wheel odometry drifts through a blind, slow, high-steering-
       angle turn) so the robot comes out pointed the same way it went
       in, THEN hand control back to nn_mask road following for fine
       lateral centering back onto the path

  A short streak of confirmed close readings (close_confirm_frames) is
  required before backing up or turning kicks in, so a single noisy
  depth frame (e.g. the "flying pixel" line often seen right at the
  ground/background boundary in stereo depth) can't trigger a false
  stop by itself. A max_backup_time_sec ceiling stops the robot from
  reversing forever if the reading never clears (e.g. a persistently
  invalid window) - it just gives up waiting and turns anyway.

  --- Changes in this revision ---

  1. Bug fix: the emergency-abort branch inside on_pose2d set
     self.backing_Up (capital U, a typo) instead of self.backing_up.
     The robot still ended up backing up in that situation (via the
     fallback branch further down), but it silently lost the "keep
     the steering angled to swing clear" behaviour and always backed
     away straight instead. Fixed by using one explicit `self.state`
     (a State enum) instead of four separate booleans, so this class
     of typo can't happen again - there's no "State.BACKING_Up" to
     accidentally assign.

  2. Proportional road-following steering. on_nn_mask used to be
     bang-bang: as soon as the mask centroid crossed a small dead
     zone it snapped straight to +-turn_angle, nothing in between.
     That is almost certainly why the robot "turns too much instead
     of a minor correction" on approach to an obstacle - the
     drivable-area mask shifts as the obstacle occludes part of the
     road, and every shift past the dead zone used to be treated as a
     full turn_angle command. Steering now scales smoothly with how
     far the mask centroid is off-center, still capped at turn_angle
     at the edge of frame (so the extreme case behaves exactly as
     before). No config change needed for this.

  3. Escape mode for the "stuck in a U-shaped room" deadlock. The
     class already had (unused) escape_counter / in_escape_mode
     attributes - this wires them up. Every fresh avoidance cycle
     checks net displacement (from pose2d) since the previous cycle
     started. After escape_after_cycles cycles without at least
     escape_progress_dist_m of real progress, the robot: alternates
     turn_sign instead of recomputing it the same way each time, gets
     extra backup room (escape_backup_time_boost), and skips trying
     to realign to the original saved heading - that heading is often
     exactly what's pointed at the dead end, so fighting to return to
     it is part of what was causing the loop. This is a coarse
     heuristic based on odometry (which this file already notes
     drifts during blind turns), so treat it as "probably stuck", not
     a precise measurement, and tune escape_progress_dist_m to the
     space you actually operate in. It reduces the loop, it doesn't
     guarantee escape from a truly boxed-in space.

     Note: avoid_steering_deg defaults to 45 degrees, which is also
     Matty's default max_steering_deg - the avoidance turn already
     asks for the platform's hardest possible turn, so there is no
     "turn harder" lever available in escape mode, only "turn
     longer / back up more / try the other side / stop fighting to
     return".

  4. Ground-hazard (drop-off / staircase) reaction. Pairs with the
     new ground_hazard output from ObstacleDetector3DZones, which
     looks at the floor immediately ahead and flags when it's missing
     or unexpectedly far away (see that module's docstring for the
     important calibration caveat before enabling this in the field).
     Debouncing (ground_hazard_confirm_frames) happens here, matching
     how stop_streak/turn_streak are already handled, rather than in
     the sensing module. The reaction is a hard stop, and - by default
     (terminate_on_ground_hazard) - it raises EmergencyStopException
     and halts the run rather than attempting an automatic maneuver,
     since blindly backing up right next to a real drop-off isn't
     obviously safer. This is a deliberately conservative starting
     point for a hazard type that hasn't been field-validated yet, not
     a permanent design decision - revisit once you trust the
     detection and want it to recover on its own.

  5. GPS waypoint heading-following. A QR code (read by oak_camera_v3's
     is_qr_detection) is expected to decode to a Google-Maps-style GPS
     coordinate string (parse_gps_qr handles the plain decimal form,
     the same pair embedded in a pasted Maps URL, and DMS). Once a
     target is set, on_nmea_data (linked from the "gps" module) tracks
     bearing/distance to it on every fix.

     Deliberately NOT using this platform's IMU-derived 'rotation' (aka
     self.last_heading) for this: there is no confirmed magnetometer/
     absolute-heading source behind it (see matty.py - yaw is raw
     ESP32 IMU output, sign convention undocumented), so it may only be
     reliable as a RELATIVE heading, of no known fixed relationship to
     true north. A GPS bearing, by contrast, IS referenced to true
     north. Comparing the two directly would silently steer toward some
     rotated-by-an-unknown-amount direction, wrong in a way that's hard
     to notice without outdoor testing. Instead, current heading of
     travel is derived the same way osgar-apps/roboorienteering/ro.py
     already does it for this exact robot: by differencing consecutive
     GPS fixes (self.travel_heading), so it's compared against
     bearing_to_target using the SAME (compass) convention on both
     sides - see initial_bearing()'s docstring. The tradeoff: heading is
     unknown (falls back to pure road-following) until the robot has
     physically moved at least gps_heading_min_baseline_m since the
     last fix used as a baseline - consumer GPS without RTK is easily
     1-5m noisy, so this can't be shrunk much below the roboorienteering
     precedent (1.0m) without the derived heading being mostly noise.

     Priority is, highest first: obstacle avoidance (entirely unchanged
     above) > drivable area > GPS bearing. This is implemented by
     _drive_steering() only ever running from contexts where obstacle
     avoidance already had first say (see call sites), and internally
     blending last_dir (road) with the bearing-derived steering using
     left_road_frac/right_road_frac as a proxy for "how much road is
     actually there to steer onto" - full road-following when the mask
     shows ~no road on the side the bearing wants, scaling up to full
     bearing-following as road_frac climbs past bearing_blend_road_frac.
     follow_gps_target=False disables all of this (falls back to plain
     last_dir, i.e. tulak_obstacle's original behaviour) for testing
     without GPS/QR involved. Once within waypoint_arrival_dist_m, the
     robot stops (persistently, like ground_hazard) rather than trying
     to hold position precisely; reading a new QR sets a new target and
     resumes, and reading the literal text "abort"/"cancel" clears the
     current target without setting a new one. The target only ever
     changes via one of those three (reached / new QR / abort QR) -
     it does NOT expire or reset on its own, including while GPS has no
     fix (on_nmea_data logs "no fix yet" but leaves target_lat/lon
     untouched). on_qr_code also ignores a re-read of whatever text is
     already active (same coordinates, or "abort" when already
     aborted) - the camera redecodes and republishes a code every frame
     it's visible (oak_camera_v3.py dedups its own publishes too, but
     this is a second line of defense), so without this a code held in
     view for a couple of seconds would otherwise spam identical "New
     waypoint target" lines and repeatedly reset waypoint_reached. This
     blending is a first-pass heuristic with no field testing behind it
     at all (written at night, can't test outside) - watch
     left_road_frac/right_road_frac/bearing_to_target via verbose
     logging before trusting it to actually leave the road.

  All new config keys (escape_after_cycles, escape_progress_dist_m,
  escape_backup_time_boost, ground_hazard_confirm_frames,
  terminate_on_ground_hazard, follow_gps_target,
  gps_heading_min_baseline_m, waypoint_arrival_dist_m,
  bearing_blend_road_frac) have defaults, so existing JSON keeps working
  unchanged except for wiring the new ground_hazard/qr_code/nmea_data
  inputs (see the updated config file). stop_dist/turning_dist/
  close_confirm_frames behaviour is untouched. Bench-test before
  trusting this outdoors - it has not been run against the real osgar
  harness/hardware.
"""
import datetime
import math
import re
from enum import Enum

import numpy as np

from osgar.node import Node
from osgar.exceptions import EmergencyStopException


EARTH_RADIUS_M = 6371000


def mask_center(mask):
    if mask.max() == 0:
        return mask.shape[0] // 2, mask.shape[1] // 2
    assert mask.max() == 1, mask.max()
    indices = np.argwhere(mask == 1)  # shape (num_points, 2)
    return tuple(int(x) for x in indices.mean(axis=0))


def normalize_angle(angle):
    """wrap to (-pi, pi]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def haversine_distance(lat1, lon1, lat2, lon2):
    """great-circle distance in meters between two decimal-degree points"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def initial_bearing(lat1, lon1, lat2, lon2):
    """Compass bearing (radians, 0=north, clockwise/east-positive) from
    point 1 to point 2 along the great circle. NOT the same convention as
    this platform's rotation/pose2d heading (0=east, anticlockwise) - by
    design, an initial_bearing() result is only ever compared against
    another initial_bearing() result in this file (see _drive_steering),
    never mixed with self.last_heading, so the convention cancels out and
    never needs converting."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.atan2(y, x) % (2 * math.pi)


def parse_gps_qr(text):
    """Extract (lat, lon) in decimal degrees from a Google-Maps-style
    coordinate string - what you get long-pressing a pin in Maps and
    tapping the coordinates to copy them, e.g. "50.087451, 14.420671".
    Also matches the same pair embedded in a pasted Maps URL
    (.../@50.087451,14.420671,17z or ?q=50.087451,14.420671), and the
    DMS format Maps sometimes displays instead (50 deg 5'14.8"N
    14 deg 25'14.4"E). Returns None if no plausible coordinate pair is
    found (out-of-range values are rejected as a sanity check, e.g. to
    avoid matching an unrelated decimal pair)."""
    text = text.strip()

    m = re.search(r'(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)', text)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if abs(lat) <= 90 and abs(lon) <= 180:
            return lat, lon

    m = re.search(
        r'(\d{1,3})[°]\s*(\d{1,2})[\'′]\s*([\d.]+)["″]\s*([NS])[,\s]+'
        r'(\d{1,3})[°]\s*(\d{1,2})[\'′]\s*([\d.]+)["″]\s*([EW])',
        text)
    if m:
        d1, mi1, s1, hemi1, d2, mi2, s2, hemi2 = m.groups()
        lat = float(d1) + float(mi1) / 60 + float(s1) / 3600
        lon = float(d2) + float(mi2) / 60 + float(s2) / 3600
        if hemi1 == 'S':
            lat = -lat
        if hemi2 == 'W':
            lon = -lon
        if abs(lat) <= 90 and abs(lon) <= 180:
            return lat, lon

    return None


class State(Enum):
    DRIVE = 0
    BACKING_UP = 1
    TURNING = 2
    REALIGNING = 3


class TulakObstacle(Node):
    def __init__(self, config, bus):
        super().__init__(config, bus)
        bus.register('desired_steering')

        # driving
        self.max_speed = config.get('max_speed', 0.5)
        self.turn_angle = math.radians(config.get('turn_angle_deg', 20))

        # obstacle safety / avoidance
        self.stop_dist = config.get('stop_dist', 0.5)  # meters, hard stop - always active
        self.turning_dist = config.get('turning_dist', 1.0)  # meters, start avoidance turn
        self.close_confirm_frames = config.get('close_confirm_frames', 3)
        self.min_turn_time = datetime.timedelta(seconds=config.get('min_turn_time_sec', 3.0))
        self.max_turn_time = datetime.timedelta(seconds=config.get('max_turn_time_sec', 8.0))
        self.avoid_obstacles = config.get('avoid_obstacles', True)  # <-- the on/off switch
        self.avoid_speed = self.max_speed * config.get('avoid_speed_factor', 0.5)
        self.avoid_steering = math.radians(config.get('avoid_steering_deg', 45))
        self.backup_speed = config.get('backup_speed', 0.2)
        self.min_backup_time = datetime.timedelta(seconds=config.get('min_backup_time_sec', 1.0))
        self.max_backup_time = datetime.timedelta(seconds=config.get('max_backup_time_sec', 4.0))

        # heading hold after clearing an obstacle
        self.realign_gain = config.get('realign_gain', 1.5)
        self.realign_max_steering = math.radians(config.get('realign_max_steering_deg', 30))
        self.max_realign_time = datetime.timedelta(seconds=config.get('max_realign_time_sec', 3.0))
        self.realign_tolerance = math.radians(config.get('realign_tolerance_deg', 5))

        self.raise_exception_on_stop = config.get('terminate_on_stop', True)

        # stuck / escape-mode detection (see notes at top of file)
        self.escape_after_cycles = config.get('escape_after_cycles', 3)
        self.escape_progress_dist_m = config.get('escape_progress_dist_m', 0.4)
        self.escape_backup_time_boost = config.get('escape_backup_time_boost', 1.5)

        # ground-hazard (drop-off / staircase) reaction (see notes at top of file)
        self.ground_hazard_confirm_frames = config.get('ground_hazard_confirm_frames', 3)
        self.terminate_on_ground_hazard = config.get('terminate_on_ground_hazard', True)
        self.ground_hazard_streak = 0
        self.ground_hazard_active = False

        # GPS waypoint heading-following (see notes at top of file) - target
        # comes from a decoded QR code, current heading from differencing
        # consecutive GPS fixes (NOT from IMU 'rotation' - deliberately;
        # see docstring)
        self.follow_gps_target = config.get('follow_gps_target', True)  # <-- on/off switch
        self.gps_heading_min_baseline_m = config.get('gps_heading_min_baseline_m', 1.0)
        self.waypoint_arrival_dist_m = config.get('waypoint_arrival_dist_m', 3.0)
        self.bearing_blend_road_frac = config.get('bearing_blend_road_frac', 0.3)
        self.gps_log_interval = datetime.timedelta(seconds=config.get('gps_log_interval_sec', 2.0))
        self.target_lat = None
        self.target_lon = None
        self.target_dist = None
        self.bearing_to_target = None
        self.last_gps_pos = None  # (lat, lon) of the last fix used as a heading baseline
        self.travel_heading = None  # radians, compass convention - see initial_bearing()
        self.waypoint_reached = False
        self.last_gps_log_time = None

        # obstacle zones (from ObstacleDetector3DZones)
        self.last_obstacle = float('inf')  # center distance, meters
        self.left_dist = None
        self.right_dist = None
        self.stop_streak = 0
        self.turn_streak = 0

        # road following
        self.last_dir = 0  # steering angle (rad), from nn_mask
        self.left_road_frac = 0.5
        self.right_road_frac = 0.5

        # heading - prefer IMU 'rotation' over odometry-derived pose2d
        # heading if the platform ever sends it
        self.last_heading = 0
        self.have_imu_heading = False
        self.saved_heading = None

        # avoidance state machine - single explicit state instead of
        # several booleans, so it can't drift out of sync
        self.state = State.DRIVE
        self.state_start_time = None
        self.current_backup_steering = 0
        self.turn_sign = 1  # +1 = left, -1 = right

        # escape-mode bookkeeping
        self.escape_counter = 0
        self.in_escape_mode = False
        self.progress_anchor_xy = None

    def send_speed_cmd(self, speed, steering_angle):
        return self.bus.publish(
            'desired_steering',
            [round(speed * 1000), round(math.degrees(steering_angle) * 100)]
        )

    def on_emergency_stop(self, data):
        if data:
            self.send_speed_cmd(0, 0)
        if self.raise_exception_on_stop and data:
            raise EmergencyStopException()

    def on_bumpers_front(self, data):
        if data:
            self.send_speed_cmd(0, 0)

    def on_bumpers_rear(self, data):
        if data:
            self.send_speed_cmd(0, 0)

    def on_rotation(self, data):
        yaw_cdeg = data[0]
        self.last_heading = math.radians(yaw_cdeg / 100.0)
        self.have_imu_heading = True

    def on_qr_code(self, data):
        """data is the decoded QR text from oak.qr_code, expected to carry
        GPS coordinates (Google-Maps-style) for a new waypoint target, or
        the literal word "abort"/"cancel" to clear the current one.

        The same physical QR code gets redecoded and republished on every
        frame it's visible (no debouncing upstream, ~fps times/sec - see
        oak_camera_v3.py), so both branches below are no-ops whenever the
        decoded text doesn't actually change anything: re-reading the
        SAME coordinates must not reset waypoint_reached or re-print
        every ~100ms just because the code is still sitting in the
        camera's view, and re-reading "abort" after already-aborted must
        not do anything either."""
        text = data.strip()
        if text.lower() in ('abort', 'cancel'):
            if self.target_lat is not None:
                print(self.time, 'QR abort - clearing waypoint target', (self.target_lat, self.target_lon))
                self.target_lat = None
                self.target_lon = None
                self.bearing_to_target = None
                self.target_dist = None
                self.waypoint_reached = False
            return

        parsed = parse_gps_qr(text)
        if parsed is None:
            print(self.time, 'QR code received but no GPS coordinates found in it:', data)
            return
        if parsed == (self.target_lat, self.target_lon):
            return  # same target already active - just still in view, not a new read

        self.target_lat, self.target_lon = parsed
        self.waypoint_reached = False
        # recomputed on the next GPS fix rather than here, since we don't
        # know our own current position at QR-read time
        self.bearing_to_target = None
        self.target_dist = None
        print(self.time, 'New waypoint target from QR:', self.target_lat, self.target_lon)
        if self.last_gps_pos is not None:
            # we already have a fix from before this QR was read - report
            # bearing/distance immediately instead of waiting for the next
            # on_nmea_data (bearing_to_target above is None until then)
            self.target_dist = haversine_distance(*self.last_gps_pos, self.target_lat, self.target_lon)
            self.bearing_to_target = initial_bearing(*self.last_gps_pos, self.target_lat, self.target_lon)
            print(self.time, 'GPS', self._gps_status_line(*self.last_gps_pos))
        else:
            print(self.time, 'GPS: no fix yet, own position unknown')

    def on_nmea_data(self, data):
        lat, lon = data.get('lat'), data.get('lon')
        if lat is not None and lon is not None:
            if data.get('lat_dir') == 'S':
                lat = -lat
            if data.get('lon_dir') == 'W':
                lon = -lon

            had_heading = self.travel_heading is not None
            if self.last_gps_pos is not None:
                moved = haversine_distance(*self.last_gps_pos, lat, lon)
                if moved >= self.gps_heading_min_baseline_m:
                    # only advance the heading baseline once we've moved far
                    # enough for the bearing between fixes to be meaningful -
                    # plain GPS noise alone (no RTK) is easily 1-5m, a shorter
                    # baseline would make travel_heading mostly noise
                    self.travel_heading = initial_bearing(*self.last_gps_pos, lat, lon)
                    self.last_gps_pos = (lat, lon)
            else:
                self.last_gps_pos = (lat, lon)
            if not had_heading and self.travel_heading is not None:
                print(self.time, 'GPS heading established: %.0f deg' % math.degrees(self.travel_heading))

            if self.target_lat is not None:
                self.target_dist = haversine_distance(lat, lon, self.target_lat, self.target_lon)
                self.bearing_to_target = initial_bearing(lat, lon, self.target_lat, self.target_lon)
                if not self.waypoint_reached and self.target_dist < self.waypoint_arrival_dist_m:
                    self.waypoint_reached = True
                    print(self.time, 'waypoint reached (%.1fm), stopping' % self.target_dist)

        # throttled status log runs regardless of whether this fix was
        # valid, so a missing fix is visible ("no fix yet") instead of
        # the log just going quiet and looking like the target expired -
        # it hasn't, target_lat/target_lon are untouched by a bad fix
        if self.last_gps_log_time is None or (self.time - self.last_gps_log_time) >= self.gps_log_interval:
            self.last_gps_log_time = self.time
            if lat is not None and lon is not None:
                print(self.time, 'GPS', self._gps_status_line(lat, lon))
            elif self.target_lat is not None:
                print(self.time, 'GPS: no fix yet - target still set (%.6f,%.6f), waiting' %
                      (self.target_lat, self.target_lon))
            else:
                print(self.time, 'GPS: no fix yet, no target set')

        if self.last_gps_log_time is None or (self.time - self.last_gps_log_time) >= self.gps_log_interval:
            self.last_gps_log_time = self.time
            print(self.time, 'GPS', self._gps_status_line(lat, lon))

    def on_obstacle_zones(self, data):
        left, center, right = data
        self.last_obstacle = center
        self.left_dist = left
        self.right_dist = right

        self.stop_streak = self.stop_streak + 1 if center < self.stop_dist else 0
        # Treat None as infinitely far away so it doesn't trigger false positives
        l_dist = left if left is not None else float('inf')
        r_dist = right if right is not None else float('inf')

        # The robot is only "clear" if the center AND both sides are further than turning_dist
        is_blocked = (center < self.turning_dist) or (l_dist < self.turning_dist) or (r_dist < self.turning_dist)

        self.turn_streak = self.turn_streak + 1 if is_blocked else 0

    def on_ground_hazard(self, data):
        """data is [hazard_bool, ground_valid_frac, ground_dist] for this
        frame from ObstacleDetector3DZones - the streak/debounce logic
        lives here, same pattern as stop_streak/turn_streak above, rather
        than in the sensing module. valid_frac/dist are only for
        diagnostics/logging, not part of the trigger decision itself."""
        hazard, valid_frac, dist = data
        self.ground_hazard_streak = self.ground_hazard_streak + 1 if hazard else 0

        if self.ground_hazard_streak >= self.ground_hazard_confirm_frames:
            if not self.ground_hazard_active:
                print(self.time, 'possible drop-off/staircase detected - stopping',
                      'ground_valid_frac', valid_frac, 'ground_dist', dist)
            self.ground_hazard_active = True
            self.send_speed_cmd(0, 0)
            if self.terminate_on_ground_hazard:
                raise EmergencyStopException()
        elif self.ground_hazard_streak == 0:
            if self.ground_hazard_active:
                print(self.time, 'ground hazard cleared, resuming')
            self.ground_hazard_active = False

    def on_nn_mask(self, data):
        mask = data.copy()  # never modify the shared buffer in place
        height, width = mask.shape
        mask[:height // 2, :] = 0  # ignore sky/horizon in the top half

        center_y, center_x = mask_center(mask)
        half = width / 2
        dead = (width // 16) / half  # same dead-zone width as before, as a fraction of half-width

        offset = (center_x - half) / half  # -1 (mask hugging left edge) .. +1 (right edge)
        if abs(offset) <= dead:
            self.last_dir = 0.0
        else:
            # Proportional beyond the dead zone: a small deviation gets a
            # small nudge, and only a deviation all the way to the frame
            # edge gets the full turn_angle. Previously this snapped
            # straight to turn_angle the instant the dead zone was
            # crossed - that abruptness is what produced the "turns too
            # much instead of a minor correction" behaviour on approach
            # to an obstacle, as the mask shifts.
            scaled = min(1.0, (abs(offset) - dead) / (1 - dead))
            self.last_dir = -math.copysign(scaled * self.turn_angle, offset)

        # how much of each side still looks like road - used only to
        # help pick which way to go around an obstacle, so avoidance
        # doesn't wander onto a different path
        half_px = width // 2
        self.left_road_frac = float(mask[:, :half_px].mean())
        self.right_road_frac = float(mask[:, half_px:].mean())

    def _choose_turn_sign(self):
        """+1 = turn left, -1 = turn right. Prefer the side that is both
        physically clearer and still shows road."""
        left_clear = self.left_dist is None or self.left_dist > self.turning_dist
        right_clear = self.right_dist is None or self.right_dist > self.turning_dist

        if left_clear and not right_clear:
            return 1
        if right_clear and not left_clear:
            return -1
        # both clear, or both blocked by depth alone - break the tie
        # using whichever side still looks more like road
        if abs(self.left_road_frac - self.right_road_frac) > 0.02:
            return 1 if self.left_road_frac > self.right_road_frac else -1
        # still tied - fall back to raw clearance, default left
        left = self.left_dist if self.left_dist is not None else float('inf')
        right = self.right_dist if self.right_dist is not None else float('inf')
        return 1 if left >= right else -1

    def _drive_steering(self):
        """Steering to use when NOT actively avoiding an obstacle - i.e.
        this only ever runs from a context where obstacle avoidance has
        already had first say (it is highest priority: it fully overrides
        this method's result by never calling it while avoiding). Within
        that, drivable-area (last_dir, from the road mask) still wins over
        the GPS bearing whenever the mask shows little/no road on the side
        the bearing wants: bearing_blend_road_frac is the road-fraction
        (see on_nn_mask - the theoretical max is ~0.5 since the always-
        masked-out sky half counts toward the mean) at which the bearing
        gets full trust; below that it's scaled down proportionally, pure
        road-following at 0. This is a first-pass heuristic, not field
        tuned - watch left_road_frac/right_road_frac against turn_streak
        false positives once you can test outside."""
        if not self.follow_gps_target or self.bearing_to_target is None or self.travel_heading is None:
            return self.last_dir  # no usable GPS heading yet - pure road following

        error = normalize_angle(self.travel_heading - self.bearing_to_target)
        bearing_steering = max(-self.turn_angle, min(self.turn_angle, error))

        road_frac_that_way = self.left_road_frac if bearing_steering > 0 else self.right_road_frac
        if self.bearing_blend_road_frac > 0:
            weight = min(1.0, road_frac_that_way / self.bearing_blend_road_frac)
        else:
            weight = 1.0
        return (1 - weight) * self.last_dir + weight * bearing_steering

    def _following_status(self):
        """Human-readable reason why GPS bearing-following is or isn't
        currently steering the robot - logging only, mirrors the guard
        clauses in _drive_steering()."""
        if self.target_lat is None:
            return 'no target'
        if not self.follow_gps_target:
            return 'disabled (follow_gps_target=False)'
        if self.waypoint_reached:
            return 'arrived, stopped'
        if self.travel_heading is None:
            return 'target set, waiting for GPS heading (need >=%.1fm of movement)' % self.gps_heading_min_baseline_m
        return 'following'

    def _gps_status_line(self, lat, lon):
        """One-line summary of everything on_qr_code/on_nmea_data know
        right now: own position, direction of travel, target and bearing/
        distance to it, what steering that would currently produce, and
        whether it's actually being applied (see _following_status)."""
        parts = ['pos=(%.6f,%.6f)' % (lat, lon)]
        if self.travel_heading is not None:
            parts.append('heading=%.0fdeg' % math.degrees(self.travel_heading))
        else:
            parts.append('heading=unknown')
        if self.target_lat is not None:
            parts.append('target=(%.6f,%.6f)' % (self.target_lat, self.target_lon))
            parts.append('bearing=%.0fdeg dist=%.1fm' % (math.degrees(self.bearing_to_target), self.target_dist))
        else:
            parts.append('target=none')
        parts.append('steering_now=%.0fdeg' % math.degrees(self._drive_steering()))
        parts.append('(%s)' % self._following_status())
        return ' '.join(parts)

    def _enter_backing_up(self, steering):
        self.state = State.BACKING_UP
        self.state_start_time = self.time
        self.current_backup_steering = steering

    def _enter_turning(self):
        self.state = State.TURNING
        self.state_start_time = self.time
        if self.in_escape_mode:
            self.turn_sign = -self.turn_sign  # deliberately try the other side
        else:
            self.turn_sign = self._choose_turn_sign()

    def _enter_drive(self):
        self.state = State.DRIVE
        self.saved_heading = None

    def _start_avoidance_cycle(self, xy):
        """Called once per fresh avoidance cycle (not on the re-entrant/
        aborted-mid-maneuver path). Tracks whether the robot is making
        real progress between cycles; after escape_after_cycles cycles
        with less than escape_progress_dist_m of net movement, switches
        into escape mode."""
        if self.saved_heading is not None:
            return  # already mid-cycle
        self.saved_heading = self.last_heading
        if self.progress_anchor_xy is not None:
            dist = math.hypot(xy[0] - self.progress_anchor_xy[0],
                              xy[1] - self.progress_anchor_xy[1])
            self.escape_counter = self.escape_counter + 1 if dist < self.escape_progress_dist_m else 0
        self.progress_anchor_xy = xy
        was_escaping = self.in_escape_mode
        self.in_escape_mode = self.escape_counter >= self.escape_after_cycles
        if self.in_escape_mode and not was_escaping:
            print(self.time, 'little net progress over', self.escape_counter,
                  'avoidance cycles - entering escape mode')

    def on_pose2d(self, data):
        x_mm, y_mm, heading_cdeg = data
        xy = (x_mm / 1000.0, y_mm / 1000.0)
        if not self.have_imu_heading:
            self.last_heading = math.radians(heading_cdeg / 100.0)

        if self.ground_hazard_active:
            # confirmed drop-off/staircase - stay stopped every cycle,
            # don't let the normal state machine drive through it
            self.send_speed_cmd(0, 0)
            return

        # Helper flags for clearance
        left_clear = self.left_dist is None or self.left_dist > self.turning_dist
        right_clear = self.right_dist is None or self.right_dist > self.turning_dist
        any_side_clear = left_clear or right_clear

        # CENTRALIZED SAFETY CHECK: If an emergency stop occurs during an active maneuver,
        # instantly abort the current state and switch to backing up.
        is_emergency = self.avoid_obstacles and self.stop_streak >= self.close_confirm_frames
        if is_emergency and self.state in (State.TURNING, State.REALIGNING):
            print(self.time, 'Emergency stop during maneuver! Aborting to backup.')
            # If we were turning, keep the steering angled to swing away; otherwise back straight
            steer = self.avoid_steering if self.state == State.TURNING else 0
            self._enter_backing_up(steer)

        # --- STATE MACHINE ---
        if self.avoid_obstacles and self.state == State.BACKING_UP:
            speed, steering_angle = -self.backup_speed, self.current_backup_steering
            elapsed = self.time - self.state_start_time
            boost = self.escape_backup_time_boost if self.in_escape_mode else 1.0
            min_bt, max_bt = self.min_backup_time * boost, self.max_backup_time * boost

            if (elapsed > min_bt and any_side_clear) or elapsed > max_bt:
                if elapsed > max_bt and not any_side_clear:
                    print(self.time, 'max backup time reached, turning despite all zones blocked')
                else:
                    print(self.time, 'done backing up, open zone detected, start turning')
                self._enter_turning()

        elif self.avoid_obstacles and self.state == State.TURNING:
            speed, steering_angle = self.avoid_speed, self.turn_sign * self.avoid_steering
            elapsed = self.time - self.state_start_time
            if (elapsed > self.min_turn_time and self.turn_streak == 0) or elapsed > self.max_turn_time:
                if elapsed > self.max_turn_time and self.turn_streak != 0:
                    print(self.time, 'giving up waiting to clear, realigning anyway')
                else:
                    print(self.time, 'stop turning, realigning to', round(math.degrees(self.saved_heading)))
                if self.in_escape_mode:
                    # saved_heading is often exactly what's pointed at the
                    # dead end - accept the new heading instead of
                    # fighting to return to it
                    print(self.time, 'escape mode: accepting new heading instead of realigning')
                    self._enter_drive()
                    speed, steering_angle = self.max_speed, self._drive_steering()
                else:
                    self.state = State.REALIGNING
                    self.state_start_time = self.time

        elif self.avoid_obstacles and self.state == State.REALIGNING:
            error = normalize_angle(self.saved_heading - self.last_heading)
            elapsed = self.time - self.state_start_time
            if abs(error) < self.realign_tolerance or elapsed > self.max_realign_time:
                print(self.time, 'realigned, resuming road following')
                self._enter_drive()
                speed, steering_angle = self.max_speed, self._drive_steering()
            else:
                steering_angle = max(-self.realign_max_steering,
                                     min(self.realign_max_steering, error * self.realign_gain))
                speed = self.max_speed

        elif is_emergency or (not self.avoid_obstacles and self.stop_streak >= self.close_confirm_frames):
            if self.avoid_obstacles:
                print(self.time, 'obstacle too close, backing up', self.last_obstacle)
                self._start_avoidance_cycle(xy)
                self._enter_backing_up(0)
                speed, steering_angle = -self.backup_speed, 0
            else:
                speed, steering_angle = 0, 0

        elif self.avoid_obstacles and self.turn_streak >= self.close_confirm_frames:
            self._start_avoidance_cycle(xy)
            if not any_side_clear:
                print(self.time, 'all zones blocked, backing up straight to find room', self.last_obstacle)
                self._enter_backing_up(0)
                speed, steering_angle = -self.backup_speed, 0
            else:
                print(self.time, 'obstacle nearby, start turning', self.last_obstacle)
                self._enter_turning()
                speed, steering_angle = self.avoid_speed, self.turn_sign * self.avoid_steering

        elif self.waypoint_reached:
            speed, steering_angle = 0, 0

        else:
            speed, steering_angle = self.max_speed, self._drive_steering()

        if self.verbose:
            print(self.time, self.state, speed, steering_angle, self.last_obstacle,
                  self.left_dist, self.right_dist, 'ESCAPE' if self.in_escape_mode else '')
        self.send_speed_cmd(speed, steering_angle)
# vim: expandtab sw=4 ts=4
