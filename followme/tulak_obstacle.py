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

  All new config keys (escape_after_cycles, escape_progress_dist_m,
  escape_backup_time_boost, ground_hazard_confirm_frames,
  terminate_on_ground_hazard) have defaults, so existing JSON keeps
  working unchanged except for wiring the new ground_hazard input (see
  the updated config file). stop_dist/turning_dist/close_confirm_frames
  behaviour is untouched. Bench-test before trusting this outdoors -
  it has not been run against the real osgar harness/hardware.
"""
import datetime
import math
from enum import Enum

import numpy as np

from osgar.node import Node
from osgar.exceptions import EmergencyStopException


def mask_center(mask):
    if mask.max() == 0:
        return mask.shape[0] // 2, mask.shape[1] // 2
    assert mask.max() == 1, mask.max()
    indices = np.argwhere(mask == 1)  # shape (num_points, 2)
    return tuple(int(x) for x in indices.mean(axis=0))


def normalize_angle(angle):
    """wrap to (-pi, pi]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


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
        """data is the decoded QR text from oak.qr_code. Logging only for
        now - the QR is intended to carry waypoint coordinates, but acting
        on them (driving to the waypoint) is a follow-up task, not wired
        up yet."""
        print(self.time, 'QR code received:', data)

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
                    speed, steering_angle = self.max_speed, self.last_dir
                else:
                    self.state = State.REALIGNING
                    self.state_start_time = self.time

        elif self.avoid_obstacles and self.state == State.REALIGNING:
            error = normalize_angle(self.saved_heading - self.last_heading)
            elapsed = self.time - self.state_start_time
            if abs(error) < self.realign_tolerance or elapsed > self.max_realign_time:
                print(self.time, 'realigned, resuming road following')
                self._enter_drive()
                speed, steering_angle = self.max_speed, self.last_dir
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

        else:
            speed, steering_angle = self.max_speed, self.last_dir

        if self.verbose:
            print(self.time, self.state, speed, steering_angle, self.last_obstacle,
                  self.left_dist, self.right_dist, 'ESCAPE' if self.in_escape_mode else '')
        self.send_speed_cmd(speed, steering_angle)
# vim: expandtab sw=4 ts=4
