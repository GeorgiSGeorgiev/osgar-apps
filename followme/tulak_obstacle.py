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

  6. Retrace-based backing up + scan-based turning commit. Two changes
     to the avoidance maneuver itself, both prompted by field footage
     (tall grass: center zone pinned at the 0.0 fail-safe from too few
     valid depth pixels - see obstdet3d_zones.py - sides reading
     None/"clear" from lack of data rather than confirmed openness)
     that showed the previous version committing to a direction after
     a single momentarily-clear frame, then driving straight back into
     the same patch it had just backed away from.

     Backing up now retraces the path just driven, instead of backing
     up straight/at a fixed angle: on_pose2d logs (steering_angle,
     duration) for every forward-driving cycle into a bounded
     self.path_history buffer (retrace_buffer_sec worth of it), and a
     fresh (non-emergency-abort) entry into BACKING_UP snapshots that
     history reversed into self.retrace_queue, replaying it - same
     steering sign, reverse chronological order, negated speed - via
     _next_retrace_steering(). Physically this retraces the same curve
     the robot just drove: for a fixed front-steering vehicle, holding
     the same steering angle while reversing traces the same arc
     backward (the everyday "back out along your own tracks" maneuver -
     curvature is dtheta/ds = tan(steering)/wheelbase, a property of the
     steering angle alone, independent of which way the wheels are
     turning). This is the highest-confidence direction available: it's
     ground the robot has already just proven it can cross. Once the
     buffer runs out it falls back to backing up straight, same as
     before. The emergency-abort path (something dangerously close
     RIGHT NOW - see the centralized safety check in on_pose2d) keeps
     the old fixed-angle behaviour on purpose - there's no time/margin
     to be clever about replaying history there, and mid-avoidance the
     recent history may not even be "the way in" anymore.

     Turning now scans before committing to a heading, instead of
     exiting the instant turn_streak flickers back to 0 - which could
     be a single non-blocked frame, since there's confirm-frame
     debounce going INTO "blocked" (close_confirm_frames) but none
     coming back out of it. While TURNING, every cycle appends
     (heading, center, left, right) to self.scan_samples; the state now
     requires sweeping at least scan_min_sweep_deg of real heading
     change (from IMU-derived last_heading, not wheel odometry - it
     shouldn't be fooled by the same wheel-slip-in-grass that can
     confuse the escape-mode progress check) before it's allowed to
     stop turning, and on exit picks the best-scoring recorded heading
     instead of "wherever we happened to be pointed when a flicker
     cleared." Score is center distance plus side_zone_weight times
     whichever of left/right is smaller (both zones are field-
     calibrated now, worth actually weighing in rather than only using
     as a clear/unclear tiebreak); a small bearing_penalty_weight
     additionally favors headings closer to bearing_to_target when a
     GPS target is set - converted into the same compass convention as
     bearing_to_target via a heading_frame_offset captured once per
     avoidance cycle first (last_heading has no known relationship to
     true north either, same caveat as section 5's _drive_steering -
     comparing it to bearing_to_target directly would silently steer
     toward some rotated-by-an-unknown-amount heading) - but only
     enough to break ties between comparable gaps, not to override a
     clearer one - obstacle
     avoidance still fully overrides GPS per the priority order in
     section 5 above, this only nudges which of several acceptable
     gaps gets picked.

     This also lets escape mode drop its old special case ("accept
     whatever heading we're at instead of realigning, because
     saved_heading is probably the dead end we're trying to leave").
     saved_heading is repurposed here as REALIGNING's target heading in
     general, not just "the original pre-avoidance heading" - TURNING
     overwrites it with the scanned best_heading before handing off, in
     or out of escape mode. Since that target now comes from evidence
     gathered during the sweep, it isn't the dead end by construction,
     so there's nothing left to avoid fighting to return to.

  7. Bug fix: stop_streak (drives is_emergency, the hard-stop safety net
     that's supposed to abort TURNING/REALIGNING and start backing up
     regardless of what state-specific logic is doing) was computed from
     `center` alone, even though stop_dist is documented above as an
     always-applies hard stop. Caught from field footage: TURNING drives
     forward for its whole duration by design, and can clip a wall on
     either side - not dead ahead - well before center ever reads close;
     with no side check, that never registered as an emergency no matter
     how close it got. Now stop_now (feeding stop_streak) checks center
     OR either side against stop_dist, same None-is-far-not-clear
     handling as the existing turning_dist check just below it. This
     also caps how long the new scan-based TURNING sweep (item 6) can
     spend driving forward into a tightening corner before something
     stops it, regardless of scan_min_sweep_deg.

  8. Confident early exit from the TURNING sweep (item 6). Field-tested:
     forcing the full scan_min_sweep_deg every time made ordinary,
     unambiguous avoidance (a single obstacle with clearly open road on
     one side) feel like it "held the turn too long" - it kept sweeping
     to compare options even when the very first direction tried was
     already obviously fine. The full sweep is what actually helps in
     the ambiguous case (tall grass, nothing reads clearly open) - it
     was never needed for the easy case. Now each TURNING cycle also
     checks whether the CURRENT heading alone is confidently clear
     (center AND both sides at least scan_confident_margin times
     turning_dist - comfortably past the threshold, not barely over it),
     debounced by close_confirm_frames so one lucky frame still can't
     trigger it (the exact failure mode item 6 fixed in the first
     place). If so, it commits right away instead of grinding through
     the rest of the sweep; otherwise scan_min_sweep_deg still applies
     as before.

  9. Bug fix: GPS heading (travel_heading) was mostly noise at the
     original gps_heading_min_baseline_m=1.0, field-confirmed by an
     S-curve in open ground with nothing else in play (obstacle
     avoidance untouched by this item - turn_streak/stop_streak were 0
     the whole time). Two consecutive GPS fixes ~1.45m apart (just past
     the old 1.0m baseline) produced travel_heading readings 219 degrees
     apart in 3 seconds while driving a straight line - physically
     impossible, i.e. that baseline distance is not long enough for
     ordinary consumer-GPS fix-to-fix jitter (the module docstring
     already warned "easily 1-5m") to average out against a real bearing.
     bearing_to_target itself is unaffected (it's computed against a
     target far away, so a couple meters of fix noise barely moves it) -
     only travel_heading, computed over the much shorter distance between
     successive fixes, was the problem. Two independent mitigations, both
     scoped to GPS code only: gps_heading_min_baseline_m's default is now
     5.0 (still just a more conservative starting point, not calibrated
     for your receiver - re-tune from real fix-to-fix scatter if you have
     it), and travel_heading is now smoothed across updates with a
     circular EMA (gps_heading_smoothing_alpha, see _smooth_heading) -
     same idea as the pitch smoothing in obstdet3d_zones.py, just over
     wrap-around angles this time, so a plain linear EMA would be wrong.

  10. Adaptive stop_dist/turning_dist/cruising speed - so ONE config can
      cover both a cramped indoor room and open outdoor ground, instead
      of needing separate hand-tuned profiles. Two closed-loop pieces,
      both in on_pose2d, both defaulting on (adaptive_distances,
      adaptive_speed):

      _adaptive_speed() scales cruising speed down as sensed clearance
      (min of center/left/right, None-as-inf same as elsewhere) shrinks,
      linearly between min_speed and max_speed, saturating at full
      max_speed once clearance reaches speed_clearance_ceiling. This
      alone is what makes cramped spaces "just work" without a separate
      profile: the robot naturally crawls where it's tight and speeds up
      where it's open, driven by the same sensor already used for
      avoidance.

      _update_adaptive_distances() then derives stop_dist/turning_dist
      from a plain stopping-distance model - reaction distance
      (v * (close_confirm_frames/depth_fps + reaction_margin_sec), the
      debounce dead time already baked into the state machine, now made
      explicit as a distance) plus braking distance (v^2/(2*decel_mps2))
      plus a fixed margin, clamped to [stop_dist_min, stop_dist_max].
      turning_dist adds a further v*turning_lead_sec on top, clamped to
      [turning_dist_min, turning_dist_max] - the gap between the two
      widens with speed, same as the reaction distance does, instead of
      staying a fixed 0.1-0.2m regardless of how fast the robot is
      going. Crucially this reads _last_commanded_speed - whatever was
      ACTUALLY sent last cycle, not self.max_speed - so stop_dist
      correctly shrinks during BACKING_UP/TURNING (slower than cruising)
      instead of staying pinned to cruising-speed math while already
      moving cautiously.

      decel_mps2 is an ESTIMATE (chosen so the formula reproduces the
      hand-tuned stop_dist~0.6m already validated at max_speed=0.5 in
      the field - not independent bench data). To calibrate for real:
      drive at a known speed, command a stop, measure the distance, back
      out a = v^2/(2*d). The min/max bounds on both distances and speed
      are the actual safety net if the formula is ever wrong for your
      platform/terrain - set adaptive_distances/adaptive_speed to False
      to fall back to the old flat stop_dist/turning_dist/max_speed
      behaviour outright.

      One real limitation this does NOT solve: a truly tiny room smaller
      than Matty's own turning/backup footprint can't be fixed by
      slowing down - that's a fixed geometric constraint independent of
      speed. Adaptive distances only handle the speed axis of "cramped
      vs open", not "does the maneuver physically fit here at all".

  11. Free-space steering (depth_profile) + smoothest-path blend +
      steering rate limit. Together these are meant to keep the robot
      away from a tightening side well before turning_dist would trigger
      the discrete avoidance state machine, so an ordinary narrow
      driveway doesn't need the big backup+turn maneuver just to stay
      roughly centered - the state machine still owns "something forced
      an unavoidable maneuver", this only handles "which way should
      ordinary cruising lean".

      _free_space_steering() reads the new depth_profile (a coarse N-bin
      distance scan across the frame, from ObstacleDetector3DZones - see
      that module's docstring) and aims toward whichever bins have the
      most clearance beyond a threshold derived from the CURRENT (and,
      with item 10 on, currently speed-scaled) turning_dist times
      free_space_lead_margin - so this signal's "is that open enough"
      question always tracks whatever turning_dist currently means,
      rather than drifting out of sync with it. _drive_steering() blends
      this with last_dir (the road mask) via free_space_weight - this
      pairing is the answer to "pick the smoothest path that's also
      marked drivable": smoothness/openness from depth, drivable-marking
      from the road mask, combined rather than either alone deciding.
      Deliberately NOT part of the State enum/state machine - this is a
      continuous control question, not a discrete one, and forcing it
      into the state machine would only complicate the one thing that
      actually needs to stay simple and decisive (the avoidance
      maneuvers themselves).

      Finally, _rate_limit_steering() caps how fast _drive_steering()'s
      blended result can change (max_steering_rate_deg_s, continuous
      across a DRIVE<->avoidance transition via _last_commanded_steering)
      - applied ONLY to ordinary cruising, never to the avoidance state
      machine's own steering, which still needs to commit decisively
      the instant it's triggered.

  12. enable_ground_hazard: off/on switch for the drop-off/staircase
      reaction (item 4) - ObstacleDetector3DZones keeps computing and
      publishing ground_hazard either way (unchanged, still useful for
      logging/diagnostics), this just makes on_ground_hazard a no-op
      when False, e.g. for terrain where the ground window's calibration
      trap (see obstdet3d_zones.py's docstring) isn't worth fighting yet.

  13. Two field-reported issues, both fixed WITHOUT touching stop_dist/
      turning_dist or the adaptive-distance formula at all (item 10) -
      only how the avoidance maneuver behaves once already triggered by
      those unchanged thresholds:

      (a) "Overcorrects 60-90deg instead of a smaller correction" on a
      diagonal wall approach. Previously EVERY turn_streak trigger got
      the identical response: full avoid_steering, full scan_min_sweep -
      appropriate for a squarely-blocked center, way too much for a
      shallow graze where only one side zone barely crossed turning_dist
      while center/the other side are still comfortably clear. Worse,
      for a genuinely diagonal wall, clearance keeps rising the further
      you turn away from it, so the old fixed 30deg-minimum sweep would
      often end up picking a near-the-end-of-the-sweep heading anyway -
      and if that single pass didn't fully clear it, a SECOND full-lock
      cycle could stack on top, which is what actually produced the
      60-90deg the field report described (two ~30-45deg corrections
      back to back reads as one big one). _enter_turning() now computes
      a severity 0..1 from how far the closest zone already is past
      turning_dist toward stop_dist (0 = just barely triggered, 1 =
      already down near stop_dist) and scales this cycle's turn amplitude
      and required sweep between new min_avoid_steering_deg/
      min_scan_sweep_deg floors and the existing avoid_steering_deg/
      scan_min_sweep_deg ceilings - reached at severity=1, i.e. IDENTICAL
      to the old always-maximal behaviour for a real, close block.
      Escape mode (see below) always forces severity=1 too - full
      commitment there is unchanged from before this item.

      (b) Dead-end oscillation: forward-left, blocked, back, forward-left
      again, back, forward-left again... Root cause: _best_scan_heading()
      is a pure greedy optimizer with no memory - every fresh cycle it
      just picks whatever scored best THIS sweep, with nothing stopping
      it from re-picking essentially the same heading that already
      failed one or more cycles ago, if it still happens to score best
      among the (possibly all mediocre) options actually found. Fixed
      with failed_headings (a short deque): _start_avoidance_cycle
      already detects "no real progress since the last cycle" for escape
      mode (escape_after_cycles/escape_progress_dist_m, unchanged) - now
      that same no-progress detection also records last_committed_heading
      (the heading that specific cycle aimed for and that didn't pan
      out) into failed_headings, cleared the moment real progress IS
      detected again. _best_scan_heading()'s scoring then penalizes
      candidates close to anything in failed_headings, tapering to zero
      at failed_heading_tolerance_deg - so a repeatedly-failing spot
      progressively pushes the greedy choice toward something ACTUALLY
      different, including a locally worse-scoring option, rather than
      reconverging on the same trap - this is what the field report
      asked for ("sometimes choose the suboptimal route... if it means
      not getting stuck"), implemented as a bias rather than a random
      choice so it stays deterministic/debuggable. The penalty is purely
      subtractive and only ever reorders candidates the sweep already
      found safe under the UNCHANGED stop_dist/turning_dist/is_emergency
      checks - it cannot invent a new option or bypass a safety check,
      so this cannot introduce a crash risk on its own.

      Both fixes are zero-effect in the common case: with an empty
      failed_headings and severity=1 (a real close block), everything
      behaves EXACTLY as before this item - they only activate for the
      specific shallow-encounter / repeated-failure situations they're
      meant to fix, to minimize risk to already-field-validated behaviour.

  14. Bug fix: on_bumpers_front/on_bumpers_rear only ever zeroed the
      speed command for a single instant - nothing changed self.state,
      so on_pose2d's state machine would just recompute from whatever
      state it was already in on the very next cycle and resume the
      SAME motion. In the open, this rarely mattered (bumper contact
      should be rare if the depth-based avoidance is doing its job). In
      a tightly boxed-in space it's a real gap: the depth camera has no
      coverage at all behind/beside the robot while BACKING_UP (only the
      rear bumper does), so a bumper hit there is exactly the situation
      needing a real reaction, and there wasn't one - field-reported as
      "sometimes even crashing when reversing" in a very cramped spot.
      Deliberately NOT fixed by touching stop_dist/turning_dist (not
      possible anyway - there's no sensor coverage back there to derive
      a distance from) - fixed by making contact itself trigger a real
      state change: a rear hit pivots to TURNING using current, live
      forward sensor data (severity computed fresh, so if backing up
      already gained a bit of room, the response reflects that); a front
      hit backs off straight, same fixed/non-clever reaction as the
      existing is_emergency abort. Past max_bumper_hits repeated hits
      with no real progress since (bumper_hit_streak, reset at the same
      point as escape_counter/failed_headings), gives up and raises
      EmergencyStopException rather than continuing to retry - a spot
      that keeps producing contact regardless of what's tried needs a
      human, not another automatic attempt. Unlike ground_hazard, this
      has no non-terminating mode: recovery there is a sensor reading
      becoming valid again on its own; a bumper is a one-shot contact
      event with no equivalent "it got better" signal to wait for while
      sitting still, so there's nothing sensible for a non-terminating
      mode to do besides freeze forever.

      Also removed dead/duplicate code in on_nmea_data: two copies of
      the same throttled status-log block ran back to back. The second
      was unreachable in practice (the first always just reset
      last_gps_log_time to now, so the second's own time-since-last-log
      check could never pass) - but had it ever run with no GPS fix
      (lat/lon None), it would have crashed formatting None into
      _gps_status_line's '%.6f'. Removed, no behaviour change (it never
      executed).

  All new config keys (escape_after_cycles, escape_progress_dist_m,
  escape_backup_time_boost, ground_hazard_confirm_frames,
  terminate_on_ground_hazard, follow_gps_target,
  gps_heading_min_baseline_m, waypoint_arrival_dist_m,
  bearing_blend_road_frac, retrace_buffer_sec, scan_min_sweep_deg,
  scan_confident_margin, side_zone_weight, bearing_penalty_weight,
  gps_heading_smoothing_alpha, adaptive_distances, decel_mps2,
  reaction_margin_sec, depth_fps, stop_dist_margin_m, turning_lead_sec,
  stop_dist_min, stop_dist_max, turning_dist_min, turning_dist_max,
  adaptive_speed, min_speed, speed_clearance_ceiling, free_space_weight,
  free_space_lead_margin, max_steering_rate_deg_s, enable_ground_hazard,
  min_avoid_steering_deg, min_scan_sweep_deg, failed_heading_penalty_weight,
  failed_heading_tolerance_deg, failed_headings_maxlen, max_bumper_hits)
  have defaults, so existing JSON keeps working unchanged except for
  wiring the new ground_hazard/qr_code/nmea_data/depth_profile inputs
  (see the updated config file). close_confirm_frames behaviour is
  untouched; stop_dist/turning_dist are now adaptive BY DEFAULT (item
  10) - set adaptive_distances=False to keep them flat like before.
  Bench-test before trusting this outdoors - it has not been run against the real
  osgar harness/hardware.
"""
import datetime
import math
import re
from collections import deque
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
        # stop_dist/turning_dist below are the STATIC starting values -
        # if adaptive_distances is on (default, see further down and item
        # 10 at top of file), on_pose2d recomputes both every cycle from
        # the robot's actual current speed instead, and these are only
        # what's used before the first cycle / whenever that's off.
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

        # adaptive stop_dist/turning_dist (item 10 at top of file) -
        # derived every cycle from the robot's actual last-commanded
        # speed via a simple stopping-distance model (reaction distance +
        # braking distance + margin), instead of flat numbers that
        # silently stop making sense the moment you run at a different
        # speed than whatever they were hand-tuned for. decel_mps2 is an
        # ESTIMATE, not bench-measured - see item 10 for how to calibrate
        # it for real (drive at a known speed, command a stop, measure
        # the distance, back out a = v^2/(2*d)).
        self.adaptive_distances = config.get('adaptive_distances', True)
        self.decel_mps2 = config.get('decel_mps2', 1.0)
        self.reaction_margin_sec = config.get('reaction_margin_sec', 0.2)
        self.depth_fps = config.get('depth_fps', 10)  # must match the oak module's own fps - not auto-linked, separate config block
        self.stop_dist_margin_m = config.get('stop_dist_margin_m', 0.15)
        self.turning_lead_sec = config.get('turning_lead_sec', 0.4)
        self.stop_dist_min = config.get('stop_dist_min', 0.3)
        self.stop_dist_max = config.get('stop_dist_max', 2.0)
        self.turning_dist_min = config.get('turning_dist_min', 0.4)
        self.turning_dist_max = config.get('turning_dist_max', 3.0)

        # adaptive cruising speed (item 10) - scales speed down as sensed
        # clearance shrinks, within [min_speed, max_speed]. Feeds the
        # distance formula above via _last_commanded_speed, closing the
        # loop: tight space -> slower speed -> smaller stop/turning
        # distance, open space -> faster -> larger distance - one
        # mechanism instead of separate "cramped" vs "outdoor" profiles.
        self.adaptive_speed = config.get('adaptive_speed', True)
        self.min_speed = config.get('min_speed', 0.15)
        self.speed_clearance_ceiling = config.get('speed_clearance_ceiling', 3.0)
        self._last_commanded_speed = 0.0
        self._last_commanded_steering = 0.0

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
        self.enable_ground_hazard = config.get('enable_ground_hazard', True)  # <-- on/off switch
        self.ground_hazard_confirm_frames = config.get('ground_hazard_confirm_frames', 3)
        self.terminate_on_ground_hazard = config.get('terminate_on_ground_hazard', True)
        self.ground_hazard_streak = 0
        self.ground_hazard_active = False

        # GPS waypoint heading-following (see notes at top of file) - target
        # comes from a decoded QR code, current heading from differencing
        # consecutive GPS fixes (NOT from IMU 'rotation' - deliberately;
        # see docstring)
        self.follow_gps_target = config.get('follow_gps_target', True)  # <-- on/off switch
        # field-tested (see item 9 at top of file): 1.0m let ordinary GPS
        # jitter alone cross the baseline and get read as real motion,
        # producing a travel_heading that swung 200+ degrees in a few
        # seconds while driving straight. 5.0m is a more conservative
        # starting point, not a calibrated value for your receiver either.
        self.gps_heading_min_baseline_m = config.get('gps_heading_min_baseline_m', 5.0)
        # circular EMA (heading wraps at 2pi, a plain linear EMA would be
        # wrong near the wrap) applied to travel_heading across successive
        # baseline-gated updates - a second layer of noise rejection on
        # top of the baseline distance itself, same spirit as the pitch
        # smoothing in obstdet3d_zones.py. 1.0 = no smoothing.
        self.gps_heading_smoothing_alpha = config.get('gps_heading_smoothing_alpha', 0.4)
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
        self.depth_profile = []  # free_space_bins distances - see on_depth_profile/_free_space_steering
        # the OAK pipeline takes several seconds to boot, during which
        # last_obstacle/left_dist/right_dist above still hold their
        # "assume clear" init values - stay stopped in on_pose2d until the
        # first real obstacle_zones frame arrives instead of driving on
        # that default (see on_pose2d and on_obstacle_zones)
        self.have_obstacle_data = False

        # road following
        self.last_dir = 0  # steering angle (rad), from nn_mask
        self.left_road_frac = 0.5
        self.right_road_frac = 0.5

        # free-space steering (item 11 at top of file) - continuous
        # depth-based centering nudge, blended with last_dir above,
        # meant to keep the robot away from a tightening side well
        # before turning_dist would trigger the discrete avoidance state
        # machine, so a narrow driveway doesn't need the big backup+turn
        # maneuver just to stay roughly centered.
        self.free_space_weight = config.get('free_space_weight', 0.5)
        self.free_space_lead_margin = config.get('free_space_lead_margin', 1.3)
        rate_deg_s = config.get('max_steering_rate_deg_s', 60)
        self.max_steering_rate = math.radians(rate_deg_s) if rate_deg_s and rate_deg_s > 0 else None

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

        # proportional avoidance-turn severity (item 13 at top of file) -
        # does NOT touch stop_dist/turning_dist (when avoidance triggers
        # is unchanged) - only how hard it turns once triggered. Full
        # avoid_steering/scan_min_sweep are still the ceiling, reached at
        # severity=1 - unchanged from before for a close/severe encounter
        # and, deliberately, always for escape mode (see _enter_turning).
        self.min_avoid_steering = math.radians(config.get('min_avoid_steering_deg', 20))
        self.min_scan_sweep = math.radians(config.get('min_scan_sweep_deg', 10))
        self.current_avoid_steering = self.avoid_steering  # this cycle's actual turn amplitude, set in _enter_turning
        # current_scan_min_sweep is initialized further down, right after
        # scan_min_sweep (its ceiling) is defined - see that section

        # failed-heading memory (item 13) - lets repeated avoidance
        # cycles at the same stuck spot bias AWAY from a heading that
        # already turned out not to work, instead of the greedy scan
        # scorer reconverging on the same locally-best-but-actually-bad
        # choice every time. Only ever nudges the choice AMONG headings
        # the sweep already found safe - never bypasses stop_dist/
        # turning_dist/is_emergency, all unchanged.
        self.failed_heading_penalty_weight = config.get('failed_heading_penalty_weight', 1.0)
        self.failed_heading_tolerance = math.radians(config.get('failed_heading_tolerance_deg', 30))
        self.failed_headings = deque(maxlen=config.get('failed_headings_maxlen', 4))
        self.last_committed_heading = None  # heading the most recent TURNING committed to - see _start_avoidance_cycle

        # retrace-based backing up (see notes at top of file) - replay
        # the recent forward-driving steering history in reverse when
        # backing away from an obstacle, instead of backing up straight/
        # at a fixed angle
        self.retrace_buffer_sec = datetime.timedelta(
            seconds=config.get('retrace_buffer_sec', 12.0))
        self.path_history = deque()  # (steering_angle, duration), oldest first, forward-driving cycles only
        self.path_history_total = datetime.timedelta(0)
        self.retrace_queue = deque()  # snapshotted from path_history when a retrace backup starts
        self._last_cycle_time = None  # for per-cycle dt - see on_pose2d

        # scan-based turning commit (see notes at top of file) - sweep a
        # minimum angle and pick the best heading seen, instead of
        # grabbing the first momentarily-clear frame
        self.scan_min_sweep = math.radians(config.get('scan_min_sweep_deg', 50))
        self.current_scan_min_sweep = self.scan_min_sweep  # this cycle's actual required sweep, set in _enter_turning (item 13)
        self.side_zone_weight = config.get('side_zone_weight', 0.5)
        self.bearing_penalty_weight = config.get('bearing_penalty_weight', 0.4)
        # early-exit: don't force the full scan_min_sweep when the very
        # first direction tried is already obviously open - only when
        # nothing looks confidently clear (the tall-grass case, where the
        # full sweep-and-compare is actually needed)
        self.scan_confident_dist = self.turning_dist * config.get('scan_confident_margin', 1.5)
        self.scan_confident_streak = 0
        self.scan_samples = []  # (heading, center, left, right), recorded during TURNING
        self.scan_start_heading = None
        # last_heading (IMU) <-> bearing_to_target (compass) frame offset,
        # captured once per avoidance cycle - see _start_avoidance_cycle
        # and _best_scan_heading for why this can't compare the two
        # directly (same caveat _drive_steering already documents)
        self.heading_frame_offset = None

        # escape-mode bookkeeping
        self.escape_counter = 0
        self.in_escape_mode = False
        self.progress_anchor_xy = None

        # bumper contact (item 14 at top of file) - last-resort backstop
        # for the depth camera's blind spots, the big one being straight
        # behind while BACKING_UP. A hit now aborts the current maneuver
        # for real, instead of just zeroing the command for one instant
        # and letting the state machine resume the very same motion next
        # cycle. Past max_bumper_hits with no real progress since, gives
        # up and stops for good - same "don't keep trying forever"
        # pattern as ground_hazard/max_backup_time/max_turn_time.
        self.max_bumper_hits = config.get('max_bumper_hits', 3)
        self.bumper_hit_streak = 0
        self.bumper_stop_active = False
        self._last_xy = (0.0, 0.0)  # updated every on_pose2d cycle - see _on_bumper_hit

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
            self._on_bumper_hit(rear=False)

    def on_bumpers_rear(self, data):
        if data:
            self._on_bumper_hit(rear=True)

    def _on_bumper_hit(self, rear):
        """Physical contact - the last-resort backstop for exactly the
        blind spot the depth camera can't cover (nothing looks behind or
        to the sides while BACKING_UP; a front hit means the forward
        camera missed something too - low object, invalid-data gap,
        etc). Previously on_bumpers_front/rear only zeroed the command
        for a single instant - nothing here changed self.state, so
        on_pose2d's state machine would just recompute and resume the
        SAME motion on the very next cycle. That gap is very likely why
        reversing could still end in contact in a tightly boxed-in space:
        the bumper fired, but nothing stopped the robot from immediately
        trying the same direction again.

        Now a hit persistently aborts whatever was happening: a rear hit
        means backing up further this way isn't actually safe, so pivot
        to trying a turn (using current, live forward sensor data -
        _enter_turning computes severity fresh, so if the front has
        opened up a bit from backing up already, it responds
        accordingly); a front hit backs off straight, same fixed/non-
        clever reaction as the existing is_emergency abort (not the
        moment to trust retrace). _start_avoidance_cycle is called first
        (a no-op if already mid-cycle, e.g. a rear hit during an already-
        active BACKING_UP) so saved_heading/progress-tracking stay
        consistent either way. bumper_hit_streak (reset on real progress,
        same place as escape_counter/failed_headings - see
        _start_avoidance_cycle) escalates to a persistent full stop past
        max_bumper_hits with no progress - a spot that keeps producing
        contact no matter what's tried needs a human, not another retry."""
        self.send_speed_cmd(0, 0)
        self.bumper_hit_streak += 1
        side = 'rear' if rear else 'front'
        print(self.time, 'bumper contact (%s) - aborting current maneuver (streak %d/%d)' %
              (side, self.bumper_hit_streak, self.max_bumper_hits))
        if self.bumper_hit_streak >= self.max_bumper_hits:
            print(self.time, 'repeated bumper contact with no progress since - giving up, stopping')
            self.bumper_stop_active = True
            raise EmergencyStopException()
        self._start_avoidance_cycle(self._last_xy)
        if rear:
            self._enter_turning()
        else:
            self._enter_backing_up(0, use_retrace=False)

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

    def _smooth_heading(self, new_heading):
        """Circular EMA - travel_heading wraps at +-pi, so a plain linear
        EMA would be wrong right around the wrap (e.g. blending 179deg and
        -179deg should stay near +-180, not average to 0). Blend as unit
        vectors instead and convert back. gps_heading_smoothing_alpha=1.0
        makes this a no-op (always the latest reading, old behaviour)."""
        if self.travel_heading is None:
            return new_heading
        alpha = self.gps_heading_smoothing_alpha
        x = (1 - alpha) * math.cos(self.travel_heading) + alpha * math.cos(new_heading)
        y = (1 - alpha) * math.sin(self.travel_heading) + alpha * math.sin(new_heading)
        return math.atan2(y, x)

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
                    # baseline would make travel_heading mostly noise - and
                    # even at the baseline distance, smooth across updates
                    # rather than snapping fully to each new one (see
                    # _smooth_heading and item 9 at top of file)
                    self.travel_heading = self._smooth_heading(initial_bearing(*self.last_gps_pos, lat, lon))
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

    def on_obstacle_zones(self, data):
        if not self.have_obstacle_data:
            self.have_obstacle_data = True
            print(self.time, 'first obstacle_zones reading received, releasing startup hold')
        left, center, right = data
        self.last_obstacle = center
        self.left_dist = left
        self.right_dist = right

        # Treat None as infinitely far away so it doesn't trigger false positives
        l_dist = left if left is not None else float('inf')
        r_dist = right if right is not None else float('inf')

        # stop_dist is the hard-stop distance and is documented ("always
        # applies") as an absolute safety net regardless of state - center
        # alone isn't enough for that: while TURNING the robot drives
        # forward and can clip a wall on either side well before center
        # ever reads close, exactly like this. None (untrusted/no data) on
        # a side must NOT suppress this check - only a genuinely far
        # reading should - hence l_dist/r_dist (inf for None), not the
        # raw left/right, here too.
        stop_now = center < self.stop_dist or l_dist < self.stop_dist or r_dist < self.stop_dist
        self.stop_streak = self.stop_streak + 1 if stop_now else 0

        # The robot is only "clear" if the center AND both sides are further than turning_dist
        is_blocked = (center < self.turning_dist) or (l_dist < self.turning_dist) or (r_dist < self.turning_dist)

        self.turn_streak = self.turn_streak + 1 if is_blocked else 0

    def on_depth_profile(self, data):
        self.depth_profile = data

    def on_ground_hazard(self, data):
        """data is [hazard_bool, ground_valid_frac, ground_dist] for this
        frame from ObstacleDetector3DZones - the streak/debounce logic
        lives here, same pattern as stop_streak/turn_streak above, rather
        than in the sensing module. valid_frac/dist are only for
        diagnostics/logging, not part of the trigger decision itself."""
        if not self.enable_ground_hazard:
            return  # ObstacleDetector3DZones still computes/publishes it - just ignored here
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

    def _best_scan_heading(self):
        """Pick the best-scoring heading recorded in scan_samples during
        the just-finished TURNING sweep. Score is center clearance plus
        side_zone_weight times whichever of left/right is smaller at that
        heading (both zones are field-calibrated now, so worth actually
        weighing in), plus - if a GPS target is set and heading_frame_offset
        is known - a small pull from bearing_penalty_weight toward headings
        closer to bearing_to_target (converted into the same compass
        convention via heading_frame_offset first - see
        _start_avoidance_cycle), only enough to break ties between
        comparable gaps, not to override a clearer one. Also penalizes
        headings close to anything in failed_headings (item 13) - a
        heading recorded there already led nowhere last time (see
        _start_avoidance_cycle's no-progress branch), so re-picking it
        without any push toward something else is exactly what produces
        a forward/back/forward loop in a genuine dead end. The penalty
        tapers linearly to 0 at failed_heading_tolerance and never turns
        positive, so it can only ever make an already-failed direction
        LESS attractive relative to the others actually found this sweep
        - it can't invent a new option or override a real safety check.
        Falls back to saved_heading (the pre-avoidance heading) if the
        sweep produced no samples at all."""
        if not self.scan_samples:
            return self.saved_heading

        def score(sample):
            heading, center, left, right = sample
            sides = [d for d in (left, right) if d is not None]
            side_component = min(sides) if sides else 0.0
            s = center + self.side_zone_weight * side_component
            if self.bearing_to_target is not None and self.heading_frame_offset is not None:
                # convert this sample's IMU heading into the same
                # (compass) convention as bearing_to_target before
                # comparing - see heading_frame_offset
                heading_as_bearing = normalize_angle(heading + self.heading_frame_offset)
                angle_diff = abs(normalize_angle(heading_as_bearing - self.bearing_to_target))
                s -= self.bearing_penalty_weight * angle_diff
            for failed in self.failed_headings:
                diff = abs(normalize_angle(heading - failed))
                if diff < self.failed_heading_tolerance:
                    s -= self.failed_heading_penalty_weight * (1 - diff / self.failed_heading_tolerance)
            return s

        best_heading, *_ = max(self.scan_samples, key=score)
        return best_heading

    def _free_space_steering(self):
        """Continuous depth-based centering nudge from depth_profile
        (item 11 at top of file) - aim toward whichever bins have the
        most clearance beyond a threshold derived from the CURRENT (and,
        with adaptive_distances on, currently speed-scaled) turning_dist,
        weighted by how much clearance beyond that threshold they have.
        Bins at/under the threshold contribute nothing (weight 0), never
        a pull AWAY from anything - this only ever nudges toward openness,
        it doesn't invent a direction when everything's tight (that's the
        discrete avoidance state machine's job, not this one). Returns
        0.0 if there's no profile yet, or nothing beyond the threshold
        anywhere (that situation is exactly what turning_dist/turn_streak
        is for)."""
        if not self.depth_profile:
            return 0.0
        free_space_min_dist = self.turning_dist * self.free_space_lead_margin
        n = len(self.depth_profile)
        total_weight = 0.0
        weighted_pos = 0.0
        for i, d in enumerate(self.depth_profile):
            if d is None or d <= free_space_min_dist:
                continue
            pos = (i + 0.5) / n * 2 - 1  # bin centre, -1 (left edge) .. +1 (right edge)
            weight = d - free_space_min_dist
            total_weight += weight
            weighted_pos += weight * pos
        if total_weight <= 0:
            return 0.0
        aim = weighted_pos / total_weight
        # same sign convention as on_nn_mask's last_dir: aim>0 (open bins
        # skew right) should steer right
        return -math.copysign(min(1.0, abs(aim)) * self.turn_angle, aim)

    def _adaptive_speed(self):
        """Cruising speed scaled by currently sensed clearance, within
        [min_speed, max_speed] (item 10 at top of file). max_speed
        outright if adaptive_speed is off."""
        if not self.adaptive_speed:
            return self.max_speed
        l_dist = self.left_dist if self.left_dist is not None else float('inf')
        r_dist = self.right_dist if self.right_dist is not None else float('inf')
        clearance = min(self.last_obstacle, l_dist, r_dist, self.speed_clearance_ceiling)
        frac = clearance / self.speed_clearance_ceiling if self.speed_clearance_ceiling > 0 else 1.0
        return max(self.min_speed, min(self.max_speed, self.min_speed + frac * (self.max_speed - self.min_speed)))

    def _update_adaptive_distances(self):
        """Recomputes stop_dist/turning_dist from the robot's actual last
        COMMANDED speed (item 10) - called once near the top of
        on_pose2d, every cycle. Deliberately uses _last_commanded_speed,
        not self.max_speed or _adaptive_speed()'s output for this cycle:
        that means stop_dist correctly shrinks while BACKING_UP/TURNING
        (which move slower than cruising) instead of staying pinned to
        cruising-speed math while already moving cautiously. Only a
        rough, uncalibrated model (see decel_mps2 in __init__) - the
        stop_dist_min/max and turning_dist_min/max bounds are the actual
        safety net if the formula ever produces something unreasonable
        for your platform."""
        if not self.adaptive_distances:
            return
        v = abs(self._last_commanded_speed)
        reaction_sec = self.close_confirm_frames / self.depth_fps + self.reaction_margin_sec
        stop = v * reaction_sec + v ** 2 / (2 * self.decel_mps2) + self.stop_dist_margin_m
        stop = max(self.stop_dist_min, min(self.stop_dist_max, stop))
        turning = stop + v * self.turning_lead_sec
        turning = max(self.turning_dist_min, min(self.turning_dist_max, turning))
        self.stop_dist, self.turning_dist = stop, turning

    def _rate_limit_steering(self, target, dt):
        """Caps how fast _drive_steering()'s output can change, so normal
        cruising doesn't snap between corrections (item 11) - deliberately
        NOT applied to the avoidance state machine's own steering, which
        needs to be able to commit decisively. Baseline is
        _last_commanded_steering, updated once per cycle in on_pose2d
        regardless of which state produced it, so this stays continuous
        across a DRIVE <-> avoidance transition instead of jumping from a
        stale value. max_steering_rate=None (max_steering_rate_deg_s<=0)
        disables this outright, same as dt=None (used by _gps_status_line's
        diagnostic-only call - a status line showing a rate-limited value
        would misleadingly suggest the limiting already happened)."""
        if self.max_steering_rate is None or dt is None:
            return target
        max_delta = self.max_steering_rate * dt.total_seconds()
        delta = max(-max_delta, min(max_delta, target - self._last_commanded_steering))
        return self._last_commanded_steering + delta

    def _drive_steering(self, dt=None):
        """Steering to use when NOT actively avoiding an obstacle - i.e.
        this only ever runs from a context where obstacle avoidance has
        already had first say (it is highest priority: it fully overrides
        this method's result by never calling it while avoiding). Three
        signals blend together, in order:

        1. last_dir (road mask, appearance-based "is this drivable") and
           _free_space_steering() (depth-based, "is this open") combine
           via free_space_weight into local_dir - together these are the
           answer to "pick the smoothest path that's also marked
           drivable": smoothness/openness from depth, drivable-marking
           from the road mask, blended rather than either one alone
           deciding. This intentionally does NOT live in the avoidance
           state machine - it's a continuous control question ("how
           should ordinary cruising lean"), not a discrete one ("has
           something forced an unavoidable maneuver") - the state machine
           still owns exactly the latter, unchanged.
        2. GPS bearing still wins over local_dir whenever the mask shows
           little/no road on the side the bearing wants: bearing_blend_
           road_frac is the road-fraction (see on_nn_mask - the
           theoretical max is ~0.5 since the always-masked-out sky half
           counts toward the mean) at which the bearing gets full trust;
           below that it's scaled down proportionally, pure local_dir at
           0. This is a first-pass heuristic, not field tuned - watch
           left_road_frac/right_road_frac against turn_streak false
           positives once you can test outside.
        3. The final result is rate-limited (_rate_limit_steering) so
           ordinary driving doesn't snap between corrections - this step
           only, never the avoidance maneuvers themselves."""
        local_dir = self.last_dir
        if self.free_space_weight > 0:
            local_dir = (1 - self.free_space_weight) * local_dir + self.free_space_weight * self._free_space_steering()

        if not self.follow_gps_target or self.bearing_to_target is None or self.travel_heading is None:
            return self._rate_limit_steering(local_dir, dt)  # no usable GPS heading yet

        error = normalize_angle(self.travel_heading - self.bearing_to_target)
        bearing_steering = max(-self.turn_angle, min(self.turn_angle, error))

        road_frac_that_way = self.left_road_frac if bearing_steering > 0 else self.right_road_frac
        if self.bearing_blend_road_frac > 0:
            weight = min(1.0, road_frac_that_way / self.bearing_blend_road_frac)
        else:
            weight = 1.0
        return self._rate_limit_steering((1 - weight) * local_dir + weight * bearing_steering, dt)

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

    def _enter_backing_up(self, steering, use_retrace=False):
        """steering is the fallback angle used once the retrace queue (if
        any) runs out - or always, when use_retrace=False, which the
        emergency-abort call site deliberately keeps: something is
        dangerously close right now, this is not the moment to be clever
        about replaying history."""
        self.state = State.BACKING_UP
        self.state_start_time = self.time
        self.current_backup_steering = steering
        self.retrace_queue = deque(reversed(self.path_history)) if use_retrace else deque()

    def _next_retrace_steering(self, dt):
        """Pop (steering_angle, remaining_duration) entries off the front
        of retrace_queue - most-recent-forward-step first - consuming dt
        of "replay time" per call. Falls back to current_backup_steering
        once the queue is exhausted (or was never populated - see
        _enter_backing_up's use_retrace)."""
        while self.retrace_queue:
            steering_angle, remaining = self.retrace_queue[0]
            if remaining > dt:
                self.retrace_queue[0] = (steering_angle, remaining - dt)
                return steering_angle
            self.retrace_queue.popleft()
            dt -= remaining
        return self.current_backup_steering

    def _enter_turning(self):
        self.state = State.TURNING
        self.state_start_time = self.time
        if self.in_escape_mode:
            self.turn_sign = -self.turn_sign  # deliberately try the other side
        else:
            self.turn_sign = self._choose_turn_sign()
        self.scan_samples = []
        self.scan_start_heading = self.last_heading
        self.scan_confident_streak = 0

        # severity: 0 at the moment turning_dist was just crossed (barely
        # triggered - a shallow/diagonal graze), 1 once the closest zone
        # is already down at stop_dist (a real, committed block) - scales
        # this cycle's turn amplitude and required sweep between the
        # min_* floor and the full avoid_steering/scan_min_sweep ceiling.
        # Fixes "turns 60-90deg instead of a smaller correction" on a
        # diagonal approach: previously EVERY trigger got the full-lock
        # treatment regardless of how marginal it was. Escape mode always
        # gets full severity - it was reached specifically because milder
        # responses already weren't resolving this spot.
        if self.in_escape_mode:
            severity = 1.0
        else:
            l_dist = self.left_dist if self.left_dist is not None else float('inf')
            r_dist = self.right_dist if self.right_dist is not None else float('inf')
            closest = min(self.last_obstacle, l_dist, r_dist)
            span = max(1e-6, self.turning_dist - self.stop_dist)
            severity = max(0.0, min(1.0, (self.turning_dist - closest) / span))
        self.current_avoid_steering = self.min_avoid_steering + severity * (self.avoid_steering - self.min_avoid_steering)
        self.current_scan_min_sweep = self.min_scan_sweep + severity * (self.scan_min_sweep - self.min_scan_sweep)

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
        # last_heading (IMU yaw) has no known fixed relationship to true
        # north - see _drive_steering()'s docstring - but it should still
        # be trustworthy as a RELATIVE measure over the short span of one
        # avoidance cycle. Capture the offset between it and the compass-
        # referenced travel_heading now, while both are known for the
        # same instant, so _best_scan_heading can later convert a scanned
        # last_heading into the SAME (compass) convention as
        # bearing_to_target instead of comparing the two conventions
        # directly.
        self.heading_frame_offset = (
            normalize_angle(self.travel_heading - self.saved_heading)
            if self.travel_heading is not None else None)
        if self.progress_anchor_xy is not None:
            dist = math.hypot(xy[0] - self.progress_anchor_xy[0],
                              xy[1] - self.progress_anchor_xy[1])
            if dist < self.escape_progress_dist_m:
                self.escape_counter += 1
                # the heading the LAST cycle committed to didn't lead
                # anywhere - remember it so _best_scan_heading can bias
                # away from re-picking the same one this time (item 13)
                if self.last_committed_heading is not None:
                    self.failed_headings.append(self.last_committed_heading)
            else:
                self.escape_counter = 0
                self.failed_headings.clear()  # actually moved - old failures no longer relevant
                self.bumper_hit_streak = 0  # actually moved - past bumper contacts no longer relevant
        self.progress_anchor_xy = xy
        was_escaping = self.in_escape_mode
        self.in_escape_mode = self.escape_counter >= self.escape_after_cycles
        if self.in_escape_mode and not was_escaping:
            print(self.time, 'little net progress over', self.escape_counter,
                  'avoidance cycles - entering escape mode')

    def on_pose2d(self, data):
        x_mm, y_mm, heading_cdeg = data
        xy = (x_mm / 1000.0, y_mm / 1000.0)
        self._last_xy = xy  # for _on_bumper_hit, which fires from its own callback with no pose2d of its own
        if not self.have_imu_heading:
            self.last_heading = math.radians(heading_cdeg / 100.0)

        # per-cycle dt, used both to replay path_history (BACKING_UP) and
        # to record it (forward driving) - see notes at top of file
        dt = (self.time - self._last_cycle_time) if self._last_cycle_time is not None else datetime.timedelta(0)
        self._last_cycle_time = self.time

        # recompute stop_dist/turning_dist from the actual last-commanded
        # speed before anything below reads them this cycle - see item 10
        # at top of file and _update_adaptive_distances
        self._update_adaptive_distances()

        if not self.have_obstacle_data:
            # camera pipeline still booting (OAK-D Pro typically takes a
            # few seconds) - pose2d already flows from the platform at
            # this point, but last_obstacle/left_dist/right_dist are still
            # unset "assume clear" defaults, not a confirmed clear path.
            # Stay stopped rather than drive blind.
            self.send_speed_cmd(0, 0)
            return

        if self.ground_hazard_active:
            # confirmed drop-off/staircase - stay stopped every cycle,
            # don't let the normal state machine drive through it
            self.send_speed_cmd(0, 0)
            return

        if self.bumper_stop_active:
            # repeated bumper contact with no progress - see _on_bumper_hit.
            # terminate_on_stop's EmergencyStopException should already
            # have ended the run by the time this could ever be reached -
            # this is the same belt-and-suspenders pattern as
            # ground_hazard_active above, not a normally-reachable path
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
            speed, steering_angle = -self.backup_speed, self._next_retrace_steering(dt)
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
            # current_avoid_steering/current_scan_min_sweep are severity-
            # scaled once, at entry (_enter_turning) - see item 13. Full
            # avoid_steering/scan_min_sweep still apply unchanged for a
            # close/severe encounter or in escape mode; a shallow/diagonal
            # graze gets a smaller, quicker correction instead.
            speed, steering_angle = self.avoid_speed, self.turn_sign * self.current_avoid_steering
            elapsed = self.time - self.state_start_time
            self.scan_samples.append((self.last_heading, self.last_obstacle, self.left_dist, self.right_dist))
            swept = abs(normalize_angle(self.last_heading - self.scan_start_heading))

            # confident early exit: if THIS heading is already comfortably
            # clear on all three zones (not just barely past turning_dist),
            # don't force grinding through the rest of scan_min_sweep to
            # "prove" it - debounced (close_confirm_frames) so one lucky
            # frame can't trigger it, same guard as the bug this replaced
            l_dist = self.left_dist if self.left_dist is not None else float('inf')
            r_dist = self.right_dist if self.right_dist is not None else float('inf')
            confident_clear = (self.last_obstacle >= self.scan_confident_dist
                                and l_dist >= self.scan_confident_dist
                                and r_dist >= self.scan_confident_dist)
            self.scan_confident_streak = self.scan_confident_streak + 1 if confident_clear else 0

            enough_sweep = swept >= self.current_scan_min_sweep
            confident_enough = self.scan_confident_streak >= self.close_confirm_frames
            if (elapsed > self.min_turn_time and (enough_sweep or confident_enough)) or elapsed > self.max_turn_time:
                if elapsed > self.max_turn_time and not (enough_sweep or confident_enough):
                    print(self.time, 'giving up waiting to sweep enough, committing to best heading seen so far')
                elif confident_enough and not enough_sweep:
                    print(self.time, 'clearly open ahead, committing early without a full sweep')
                best_heading = self._best_scan_heading()
                print(self.time, 'stop turning, realigning to best scanned heading', round(math.degrees(best_heading)))
                # saved_heading now doubles as "REALIGNING's target" - see
                # notes at top of file for why this also removes the old
                # escape-mode special case that used to skip realigning
                self.saved_heading = best_heading
                # persists past the DRIVE reset of saved_heading, unlike
                # saved_heading itself - see _start_avoidance_cycle/item 13
                self.last_committed_heading = best_heading
                self.state = State.REALIGNING
                self.state_start_time = self.time

        elif self.avoid_obstacles and self.state == State.REALIGNING:
            error = normalize_angle(self.saved_heading - self.last_heading)
            elapsed = self.time - self.state_start_time
            if abs(error) < self.realign_tolerance or elapsed > self.max_realign_time:
                print(self.time, 'realigned, resuming road following')
                self._enter_drive()
                speed, steering_angle = self._adaptive_speed(), self._drive_steering(dt)
            else:
                steering_angle = max(-self.realign_max_steering,
                                     min(self.realign_max_steering, error * self.realign_gain))
                speed = self._adaptive_speed()

        elif is_emergency or (not self.avoid_obstacles and self.stop_streak >= self.close_confirm_frames):
            if self.avoid_obstacles:
                print(self.time, 'obstacle too close, backing up (retracing path)', self.last_obstacle)
                self._start_avoidance_cycle(xy)
                self._enter_backing_up(0, use_retrace=True)
                speed, steering_angle = -self.backup_speed, self._next_retrace_steering(dt)
            else:
                speed, steering_angle = 0, 0

        elif self.avoid_obstacles and self.turn_streak >= self.close_confirm_frames:
            self._start_avoidance_cycle(xy)
            if not any_side_clear:
                print(self.time, 'all zones blocked, backing up straight to find room (retracing path)', self.last_obstacle)
                self._enter_backing_up(0, use_retrace=True)
                speed, steering_angle = -self.backup_speed, self._next_retrace_steering(dt)
            else:
                print(self.time, 'obstacle nearby, start turning', self.last_obstacle)
                self._enter_turning()
                speed, steering_angle = self.avoid_speed, self.turn_sign * self.current_avoid_steering

        elif self.waypoint_reached:
            speed, steering_angle = 0, 0

        else:
            speed, steering_angle = self._adaptive_speed(), self._drive_steering(dt)

        # record forward-driving steering history for a future retrace
        # backup (see _enter_backing_up/_next_retrace_steering) - only
        # while actually moving forward, so backing up itself never
        # pollutes what "the way in" means
        if speed > 0:
            self.path_history.append((steering_angle, dt))
            self.path_history_total += dt
            while self.path_history_total > self.retrace_buffer_sec and len(self.path_history) > 1:
                _, old_dt = self.path_history.popleft()
                self.path_history_total -= old_dt

        # single source of truth for "what did we actually just command" -
        # read back by _update_adaptive_distances (speed) and
        # _rate_limit_steering (steering) next cycle, regardless of which
        # state produced this cycle's values
        self._last_commanded_speed = speed
        self._last_commanded_steering = steering_angle

        if self.verbose:
            print(self.time, self.state, speed, steering_angle, self.last_obstacle,
                  self.left_dist, self.right_dist, 'ESCAPE' if self.in_escape_mode else '')
        self.send_speed_cmd(speed, steering_angle)
# vim: expandtab sw=4 ts=4
