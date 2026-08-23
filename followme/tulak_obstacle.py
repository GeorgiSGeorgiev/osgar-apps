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

  15. Compass-based continuous heading - replaces GPS fix-differencing
      as the day-to-day heading source for GPS bearing-following. Field
      report: under tree canopy, Matty would start "doing circles" -
      GPS multipath under trees makes consecutive fixes noisy enough
      that even gps_heading_min_baseline_m/smoothing (item 9) doesn't
      fully save travel_heading. A differenced-GPS heading, by
      construction, amplifies position noise in proportion to how SHORT
      the baseline between the two fixes is - and canopy conditions are
      exactly when that noise is largest and fix availability is worst,
      i.e. the failure mode gets worse exactly when the mitigation needs
      it to get better.

      matty.py's 'rotation' topic (self.last_heading here, via
      on_rotation) already carries the ESP32's own fused yaw - and its
      raw form is commented there as "0=north, clockwise", i.e. a
      genuine COMPASS bearing, not a gyro-only integration with an
      arbitrary power-on reference. Section 5's original notes
      deliberately did NOT trust this ("no confirmed magnetometer/
      absolute-heading source... sign convention undocumented") and used
      GPS fix-differencing instead, specifically to sidestep that
      uncertainty. A confirmed hardware compass changes that calculus:
      an absolute heading available every pose2d cycle (100ms), from a
      magnetometer, is entirely immune to GPS multipath - exactly what
      continuous navigation under canopy needs - so it's now preferred,
      WITH the original uncertainty handled by self-calibrating rather
      than by hardcoding trust in an unverified sign/offset:

        - _compass_heading() does the geometric conversion only (raw
          IMU heading -> compass convention), via the exact inverse of
          matty.py's own "9000 - yaw" transform - this assumes the
          ROTATION DIRECTION (not the zero-point) is right.
          compass_sign (+1/-1) is the escape hatch if that turns out
          mirrored - verify with a hand-turn test: rotate the stationary
          robot a known amount clockwise (viewed from above) and watch
          the logged compass heading (_gps_status_line's heading=...deg
          field, once compass_offset exists) - it should INCREASE by a
          matching amount (compass convention is clockwise-positive,
          same as initial_bearing()); if it decreases instead, set
          compass_sign=-1.
        - compass_offset (radians, learned - see
          _update_compass_calibration) is the additive correction for
          everything ELSE the geometric conversion alone can't know:
          magnetic declination (a real, location-dependent, several-
          degree-plus effect - true north and magnetic north are NOT
          the same direction) and any residual mount misalignment.
          Learned automatically and continuously from whatever GPS
          heading (travel_heading) is ALREADY being trusted under the
          existing baseline/smoothing rules (item 9) - i.e. exactly the
          open-sky, clear-fix conditions where the OLD design already
          worked - via a slow circular EMA
          (compass_offset_smoothing_alpha), additionally gated on fix
          quality/HDOP (_gps_fix_quality_ok,
          compass_calibration_min_quality/compass_calibration_max_hdop)
          so a single marginal fix can't poison it. Never expires or
          resets on its own - declination is a fixed property of
          location and mount alignment does not drift by itself, so a
          value learned once (typically within the first minute or two
          of open-sky driving) keeps being usable through however long a
          subsequent patch of unreliable canopy GPS lasts, which is the
          entire point.

      _current_heading() is the single new source of truth for "which
      way is the robot pointed right now", replacing direct uses of
      self.travel_heading in _drive_steering() and (via the new
      _heading_to_bearing() helper) _best_scan_heading()'s
      bearing_penalty_weight term: compass once calibrated, falling back
      to the old travel_heading (GPS-differenced) behaviour unchanged
      before that first calibration happens (e.g. fresh boot taken
      straight into cover before ever driving in the open) or with
      use_compass_heading=False. bearing_to_target/target_dist
      themselves are UNCHANGED - still recomputed from every fix in
      on_nmea_data regardless of quality, same as before this item -
      only the "which way am I currently facing" question moves off of
      GPS. This directly matches the tradeoff being asked for: GPS still
      decides the destination and roughly how far away it is, compass
      handles being asked "which way should I turn" every single cycle
      without needing to trust a fresh, clean fix to answer it.

      Not field-tested (written without outdoor/hardware access, same
      caveat as the rest of this file) - the geometric conversion
      follows directly from matty.py's own documented "9000 - yaw"
      transform so should be correct if that comment is accurate, but
      compass_sign genuinely needs the hand-turn test above before
      trusting it, and compass_offset needs at least one real open-sky
      drive (long enough to cross gps_heading_min_baseline_m a few
      times) before it has anything useful to fall back on under trees.

  16. Near-target GPS-bearing fade + a stale-adaptive-threshold bug fix,
      both found in a review of this file (2026-08) prompted by the
      "circles under trees" report item 15 addresses:

      (a) Bug: scan_confident_dist (the "comfortably clear, stop the
      TURNING sweep early" threshold - item 8) was computed ONCE in
      __init__ from turning_dist's STATIC initial value. But turning_dist
      itself is adaptive BY DEFAULT since item 10, recomputed every cycle
      from actual commanded speed. At higher cruising speeds (a larger
      live turning_dist), the frozen scan_confident_dist could end up
      SMALLER than the current turning_dist - "confidently clear" no
      longer meant comfortably past the live threshold, so TURNING could
      exit early onto a heading that immediately re-triggers turn_streak.
      Speed-dependent, so invisible in low-speed/indoor testing and more
      likely exactly outdoors at cruising speed. Fixed by storing
      scan_confident_margin instead and recomputing the distance fresh
      from the live turning_dist at its one point of use (on_pose2d's
      TURNING branch) rather than caching it.

      (b) bearing_to_target (item 5) is recomputed from every GPS fix with
      NO smoothing, unlike travel_heading/compass_offset (item 9/15) - a
      deliberate choice at the time, since it's normally compared against
      a target far away. But the SAME fixed GPS position error produces a
      bearing error that GROWS as target_dist shrinks
      (~atan(position_error / target_dist)) - tens of degrees within the
      last 10-15m of a waypoint - a second, independent way to circle the
      target on final approach, on top of whatever item 15 already fixed
      for the heading side of this. roboorienteering/ro.py - already
      cited in item 5 as this file's own precedent for the GPS-heading-
      following approach - disables its equivalent term below 20m of a
      waypoint for exactly this reason; that protection did not carry
      over here. _bearing_distance_scale() restores an equivalent,
      fading GPS-bearing weight smoothly to 0 as target_dist approaches
      waypoint_arrival_dist_m instead of a hard cutoff (no reason to
      trust bearing right as the robot is about to stop anyway), applied
      both in _drive_steering (ordinary cruising) and _best_scan_heading
      (the avoidance-sweep tie-break, for consistency - a noisy
      near-target bearing is just as poor a tie-breaker there). New
      config: bearing_near_target_dist_m (default 15.0 - a few times
      typical non-RTK GPS noise, not calibrated). Set
      <=waypoint_arrival_dist_m to disable and get the old
      always-full-strength behaviour back. Like item 15, not
      field-tested - written from code review, not a live incident.

  17. Tight-passage oscillation, from log analysis of five indoor runs
      (2026-08-20). All five showed a forward/reverse reversal roughly
      every 6 seconds (8.7-9.8 per minute); none had a GPS fix at all, so
      none of this involved GPS/compass - it is purely local avoidance.
      Three separate causes, two fixed here and one in obstdet3d_zones:

      (a) A single side zone under turning_dist was enough to satisfy
      is_blocked, so driving ALONGSIDE a wall - the normal state of
      affairs in a corridor or doorway - kept turn_streak permanently
      confirmed and re-triggered the discrete backup+turn maneuver
      forever. Measured: a right wall steady at 0.36-0.40m with the
      centre clear at 0.72-0.91m. Fixed with side_stop_dist_factor/
      side_turning_dist_factor (see __init__) - the sides keep their own,
      smaller thresholds, because "0.4m beside me" and "0.4m dead ahead"
      are not the same situation. Both default to 1.0 (old behaviour).

      (b) With adaptive_distances on, at low speed BOTH distances hit
      their floors - stop_dist_min 0.5 and turning_dist_min 0.6 - leaving
      a 0.10m band between "start avoiding" and "emergency stop". At
      0.3m/s that is 0.33s, shorter than close_confirm_frames=5 at 10fps
      (0.5s), so stop_streak reliably completed before TURNING could
      achieve anything and every encounter degenerated into backing up.
      Addressed in config (turning_dist_min 0.6 -> 0.9) rather than in
      code - the formula was fine, its floor was too low.

      (c) REALIGNING could only be interrupted by stop_streak, never by
      turn_streak - see the REALIGNING branch in on_pose2d.

      The free-space bins were ALSO reading the floor rather than
      obstacles in those runs (free_space_rows overlapped ground_rows, so
      the percentile picked up the floor at a constant ~1.3m and every bin
      reported the same distance - no steering signal, and an early-warning
      horizon capped at 1.3m regardless of free_space_lead_margin). That
      one was already corrected in the config before these fixes landed;
      it is noted here because it is invisible from the code alone.

  18. Two loop fixes, from replaying the state machine against the
      2026-08-20 20:00-20:07 runs. Item 17's sensing fixes worked - the
      synthetic far-fill readings went from 11.2% of frames to ~0%, flat
      ground-contaminated bin profiles roughly halved, bumper contacts
      across four runs dropped from seven to one - but the forward/reverse
      reversal rate did not improve at all (8.6-11.7/min), because the
      loops were in the state machine, not the sensing.

      (a) REGRESSION introduced by item 17(c). Gating REALIGNING's new
      abort on turn_streak was wrong: turn_streak also confirms on a side
      zone, and in a corridor or doorway a side zone sits under threshold
      continuously, so the abort condition was already satisfied at the
      moment TURNING handed over. REALIGNING therefore lasted exactly one
      cycle every time - 0.6s total across a 38s run - bouncing
      REALIGNING -> DRIVE -> TURNING -> REALIGNING and never recovering the
      heading, which is what made a doorway unpassable. Now gated on
      center_blocked_streak (centre zone only, same debounce), so "there
      are walls beside me, as there have been all along" no longer reads
      as "the heading I committed to is blocked". Flank contact is still
      covered by stop_streak via is_emergency, unchanged.

      (b) Pre-existing, and the reason the second loop never resolved: an
      emergency abort out of TURNING calls _enter_backing_up directly, and
      BACKING_UP calls _enter_turning directly, so a TURNING -> abort ->
      BACKING_UP -> TURNING cycle touches NEITHER stuck detector -
      _start_avoidance_cycle returns immediately while saved_heading is
      set (so escape_counter never increments and progress is never
      re-anchored), and _enter_turning skips _choose_turn_sign entirely
      while _turn_sign_committed is True (so the same-direction detector
      never runs and the turn direction can never change). Six identical
      ~3.2s cycles were logged with every stuck detector reporting
      nothing. max_maneuver_aborts counts consecutive aborts within one
      episode and, past the limit, forces escape mode AND clears
      _turn_sign_committed so the direction is actually re-decided -
      without which escape mode alone would keep repeating the identical
      maneuver.

      (c) REGRESSION introduced by item 17(a), and the actual reason the
      doorway failed. Giving the sides their own threshold reduced how
      often a passing wall triggers avoidance, but a side under that
      (smaller) threshold still triggered it outright - including when the
      centre was wide open. Replayed from the log: TURNING entered with
      centre at 2.54m and the right door frame at 0.53m, one centimetre
      under the 0.54m side threshold. Turning away from the frame pointed
      the camera into the wall beside the doorway, so the centre collapsed
      to 0.73m within 0.7s, which then read as a genuine block and
      justified more avoidance - a self-inflicted loop directly in front
      of a passable opening. side_trigger_center_clear_factor makes a
      side-only blockage yield while the centre is open by that multiple
      of turning_dist, which is exactly the doorway/gap signature. Only
      the discrete maneuver is suppressed: _edge_bin_correction still
      steers away from the near frame continuously, and the side
      hard-stop is untouched.

      (d) Follow-ups from the 2026-08-20 20:23-20:30 runs, where (c) let
      Matty drive THROUGH doorways but brushing the frames, and a
      slightly-closed U-shaped area took ~4.5 minutes to escape:

      - _edge_bin_correction returned exactly 0 in 86% of frames with a
        flank inside 0.6m, because both edge bins were under threshold -
        which is the permanent state of affairs in any doorway or
        corridor, i.e. the centering signal switched off precisely where
        clearance was tightest. It now treats a both-blocked gap as a
        CENTERING problem, steering away from the closer flank in
        proportion to how lopsided the two are (symmetric still yields
        ~0, preserving the original deferral where it was right).
      - any_side_clear, which releases BACKING_UP, was still judged
        against the CENTRE turning_dist after item 17(a) gave the sides
        their own. That left a dead band - a flank between the side and
        centre thresholds was neither blocked nor clear - so backing up
        continued toward max_backup_time. The U-shape run spent 122s of
        284s (43%) reversing.
      - escape_backup_time_boost is set to 1.0 in config (was 1.5).
        Escape mode was active 81% of that run, so the boost was
        effectively permanent, making the single riskiest maneuver
        (reversing, where the only sensor is the rear bumper) both longer
        and more frequent, in a situation it was not resolving.
      - failed_headings_maxlen 4 -> 8 and deep_stuck_candidate_pool
        3 -> 4: the scan kept re-committing to the same 195-262deg
        sector (195, 201, 205, 230, 234, 256, 261, 262 all logged), and
        a 4-deep memory at 30deg tolerance cannot hold that many
        distinct-but-equally-dead headings at once.

  19. Frontal collisions (2026-08-20 20:41-20:46 runs: 14 front-bumper
      hits across five runs, against 3 in the batch before). Every one has
      the same signature - REALIGNING, steering locked at
      realign_max_steering, speed ramping to ~0.32m/s, flanks at
      0.33-0.64m and the centre at 0.92-0.98m. Three causes stacked:

      (a) REALIGNING had no flank awareness after item 18(a) narrowed its
      abort to the centre zone, and it is the one forward-driving state
      that never consults the depth-based steering blend - it just holds
      up to realign_max_steering toward its committed heading. Re-widened
      to turn_streak, which is safe now that item 18(c) stops a side-only
      blockage confirming while the way ahead is open.

      (b) side_stop_dist_factor at 0.6 put the flank hard-stop at 0.30m,
      too permissive to catch these. Raised to 0.9 in config. Worth being
      explicit that this trades against narrow passages in the other
      direction: a doorway frame passing at 0.45m now stops the robot.
      Stopping is recoverable, hitting the frame is not, so it is the
      right way round to be wrong - but this is the first knob to lower
      again if doorways start refusing.

      (c) The binding constraint was neither of the above. min_speed was
      0.3 in config, so _adaptive_speed could NOT slow below 0.3m/s no
      matter how tight the space - with clearance at 0.4m it still
      commanded 0.327m/s. At that speed the 5-frame (0.5s) debounce alone
      consumes 0.16m before anything reacts, and the flank in these logs
      closed from 0.52m to 0.33m in 0.4s. Simulating the fully-restored
      side_stop_dist_factor=1.0 against the recorded distances, stop_streak
      still reaches only 4 of the 5 frames it needs before contact - so
      the threshold was never going to be the fix on its own. min_speed is
      now 0.15 (the module default), and stop_confirm_frames splits the
      hard stop off the shared debounce so it can confirm in 2 frames
      while turn commitment keeps its 5.

  20. Over-caution (2026-08-20 20:59-21:05 runs). The item 19 fixes worked
      - front-bumper contacts went from 14 across five runs to 1, and
      doorways pass reliably - but Matty became slow and hesitant. Two
      causes:

      (a) _adaptive_speed took min(centre, left, right) against
      speed_clearance_ceiling, judging a wall ALONGSIDE on exactly the same
      scale as an obstacle in the direction of travel - the same mistake
      the trigger thresholds had before item 17 split them. Indoors a flank
      is nearly always the smallest of the three, so it set the speed
      essentially all the time: measured median forward speed 0.20m/s
      against max_speed 0.5, with the median closest reading 0.50m
      (0.15 + (0.50/3.0)*0.35 = 0.208 - the number came from a side).
      speed_side_clearance_factor converts a flank to centre-equivalent
      room first; speed_clearance_ceiling drops 3.0 -> 2.0 in config since
      3m of clearance is unreachable indoors and the robot could therefore
      never approach max_speed. Net effect: ~0.30m/s past a wall at 0.5m,
      still ~0.24m/s closing on the same distance dead ahead.

      (b) stop_confirm_frames=2 (item 19) was too twitchy - 78 emergency
      aborts in 266s and 415 state transitions, against 221 transitions
      before. Raised to 3, which still confirms 0.14s ahead of the
      recorded impact point and, with (a)'s lower approach speeds, with
      more margin than 2 frames bought at the old speeds.

  21. Polar clearance memory - PHASE 1, BOOKKEEPING ONLY. Nothing reads it
      to make a decision; see the notes in __init__ and _update_polar_
      memory. Motivation: the U-shaped-area failure is not really a
      decision problem, it is a memory problem. The camera sees ~58deg at
      a time, and the robot spent 101s of a 266s entrapment in TURNING -
      it had physically pointed at most directions repeatedly - but
      scan_samples is wiped per episode and consulted once at the end of a
      single sweep, so every one of those observations was thrown away and
      each cycle re-decided from a fresh ~58deg keyhole. The scanned
      headings duly kept clustering in one sector (181-224deg logged).
      This accumulates them into a 360deg ring instead.

      Built deliberately as inert bookkeeping first so it can be validated
      against recorded runs - specifically, whether it would have known
      about a way out during a real entrapment - before being allowed to
      influence anything. A later phase would read it in
      _best_scan_heading, which only ever reorders candidates that already
      passed the unchanged safety checks.

      Note on the alternative: a deliberate 360deg "survey spin" (as in
      ROS move_base's rotate_recovery) is NOT directly available here.
      Matty is articulated/Ackermann, not differential drive - from
      matty.py, radius = (0.32/2)/tan(joint/2), so even at the 45deg
      steering limit the minimum turning radius is ~0.39m and a full turn
      sweeps a ~0.8m circle plus the robot's footprint, through space no
      sensor covers. In a closed-in dead end that is exactly the maneuver
      most likely to make things worse. Accumulating the turning the robot
      already does costs nothing and needs no room.

  22. Blind-sensor hold. Outdoor runs (2026-08-23, bright sun) showed the
      depth camera returning almost nothing: whole frames 99.6-100%
      invalid INCLUDING the ground rows - not a horizon/sky problem, a
      total stereo blackout - with the centre zone pinned at its 0.0
      fail-safe for 100% of several runs. Because 0.0 means "assume the
      worst", the state machine read it as an obstacle at zero distance
      and answered with backup-and-turn, so a blind camera produced
      continuous REVERSING: the one maneuver with no forward sensor
      coverage, guided only by a rear bumper. Nothing in the file
      distinguished "something is close" from "I cannot see" - both
      arrive as center == 0.0.

      Now they are distinct. _update_blind_state confirms blindness from
      the centre fail-safe AND most of depth_profile being unknown (both
      required - a blank centre alone is still ordinary close-obstacle
      business), debounced in both directions, and on_pose2d holds the
      robot stationary ahead of every other check. It releases by itself
      once real depth returns.

      This also closes a camera-boot gap: have_obstacle_data only waits
      for the FIRST obstacle_zones message, so as soon as the pipeline
      began publishing - even while still settling and emitting blank
      frames - the robot was free to drive on empty data. It now stays
      put until frames actually carry depth, and re-arms if they stop.

      Known tradeoff: a surface pressed hard against the lens blanks the
      frame the same way a blackout does, and this cannot tell them apart
      (obstdet3d_zones documents the same ambiguity for ground_hazard).
      In that case Matty will now sit still where it previously backed
      off. That is the deliberate choice - reversing blind is exactly the
      risk this exists to remove - but it means a nose-in-a-bush stall
      needs a human or a manual nudge rather than recovering itself.

  All new config keys (escape_after_cycles, escape_progress_dist_m,
  escape_backup_time_boost, ground_hazard_confirm_frames,
  terminate_on_ground_hazard, follow_gps_target,
  gps_heading_min_baseline_m, waypoint_arrival_dist_m,
  bearing_blend_road_frac, bearing_near_target_dist_m, retrace_buffer_sec, scan_min_sweep_deg,
  scan_confident_margin, side_zone_weight, bearing_penalty_weight,
  gps_heading_smoothing_alpha, adaptive_distances, decel_mps2,
  reaction_margin_sec, depth_fps, stop_dist_margin_m, turning_lead_sec,
  stop_dist_min, stop_dist_max, turning_dist_min, turning_dist_max,
  adaptive_speed, min_speed, speed_clearance_ceiling, free_space_weight,
  free_space_lead_margin, max_steering_rate_deg_s, enable_ground_hazard,
  min_avoid_steering_deg, min_scan_sweep_deg, failed_heading_penalty_weight,
  failed_heading_tolerance_deg, failed_headings_maxlen, max_bumper_hits,
  use_compass_heading, compass_sign, compass_offset_smoothing_alpha,
  compass_calibration_min_quality, compass_calibration_max_hdop,
  side_stop_dist_factor, side_turning_dist_factor,
  side_trigger_center_clear_factor, max_maneuver_aborts,
  edge_correction_center_in_gap, stop_confirm_frames,
  speed_side_clearance_factor, polar_memory_bins, polar_memory_max_age_sec,
  polar_memory_max_travel_m, polar_profile_hfov_deg,
  polar_memory_log_interval_sec, polar_memory_far_fill_m,
  blind_hold_enabled, blind_profile_valid_frac, blind_confirm_frames,
  blind_clear_frames)
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
import random
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
        # Separate, normally SMALLER thresholds for the left/right zones
        # (item 17 at top of file). Both default to 1.0 = the old behaviour
        # (sides judged against exactly the same distance as dead ahead).
        # Rationale: those two distances answer different questions. Dead
        # ahead, turning_dist means "I am driving into this". Off to the
        # side it means "this is beside me" - and a surface you can pass
        # at 0.4m laterally is one you must stop for at 0.4m head-on. With
        # one shared number, any corridor narrower than about 2*turning_dist
        # keeps a side zone permanently under threshold, so turn_streak
        # never clears and the discrete avoidance maneuver fires
        # continuously in exactly the narrow passages it cannot help with -
        # field-confirmed (2026-08-20): a right-hand wall steady at
        # 0.36-0.40m with the centre clear at 0.72-0.91m produced a
        # forward/reverse cycle roughly every 6 seconds.
        # Keeping a wall alongside is the continuous steering signal's job
        # (_edge_bin_correction/_free_space_steering), not the state
        # machine's; the hard-stop net still watches the sides, just at
        # side_stop_dist_factor of the centre distance.
        # NOT calibrated - these are the two knobs for the narrow-passage
        # vs. clipping tradeoff. Lower = threads tighter gaps but leaves
        # less margin against clipping a flank; 1.0 restores old behaviour.
        self.side_stop_dist_factor = config.get('side_stop_dist_factor', 1.0)
        self.side_turning_dist_factor = config.get('side_turning_dist_factor', 1.0)
        # Doorway/gap rule (item 18c): ignore a SIDE-ONLY blockage while the
        # centre is open by at least this multiple of turning_dist. 0
        # disables it (any side under threshold triggers, as before).
        #
        # A doorway is defined by exactly this signature - frames close on
        # one or both sides, clear passage straight ahead - and it is the
        # one case where reacting to the side is actively wrong: the
        # avoidance turn swings the camera off the opening and into the
        # wall beside it, which then reads as a real centre blockage and
        # justifies further avoidance. Field log (2026-08-20 20:07):
        # TURNING triggered with centre at 2.54m and the right frame at
        # 0.53m, a hair under the side threshold; within 0.7s of turning
        # away the centre had collapsed 2.49 -> 1.29 -> 0.73m and the robot
        # spent the remaining 20s looping in front of a doorway it had
        # been lined up to drive through.
        #
        # Only suppresses the DISCRETE maneuver. The continuous steering
        # nudge (_edge_bin_correction) still pushes away from the near
        # frame, and the hard-stop net still watches the sides at
        # side_stop_dist_factor - so a side closing to genuinely
        # unsafe range still stops the robot; it just no longer
        # abandons a passable gap because one flank is near.
        self.side_trigger_center_clear_factor = config.get('side_trigger_center_clear_factor', 0)
        self.close_confirm_frames = config.get('close_confirm_frames', 3)
        # Separate, normally SHORTER debounce for the HARD STOP than for
        # the turn trigger (item 19). Defaults to close_confirm_frames, i.e.
        # the single shared debounce this replaced.
        #
        # These two decisions do not deserve the same amount of proof.
        # Committing to a turn is expensive and hard to undo, so waiting
        # several frames to be sure is right. Stopping is cheap and fully
        # recoverable - if it turns out to be a false alarm the robot simply
        # resumes - yet it was paying the same 5-frame (0.5s at 10fps) tax.
        # Field log (2026-08-20 20:42): closing from 0.52m to 0.33m on a
        # flank took 0.4s, so the stop never confirmed before contact; even
        # restoring side_stop_dist_factor to 1.0 only reached 4 of the 5
        # required frames. The debounce itself was the binding constraint.
        self.stop_confirm_frames = config.get('stop_confirm_frames', self.close_confirm_frames)
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
        # How much a SIDE distance counts against speed relative to the same
        # distance dead ahead (item 20). 1.0 = identical, the old behaviour.
        # A side reading is divided by this before entering the clearance
        # minimum, so e.g. 0.6 makes a flank at 0.6m count like 1.0m of
        # room ahead.
        #
        # Same mistake the trigger thresholds had before
        # side_turning_dist_factor/side_stop_dist_factor (item 17): a wall
        # ALONGSIDE was being judged on exactly the same scale as an
        # obstacle in the direction of travel. Indoors a flank is almost
        # always the smallest of the three, so it set the speed essentially
        # all the time - measured over the 2026-08-20 20:59 runs, forward
        # speed sat at a median of 0.20m/s against max_speed 0.5, with the
        # median closest reading being 0.50m: 0.15 + (0.50/3.0)*0.35 =
        # 0.208, i.e. the number came from a side, not from anything ahead.
        self.speed_side_clearance_factor = config.get('speed_side_clearance_factor', 1.0)
        # caps how fast _adaptive_speed()'s output can change per second -
        # see _rate_limit_speed. None/<=0 disables it (old behaviour:
        # speed can snap instantly to whatever clearance says)
        accel = config.get('max_speed_accel_mps2', 0.3)
        self.max_speed_accel = accel if accel and accel > 0 else None
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
        # controlled exploration once ALREADY stuck in escape mode - see
        # _best_scan_heading
        self.deep_stuck_cycle_span = config.get('deep_stuck_cycle_span', 5)
        self.deep_stuck_max_explore_frac = config.get('deep_stuck_max_explore_frac', 0.4)
        self.deep_stuck_candidate_pool = config.get('deep_stuck_candidate_pool', 3)

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
        # near-target bearing fade (item 16 at top of file) - GPS position
        # noise (unavoidable without RTK) turns into growing BEARING error
        # as target_dist shrinks (a fixed few meters of position noise is
        # ~30deg of bearing error at 10m, 45deg+ inside 5m) - see item 16
        # for the full rationale and _bearing_distance_scale(). Below this
        # distance, weight starts fading toward 0 as target_dist approaches
        # waypoint_arrival_dist_m, instead of following a noisy near-target
        # bearing at full strength the way _drive_steering used to.
        # roboorienteering/ro.py (this file's own precedent for GPS-
        # heading-following - see item 5/15) already disabled its
        # equivalent term below 20m for the same reason; this restores an
        # equivalent, fading smoothly rather than as a hard cutoff. Set
        # <= waypoint_arrival_dist_m to disable (bearing always at full
        # strength down to arrival, the old behaviour).
        self.bearing_near_target_dist_m = config.get('bearing_near_target_dist_m', 15.0)
        self.gps_log_interval = datetime.timedelta(seconds=config.get('gps_log_interval_sec', 2.0))

        # Compass-based continuous heading (item 15 at top of file) -
        # uses Matty's ESP32 IMU-fused yaw (last_heading, via
        # on_rotation) as the day-to-day heading source for GPS bearing-
        # following instead of travel_heading (GPS fix differencing),
        # once calibrated - see _current_heading. GPS still determines
        # the target itself (QR-decoded coordinates) and bearing/
        # distance to it (on_nmea_data, unchanged); this only replaces
        # what tells the robot which way IT is currently pointed, which
        # is what needs to keep working under tree canopy where GPS
        # itself doesn't.
        self.use_compass_heading = config.get('use_compass_heading', True)  # <-- on/off switch
        # field-calibration knob for the IMU yaw's rotation direction -
        # see the hand-turn test in item 15. +1 assumes matty.py's own
        # "0=north, clockwise" comment on raw yaw is correct as-is.
        self.compass_sign = config.get('compass_sign', 1)
        # circular EMA alpha for compass_offset (magnetic declination +
        # mount misalignment, learned from GPS - see
        # _update_compass_calibration). Deliberately slow/conservative -
        # this value is meant to still be trustworthy long after GPS
        # itself has gone bad (tree canopy), so a single so-so open-sky
        # fix should barely move it.
        self.compass_offset_smoothing_alpha = config.get('compass_offset_smoothing_alpha', 0.1)
        # fix-quality gate on LEARNING a new calibration point (see
        # _gps_fix_quality_ok) - deliberately a higher bar than what's
        # needed just to update position/bearing_to_target (still done
        # unconditionally elsewhere), since a bad calibration point
        # poisons compass_offset for every future cycle relying on it,
        # including the tree-canopy ones this feature exists to help.
        # quality/hdop come from the GGA sentence (osgar/drivers/gps.py)
        # and may be None on a receiver that doesn't report them -
        # treated as passable rather than failing closed in that case.
        self.compass_calibration_min_quality = config.get('compass_calibration_min_quality', 1)
        self.compass_calibration_max_hdop = config.get('compass_calibration_max_hdop', 3.0)
        self.compass_offset = None  # radians, learned - see _update_compass_calibration
        self.compass_last_calibrated = None  # self.time of the last calibration update, diagnostics only

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
        # centre-only counterpart of turn_streak (item 18) - "is the path
        # STRAIGHT AHEAD blocked", as opposed to turn_streak's "is anything
        # in any zone blocked". REALIGNING needs specifically this one: it
        # drives forward toward a committed heading, and in a corridor or
        # doorway a side zone is under threshold continuously, so
        # turn_streak is essentially always confirmed there and cannot
        # distinguish "the way ahead just closed" from "there are walls
        # beside me, as there have been all along".
        self.center_blocked_streak = 0
        self.depth_profile = []  # free_space_bins distances - see on_depth_profile/_free_space_steering

        # --- blind-sensor hold (item 22) ---
        # When the depth camera returns essentially nothing, the robot
        # holds still instead of maneuvering. Distinct from every other
        # stop in this file: those all mean "something is there", this one
        # means "I cannot see, full stop".
        #
        # Why it needs to be its own state rather than falling out of the
        # existing logic: an all-invalid centre window makes _dist return
        # its fail_value of 0.0 ("assume worst"), which the state machine
        # reads as an obstacle at zero distance and answers with the
        # standard backup-and-turn. So a blind camera produced continuous
        # REVERSING - the single maneuver with no forward sensor coverage
        # at all, steered only by a rear bumper. Field data (2026-08-23
        # outdoors, bright sun): whole frames 99.6-100% invalid INCLUDING
        # the ground rows, centre pinned at the 0.0 fail-safe for 100% of
        # several runs, robot reversing the entire time.
        #
        # Also covers camera boot. have_obstacle_data only waits for the
        # FIRST obstacle_zones message, so once the pipeline starts
        # publishing - even if those first frames are blank while it is
        # still settling - that gate opens and the robot drives on empty
        # data. This hold keeps it stationary until frames actually carry
        # depth, and re-arms automatically if they stop.
        #
        # Deliberately requires BOTH conditions: the centre blank AND most
        # of the wider profile blank. The centre alone going invalid is
        # something the ordinary obstacle logic should keep handling as a
        # possible close object.
        self.blind_hold_enabled = config.get('blind_hold_enabled', True)
        self.blind_profile_valid_frac = config.get('blind_profile_valid_frac', 0.25)
        self.blind_confirm_frames = config.get('blind_confirm_frames', 3)
        self.blind_clear_frames = config.get('blind_clear_frames', 3)
        self.depth_blind_active = False
        self.blind_streak = 0
        self.blind_clear_streak = 0
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
        # edge-specific complement to the whole-profile average above -
        # see _edge_bin_correction()
        self.edge_bins_watched = config.get('edge_bins_watched', 1)
        self.edge_correction_min_rad = math.radians(config.get('edge_correction_min_deg', 8))
        self.edge_correction_max_rad = math.radians(config.get('edge_correction_max_deg', 35))
        # centre the robot in a gap where BOTH flanks are inside the
        # threshold (doorway/corridor) instead of contributing nothing
        # there - see _edge_bin_correction. False restores the old
        # both-blocked-means-stay-silent behaviour.
        self.edge_correction_center_in_gap = config.get('edge_correction_center_in_gap', True)
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
        # once _choose_turn_sign() has picked a direction for the current
        # avoidance episode, re-entries into TURNING within that SAME
        # episode (e.g. after an emergency-abort-to-backup mid-turn) are
        # biased to keep it rather than re-deciding from scratch - see
        # _choose_turn_sign. Reset per fresh episode in
        # _start_avoidance_cycle.
        self._turn_sign_committed = False
        self.turn_sign_switch_margin_m = config.get('turn_sign_switch_margin_m', 0.2)
        # how much clearer (bin open-fraction) the OTHER side needs to be
        # before _bin_override_turn_sign flips the zone-preferred pick -
        # see that method
        self.bin_override_margin = config.get('bin_override_margin', 0.3)
        # same-direction-repeat loop detector - a faster, position-
        # independent complement to the XY-progress-based escape trigger
        # below - see _choose_turn_sign
        self.last_turn_sign_choice = None
        self.same_direction_repeats = 0
        self.max_same_direction_repeats = config.get('max_same_direction_repeats', 1)
        # repeated emergency-aborted maneuvers within ONE episode (item 18).
        # Both existing stuck detectors are blind to this loop: the
        # XY-progress check lives in _start_avoidance_cycle, which returns
        # immediately while saved_heading is set, and the same-direction
        # check lives in _choose_turn_sign, which _enter_turning skips once
        # _turn_sign_committed is True. An emergency abort out of TURNING
        # goes straight to _enter_backing_up and back to _enter_turning
        # without touching either, so a TURNING -> abort -> BACKING_UP ->
        # TURNING cycle can repeat indefinitely while every stuck detector
        # reports nothing. Field log (2026-08-20 20:07): six identical
        # ~3.2s cycles, no escape mode, no progress, run ended by the
        # operator's emergency stop.
        self.max_maneuver_aborts = config.get('max_maneuver_aborts', 3)
        self.maneuver_abort_streak = 0

        # proportional avoidance-turn severity (item 13 at top of file) -
        # does NOT touch stop_dist/turning_dist (when avoidance triggers
        # is unchanged) - only how hard it turns once triggered. Full
        # avoid_steering/scan_min_sweep are still the ceiling, reached at
        # severity=1 - unchanged from before for a close/severe encounter
        # and, deliberately, always for escape mode (see _enter_turning).
        self.min_avoid_steering = math.radians(config.get('min_avoid_steering_deg', 20))
        self.min_scan_sweep = math.radians(config.get('min_scan_sweep_deg', 10))
        # bin-based feasibility cap on the turn amplitude above - see
        # _avoidance_bin_scale()
        self.avoidance_bin_scale_floor = config.get('avoidance_bin_scale_floor', 0.5)
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
        # full sweep-and-compare is actually needed). scan_confident_margin
        # is stored (not resolved into a fixed distance here) because
        # turning_dist itself is adaptive (see item 10/_update_adaptive_
        # distances) - a distance computed once at boot from the STATIC
        # initial turning_dist would silently go stale the moment speed-
        # based adaptation kicks in, understating "confident" at higher
        # cruising speeds (where turning_dist grows) and letting TURNING
        # exit early on a reading that's no longer actually past the
        # CURRENT threshold. Recomputed fresh from the live turning_dist
        # at the point of use instead - see on_pose2d's TURNING branch.
        self.scan_confident_margin = config.get('scan_confident_margin', 1.5)
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

        # --- polar clearance memory (item 21) - PHASE 1: BOOKKEEPING ONLY ---
        # Nothing in this class reads polar_memory to make a decision. It is
        # built and logged so it can be validated offline against recorded
        # runs before it is ever allowed to steer anything, which is the
        # whole point of doing it in this order: it cannot regress current
        # behaviour because no behaviour consults it.
        #
        # What it is: a ring of polar_memory_bins direction bins covering
        # the full 360deg around the robot, each holding the most recent
        # clearance observed in that direction. The camera only sees ~58deg
        # at a time, so any single frame fills about four bins - but the
        # robot turns constantly while avoiding (101s of TURNING in the
        # 2026-08-20 20:59 U-shaped-area run), so over a few seconds of
        # maneuvering it has physically pointed at most directions already.
        # Today every one of those observations is discarded: scan_samples
        # is wiped per episode and only consulted at the end of one sweep.
        # This keeps them.
        #
        # Direction is indexed by last_heading, which the 2026-08-07 log
        # analysis confirmed is a genuine magnetometer-referenced compass
        # (~0.02deg/s drift over 6min outdoors), so the index is stable.
        # Indoors the magnetometer is distorted, but this memory is only
        # ever meant to be valid for seconds, over which relative heading
        # is fine either way.
        #
        # Entries expire on BOTH age and displacement. A polar map in the
        # robot's own frame stops describing the world once the robot
        # translates - but "stuck" means it is not translating, so the
        # memory is freshest exactly when it matters. polar_memory_bins=0
        # disables the whole thing.
        self.polar_memory_bins = config.get('polar_memory_bins', 24)
        self.polar_memory_max_age = datetime.timedelta(
            seconds=config.get('polar_memory_max_age_sec', 20.0))
        self.polar_memory_max_travel_m = config.get('polar_memory_max_travel_m', 1.5)
        # horizontal FOV the depth_profile bins span. free_space_cols
        # (60..580 of 640) is most of the frame, so this is close to the
        # camera's full HFOV. NOT calibrated - the OV9282's listed HFOV for
        # the standard (non-wide) OAK-D Pro is ~72deg; check against a real
        # scene before trusting the bin->angle mapping precisely.
        self.polar_profile_hfov = math.radians(config.get('polar_profile_hfov_deg', 66))
        self.polar_memory_log_interval = datetime.timedelta(
            seconds=config.get('polar_memory_log_interval_sec', 5.0))
        # Readings at/above this are not stored at all. They are the depth
        # far-mask fill constant (oak_camera_v3's depth_far_mask_value_mm,
        # 15000 by default), not measurements - see obstdet3d_zones'
        # _sanitize_zones, which drops them from the L/C/R zones but
        # CANNOT clean depth_profile the same way: its own
        # _sanitize_profile only strips a far bin isolated between two
        # nearer neighbours, so a coherent multi-bin fill region passes
        # straight through. Caught by the phase-1 offline check on the
        # 2026-08-20 21:22 run, where the memory was cheerfully recording
        # "15.00m open" behind the robot in an indoor U-shaped area.
        # Remembering a phantom opening is far worse than remembering
        # nothing: unlike a live reading it is never revisited or
        # contradicted, so it would sit in the ring as the most attractive
        # direction available until it aged out. Keep this just under the
        # oak module's depth_far_mask_value_mm/1000.
        self.polar_memory_far_fill_m = config.get('polar_memory_far_fill_m', 13.5)
        self.polar_memory = [None] * max(0, self.polar_memory_bins)  # per bin: (dist_m, time, xy)
        self._last_polar_log_time = None

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

    @staticmethod
    def _blend_angle(old_angle, new_angle, alpha):
        """Circular EMA between two angles (radians) - wrap-around-safe
        (e.g. blending 179deg and -179deg should stay near +-180, not
        average to 0deg). Blend as unit vectors instead of the raw
        angles and convert back. alpha=1.0 makes this a no-op (always
        new_angle). Shared by _smooth_heading (travel_heading) and
        _update_compass_calibration (compass_offset) - same wrap-around
        problem, two different callers."""
        x = (1 - alpha) * math.cos(old_angle) + alpha * math.cos(new_angle)
        y = (1 - alpha) * math.sin(old_angle) + alpha * math.sin(new_angle)
        return math.atan2(y, x)

    def _smooth_heading(self, new_heading):
        """See _blend_angle. gps_heading_smoothing_alpha=1.0 makes this a
        no-op (always the latest reading, old behaviour)."""
        if self.travel_heading is None:
            return new_heading
        return self._blend_angle(self.travel_heading, new_heading, self.gps_heading_smoothing_alpha)

    def _compass_heading(self):
        """self.last_heading (OSGAR convention: 0=east, anticlockwise -
        see on_rotation/matty.py's 'rotation' topic) converted into the
        SAME compass convention (0=north, clockwise/east-positive) that
        initial_bearing()/bearing_to_target already use - the exact
        inverse of matty.py's own '9000 - yaw' conversion:
        compass = 90deg - osgar_heading. This assumes matty.py's rpy
        comment ("Roll, Pitch, Yaw - debug output 1:1 to ESP32 (0=north,
        clockwise)") is accurate, i.e. the ESP32 fuses an onboard
        magnetometer into yaw rather than just integrating the gyro from
        an arbitrary power-on reference - see item 15 at top of file for
        why that was previously treated as unconfirmed.

        compass_sign is the field-calibration knob for the ROTATION
        DIRECTION specifically (a mirrored axis, not just an offset) -
        see the hand-turn test in item 15's docs; it does NOT include
        compass_offset (the separately-learned correction for magnetic
        declination + mount misalignment, see
        _update_compass_calibration) - callers wanting the final,
        calibration-corrected heading should use _current_heading()
        instead. Returns None if the ESP32 has never sent a 'rotation'
        sample yet (self.have_imu_heading)."""
        if not self.have_imu_heading:
            return None
        return normalize_angle(math.pi / 2 - self.compass_sign * self.last_heading)

    def _gps_fix_quality_ok(self, data):
        """True if this NMEA fix (the raw dict from on_nmea_data - see
        osgar/drivers/gps.py's parse_nmea for the available fields) is
        trustworthy enough to learn a new compass calibration point from
        - see _update_compass_calibration and compass_calibration_min_
        quality/compass_calibration_max_hdop in __init__ for why this is
        a deliberately higher bar than what gates position/bearing_to_
        target updates elsewhere (unconditional, unchanged)."""
        quality = data.get('quality')
        if quality is not None and quality < self.compass_calibration_min_quality:
            return False
        hdop = data.get('hdop')
        if hdop is not None and hdop > self.compass_calibration_max_hdop:
            return False
        return True

    def _update_compass_calibration(self, trusted_gps_heading):
        """Learns compass_offset - the additive correction (magnetic
        declination + any mount misalignment) between the raw geometric
        compass conversion (_compass_heading) and a GPS-verified true-
        north heading - every time a fresh, baseline-confirmed, good-
        quality travel_heading becomes available (see on_nmea_data/
        _gps_fix_quality_ok). This only ever runs while GPS heading is
        ALREADY trustworthy (open sky, moving in a reasonably straight
        line, decent fix quality) - exactly the situation the old
        GPS-differenced-only design already worked fine in. Once
        learned, compass_offset lets _current_heading() keep steering by
        compass alone through a subsequent patch of unreliable GPS (tree
        canopy) that would otherwise send travel_heading - and with it
        the old design's only heading source - into noise, without ever
        needing a hardcoded declination value for wherever Matty happens
        to be operating.

        Smoothed (circular EMA, compass_offset_smoothing_alpha) rather
        than snapped to each new estimate, same spirit as
        _smooth_heading's own reasoning for travel_heading itself - a
        single momentarily-so-so fix shouldn't be able to yank an
        otherwise-good calibration around. Snaps directly to the first
        estimate ever seen (nothing to blend against yet), same pattern
        as obstdet3d_zones.py's pitch EMA startup snap. Deliberately
        never resets/expires - magnetic declination is a fixed property
        of location and mount misalignment does not drift on its own, so
        a value learned once stays valid indefinitely, which is exactly
        what lets the robot stay unaffected once travel_heading itself
        goes stale under canopy."""
        raw_compass = self._compass_heading()
        if raw_compass is None:
            return
        target_offset = normalize_angle(trusted_gps_heading - raw_compass)
        if self.compass_offset is None:
            self.compass_offset = target_offset
            print(self.time, 'compass calibrated for the first time: offset %.1f deg '
                              '(magnetic declination + mount misalignment)' % math.degrees(target_offset))
        else:
            self.compass_offset = self._blend_angle(
                self.compass_offset, target_offset, self.compass_offset_smoothing_alpha)
        self.compass_last_calibrated = self.time

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
                    if self.use_compass_heading and self._gps_fix_quality_ok(data):
                        # right when travel_heading is at its most
                        # trustworthy (baseline-confirmed, quality-gated)
                        # - see _update_compass_calibration
                        self._update_compass_calibration(self.travel_heading)
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
        # sides get their own (normally smaller) thresholds - see
        # side_stop_dist_factor/side_turning_dist_factor in __init__.
        # Both default to 1.0, i.e. identical to the single-threshold
        # behaviour this replaced.
        side_stop_dist = self.stop_dist * self.side_stop_dist_factor
        side_turning_dist = self.turning_dist * self.side_turning_dist_factor
        stop_now = center < self.stop_dist or l_dist < side_stop_dist or r_dist < side_stop_dist
        self.stop_streak = self.stop_streak + 1 if stop_now else 0

        # The robot is only "clear" if the center AND both sides are further
        # than their respective thresholds - except that a side-only
        # blockage is ignored while the way ahead is clearly open, which is
        # what a doorway/gap looks like (see
        # side_trigger_center_clear_factor in __init__)
        center_blocked = center < self.turning_dist
        side_blocked = (l_dist < side_turning_dist) or (r_dist < side_turning_dist)
        if side_blocked and self.side_trigger_center_clear_factor > 0:
            if center > self.turning_dist * self.side_trigger_center_clear_factor:
                side_blocked = False
        is_blocked = center_blocked or side_blocked

        self.turn_streak = self.turn_streak + 1 if is_blocked else 0
        # see center_blocked_streak in __init__ - deliberately centre only
        self.center_blocked_streak = self.center_blocked_streak + 1 if center < self.turning_dist else 0

    def on_depth_profile(self, data):
        self.depth_profile = data
        # evaluated here rather than in on_obstacle_zones because
        # ObstacleDetector3DZones publishes obstacle_zones first and
        # depth_profile second for the same frame - by this point both
        # halves of the test below are from the same depth image
        self._update_blind_state()

    def _update_blind_state(self):
        """Is the depth camera returning anything usable at all? See the
        blind-sensor hold notes in __init__.

        Two independent signals, both required:
          - the centre zone reporting exactly its fail_value (0.0). That
            is the sentinel _dist() returns when a window has less than
            min_valid_frac usable pixels; a real measurement can never be
            0.0, since any nonzero depth in mm is a positive number of
            metres once divided.
          - most of depth_profile unknown, i.e. the blankness is across
            the view rather than confined to the centre window.

        Debounced both ways (blind_confirm_frames / blind_clear_frames) so
        neither a single dropped frame freezes the robot nor a single
        lucky frame releases it."""
        if not self.blind_hold_enabled:
            return
        center_blank = self.last_obstacle <= 0.0
        if self.depth_profile:
            valid_frac = sum(1 for d in self.depth_profile if d is not None) / len(self.depth_profile)
        else:
            valid_frac = 0.0
        blind_now = center_blank and valid_frac < self.blind_profile_valid_frac

        if blind_now:
            self.blind_streak += 1
            self.blind_clear_streak = 0
        else:
            self.blind_clear_streak += 1
            self.blind_streak = 0

        if not self.depth_blind_active and self.blind_streak >= self.blind_confirm_frames:
            self.depth_blind_active = True
            print(self.time, 'depth camera is blind (centre invalid, %.0f%% of the profile unknown) - '
                              'holding still until it recovers' % (100 * (1 - valid_frac)))
        elif self.depth_blind_active and self.blind_clear_streak >= self.blind_clear_frames:
            self.depth_blind_active = False
            print(self.time, 'depth data recovered (%.0f%% of the profile valid) - resuming'
                   % (100 * valid_frac))

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
        physically clearer and still shows road.

        Once this avoidance episode has already committed to a direction
        (_turn_sign_committed), a re-entry into TURNING within the SAME
        episode - typically after an emergency-abort-to-backup mid-turn -
        is biased to KEEP that direction instead of re-deciding from
        scratch. left_dist/right_dist against an irregular surface (a
        wire-mesh gabion wall backed by loose rock is a good example -
        a different noise source than the glossy-floor specular dropout,
        but the same class of problem: which of several partially-valid
        readings a frame happens to land on can genuinely shift a few
        tens of cm with nothing real having changed) can flip which side
        looks clearer from one frame to the next. Re-picking freely on
        every re-entry let that noise flip the chosen side mid-maneuver,
        sending the robot back toward the very obstacle it had just
        started turning away from. Now switching sides once committed
        needs the OTHER side to be clearer by at least
        turn_sign_switch_margin_m, not just barely-inside-vs-barely-
        outside turning_dist.

        Fresh-episode picks additionally feed a same-direction-repeat
        counter (see the end of this method) - a second, position-
        independent way to notice "we're looping" alongside the older
        XY-progress check in _start_avoidance_cycle. That older check can
        be fooled: it only asks "did the robot move escape_progress_dist_m
        since the last avoidance cycle started", and ordinary DRIVE-state
        wandering (_edge_bin_correction can now swing up to avoid_steering
        for an urgent edge call) can rack up that much displacement while
        going nowhere useful, resetting escape_counter and masking a
        genuine dead end. Directly noticing "we just picked the same side
        again" sidesteps that ambiguity entirely - it doesn't care how far
        anything moved, only whether the decision repeated."""
        left = self.left_dist if self.left_dist is not None else float('inf')
        right = self.right_dist if self.right_dist is not None else float('inf')

        if self._turn_sign_committed:
            current_side = left if self.turn_sign > 0 else right
            other_side = right if self.turn_sign > 0 else left
            if other_side > current_side + self.turn_sign_switch_margin_m:
                return -self.turn_sign
            return self.turn_sign
        self._turn_sign_committed = True

        left_clear = left > self.turning_dist
        right_clear = right > self.turning_dist
        if left_clear and not right_clear:
            pick = 1
        elif right_clear and not left_clear:
            pick = -1
        # both clear, or both blocked by depth alone - break the tie
        # using whichever side still looks more like road
        elif abs(self.left_road_frac - self.right_road_frac) > 0.02:
            pick = 1 if self.left_road_frac > self.right_road_frac else -1
        else:
            # still tied - fall back to raw clearance, default left
            pick = 1 if left >= right else -1
        pick = self._bin_override_turn_sign(pick)

        if pick == self.last_turn_sign_choice:
            self.same_direction_repeats += 1
        else:
            self.same_direction_repeats = 0
        self.last_turn_sign_choice = pick

        if self.same_direction_repeats >= self.max_same_direction_repeats:
            # picked the same side max_same_direction_repeats+1 times in a
            # row across fresh episodes - treat as stuck NOW rather than
            # waiting for the (foolable) XY-progress check to notice.
            # escape_counter is bumped (not just in_escape_mode set
            # directly) so _start_avoidance_cycle's own recomputation of
            # in_escape_mode next cycle stays consistent with it - one
            # source of truth, two ways to raise it. Flipping pick here
            # applies the escape response to THIS cycle immediately
            # (in_escape_mode wasn't True yet when this method's caller
            # branched into calling it, so _enter_turning's own "if
            # in_escape_mode: flip" branch didn't fire this time) -
            # further cycles will go through that branch directly instead
            # of back through this method, same as any other escape entry.
            print(self.time, 'picked the same avoidance direction %d times in a row - '
                              'treating as stuck, forcing escape mode' % (self.same_direction_repeats + 1))
            self.escape_counter = max(self.escape_counter, self.escape_after_cycles)
            self.in_escape_mode = True
            self.same_direction_repeats = 0
            pick = -pick
            self.last_turn_sign_choice = pick
        return pick

    def _bin_open_frac(self, side_sign):
        """Fraction of depth_profile's side_sign half (>0 -> left half of
        the bins, <0 -> right half) currently reading open (beyond
        free_space_min_dist). Shared by _bin_override_turn_sign (below)
        and _avoidance_bin_scale. 1.0 (fully open) with no profile yet -
        missing data should never manufacture LESS confidence than
        whatever already-validated signal (L/R zones, severity) is using
        it as a modifier on top of."""
        if not self.depth_profile:
            return 1.0
        free_space_min_dist = self.turning_dist * self.free_space_lead_margin
        n = len(self.depth_profile)
        half = n // 2
        if half == 0:
            return 1.0
        side_bins = self.depth_profile[:half] if side_sign > 0 else self.depth_profile[-half:]
        return sum(1 for d in side_bins if d is not None and d > free_space_min_dist) / len(side_bins)

    def _depth_confidence(self):
        """Fraction of depth_profile bins currently reading a real value
        (not None/unknown) - a rough, cheap proxy for "how much should
        the depth-based steering component be trusted THIS frame".
        Field-caught failure mode: direct sunlight on a glossy surface
        can overwhelm the depth camera's IR pattern badly enough that
        MOST or ALL bins go unknown at once - not the usual one-bin
        dropout _edge_bin_correction/_free_space_steering already handle
        gracefully on their own, but the whole picture going dark
        together. Used by _drive_steering to scale DOWN how much weight
        the depth component gets in the road-following blend when this
        happens, so a widespread depth blackout doesn't just silently
        stop contributing useful information - it actively hands more of
        the steering decision to last_dir (the RGB-based road mask),
        which isn't affected by the same failure (sunlight overwhelming
        an IR emitter doesn't blind a passive RGB camera the same way).
        Deliberately NOT used to relax the hard-stop safety net
        (stop_dist/turning_dist) - road-extraction can say "this looks
        like drivable ground" but has no notion of actual distance to a
        solid obstacle, so it must never substitute for depth there; a
        widespread dropout should make the CENTER zone read its own
        fail_value=0.0 ("assume worst") regardless of what this returns.
        1.0 (fully confident) with no profile yet, same "missing data
        must not manufacture LESS confidence than what's already there"
        bias used elsewhere."""
        if not self.depth_profile:
            return 1.0
        return sum(1 for d in self.depth_profile if d is not None) / len(self.depth_profile)

    def _bin_override_turn_sign(self, pick):
        """The L/R-zone-and-road-frac pick above only ever looks as far
        as the L/C/R zones' own narrow column window - it has no way to
        tell "clearly open right in front of the robot" from "opens up
        for a meter then dead-ends", which is exactly how a real incident
        happened (see item 6/gabion-wall notes at top of file): committed
        one way based on a locally-clear reading, only to run out of room
        moments later with no space left to back out of. depth_profile's
        wider, finer-grained view is a cheap independent check on that
        BEFORE committing: if the picked side's bins are open in a much
        SMALLER fraction than the other side's (by more than
        bin_override_margin), override to the other side instead.

        Deliberately runs only ONCE per avoidance episode, right here at
        first commitment - re-entries within the same episode still go
        through the turn_sign_committed hysteresis above, unaffected by
        this, so this cannot reintroduce the mid-episode flip-flopping
        that hysteresis was added to fix. Also does not run in escape
        mode (_enter_turning flips turn_sign directly there, without
        calling this method at all) - escape mode's whole point is
        "deliberately try the other side because the zone-preferred one
        already failed repeatedly"; letting bins override that back would
        undermine exactly what escape mode exists to do.

        This is a snapshot check, not a guarantee - it cannot see a
        corridor that curves out of bin range or a dead end that only
        reveals itself mid-turn, same limitation as every other
        depth-based heuristic in this file. It reduces the "committed
        into a dead end" failure mode, it doesn't eliminate it."""
        if not self.depth_profile:
            return pick
        picked_frac = self._bin_open_frac(pick)
        other_frac = self._bin_open_frac(-pick)
        if other_frac > picked_frac + self.bin_override_margin:
            print(self.time, 'bins show the zone-preferred side is comparatively closed (open frac %.2f vs %.2f) '
                              '- turning the other way instead' % (picked_frac, other_frac))
            return -pick
        return pick

    def _best_scan_heading(self):
        """Pick the best-scoring heading recorded in scan_samples during
        the just-finished TURNING sweep. Score is center clearance plus
        side_zone_weight times whichever of left/right is smaller at that
        heading (both zones are field-calibrated now, so worth actually
        weighing in), plus - if a GPS target is set and a heading
        conversion is available - a small pull from bearing_penalty_weight
        toward headings closer to bearing_to_target (converted into the
        same compass convention via _heading_to_bearing first - calibrated
        ESP32 compass if known, else the older per-episode
        heading_frame_offset snapshot - see item 15 at top of file and
        _start_avoidance_cycle), only enough to break ties between
        comparable gaps, not to override a clearer one - and, same as
        _drive_steering's bearing term, faded by _bearing_distance_scale()
        as target_dist shrinks (item 16), since a noisy near-target
        bearing is just as poor a tie-breaker here as it is a steering
        signal there. Also penalizes
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
        sweep produced no samples at all.

        Once genuinely stuck - deep_stuck_cycles counts how many
        no-progress avoidance cycles have happened PAST the point escape
        mode already triggered (see _start_avoidance_cycle/escape_counter)
        - this stops being pure greedy argmax and starts occasionally
        accepting a locally-worse-scoring candidate instead of always the
        single best one, with growing probability the longer nothing has
        worked (deep_stuck_cycle_span cycles past the trigger to reach
        deep_stuck_max_explore_frac). Rationale: failed_headings' penalty
        above can only ever REORDER candidates this sweep already found -
        it cannot make the search consider an option it would otherwise
        never try, so a pure greedy re-optimizer can keep reconverging on
        the same locally-best-but-actually-bad choice indefinitely in a
        genuine local minimum. This is only ever a choice among
        deep_stuck_candidate_pool of the TOP-scoring candidates from THIS
        sweep though, not a random heading pulled from nowhere - every
        candidate already passed the exact same stop_dist/turning_dist/
        is_emergency safety checks as the single best one would have,
        "suboptimal" here means "not the highest score by this heuristic",
        never "unsafe". Not calibrated - watch for it happening too
        often (thrashing) or not often enough (still looping) in the
        field and adjust the span/frac from there."""
        if not self.scan_samples:
            return self.saved_heading

        # item 16 - same near-target fade _drive_steering applies to its
        # own bearing term; computed once here rather than per-sample
        # since it doesn't depend on the sample being scored
        bearing_scale = self._bearing_distance_scale()

        def score(sample):
            heading, center, left, right = sample
            sides = [d for d in (left, right) if d is not None]
            side_component = min(sides) if sides else 0.0
            s = center + self.side_zone_weight * side_component
            if self.bearing_to_target is not None:
                # convert this sample's IMU heading into the same
                # (compass) convention as bearing_to_target before
                # comparing - calibrated compass conversion if known,
                # else the older per-episode heading_frame_offset
                # fallback - see _heading_to_bearing/item 15
                heading_as_bearing = self._heading_to_bearing(heading)
                if heading_as_bearing is not None:
                    angle_diff = abs(normalize_angle(heading_as_bearing - self.bearing_to_target))
                    s -= self.bearing_penalty_weight * bearing_scale * angle_diff
            for failed in self.failed_headings:
                diff = abs(normalize_angle(heading - failed))
                if diff < self.failed_heading_tolerance:
                    s -= self.failed_heading_penalty_weight * (1 - diff / self.failed_heading_tolerance)
            return s

        ranked = sorted(self.scan_samples, key=score, reverse=True)
        deep_stuck_cycles = max(0, self.escape_counter - self.escape_after_cycles)
        if deep_stuck_cycles > 0 and len(ranked) > 1:
            explore_frac = min(self.deep_stuck_max_explore_frac,
                                deep_stuck_cycles / self.deep_stuck_cycle_span * self.deep_stuck_max_explore_frac)
            if random.random() < explore_frac:
                pool = ranked[:max(1, min(len(ranked), self.deep_stuck_candidate_pool))]
                heading, *_ = random.choice(pool)
                print(self.time, 'deeply stuck (%d cycles past escape trigger, %.0f%% explore chance) - '
                                  'trying a deliberately suboptimal heading:' % (deep_stuck_cycles, explore_frac * 100),
                      round(math.degrees(heading)))
                return heading

        best_heading, *_ = ranked[0]
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
        is for).

        Per-bin weight is capped at free_space_min_dist (i.e. saturates
        once a bin is at least twice as far as the "counts as open"
        threshold) instead of growing without bound with distance. Field
        footage showed the uncapped version chasing a single narrow bin
        that happened to see far past/through a tight gap (e.g. a sliver
        of visibility between a parked car and a gabion wall) hard enough
        to outweigh several other, more moderately-open bins combined -
        aiming the robot straight at geometry it couldn't actually fit
        through, well before turning_dist/turn_streak ever got a say.
        Capping keeps the pull roughly proportional to HOW MANY bins are
        genuinely open rather than letting the single farthest one decide
        alone - not a full fix for "far sliver looks like a way through"
        (that needs knowing the gap is too narrow, which a 1D per-column
        distance scan can't tell by itself), but it stops that single
        outlier from dominating the vote."""
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
            weight = min(d - free_space_min_dist, free_space_min_dist)
            total_weight += weight
            weighted_pos += weight * pos
        if total_weight <= 0:
            return 0.0
        aim = weighted_pos / total_weight
        # same sign convention as on_nn_mask's last_dir: aim>0 (open bins
        # skew right) should steer right
        return -math.copysign(min(1.0, abs(aim)) * self.turn_angle, aim)

    def _edge_bin_correction(self):
        """Extra, more sensitive steering nudge from just the leftmost/
        rightmost edge_bins_watched depth_profile bins - a deliberate
        complement to _free_space_steering()'s whole-profile weighted
        average, not a replacement for it. Rationale: a problem confined
        to one very edge (about to clip something on that flank) can get
        diluted into near-invisibility by the whole-profile average if
        the center/other side are wide open - the average still looks
        "mostly fine" even while the edge is not. This bypasses that
        dilution entirely: if the edge bins on ONE side are not open
        (same free_space_min_dist threshold _free_space_steering uses)
        while the other side's edge is clear, steer away from the
        blocked side, regardless of how open the rest of the profile
        looks. If BOTH edges are blocked, this contributes NOTHING -
        deliberately deferring to _free_space_steering()'s whole-profile
        average, which already answers "which direction has the most
        open room overall" for that case; duplicating that here with a
        cruder edges-only view would just re-litigate the same question
        worse.

        Magnitude scales with how urgent the blocked edge already is -
        edge_correction_min_deg right as it first crosses
        free_space_min_dist (a shallow graze, matching the severity=0
        case in _enter_turning), ramping up to edge_correction_max_deg
        once the blocked edge is as close as stop_dist itself (as
        urgent as the hard-stop safety net allows before this even
        becomes the discrete avoidance state machine's problem instead).
        A flat correction was field-tested too weak to resolve some
        approaches on its own (still ran into turning_dist/turn_streak
        and triggered the full avoidance maneuver) - scaling it the same
        way _enter_turning's severity already scales the discrete turn
        gives a close edge call real teeth instead of a token nudge,
        while a barely-triggered one stays gentle. Note the result can
        exceed turn_angle (ordinary cruising's normal steering ceiling) -
        see _drive_steering, which clips the whole-profile component to
        turn_angle BEFORE adding this on top, then clips the total to
        avoid_steering (the same ceiling the real avoidance maneuver
        uses) - so this can get as assertive as a real avoidance turn in
        the most urgent edge case, but structurally never more."""
        if not self.depth_profile or len(self.depth_profile) < 2 * self.edge_bins_watched:
            return 0.0
        free_space_min_dist = self.turning_dist * self.free_space_lead_margin
        left_edge = self.depth_profile[:self.edge_bins_watched]
        right_edge = self.depth_profile[-self.edge_bins_watched:]

        def worst(edge):
            # None (untrusted) is treated as maximally urgent (0.0m) -
            # same "assume worst when uncertain" bias as the center
            # zone's fail_value=0.0 elsewhere in this module
            return min(0.0 if d is None else d for d in edge)

        left_worst = worst(left_edge)
        right_worst = worst(right_edge)
        left_blocked = left_worst <= free_space_min_dist
        right_blocked = right_worst <= free_space_min_dist
        if left_blocked and right_blocked:
            # Narrow passage - BOTH flanks inside the threshold. This used
            # to return 0.0 and defer entirely to _free_space_steering()'s
            # whole-profile average, on the reasoning that an edges-only
            # view can't answer "which way has more room overall". True in
            # general - but it silently disabled the one signal that keeps
            # the robot off a flank in exactly the situation where flank
            # clearance is tightest and centering matters most. Field log
            # (2026-08-20 20:23): in frames with a flank inside 0.6m this
            # method returned exactly zero 86% of the time, 186 of those
            # being this branch, while the whole-profile average offered
            # only ~3deg - and Matty brushed the door frames.
            #
            # A narrow passage is not a "which side is open" question, it
            # is a CENTERING one, and the two edge bins answer that
            # directly: steer away from whichever flank is closer, scaled
            # by how lopsided they are. Perfectly symmetric (dead centre in
            # the gap) still returns ~0, so the deferral above is preserved
            # for the case it was actually right about.
            if not self.edge_correction_center_in_gap:
                return 0.0
            imbalance = right_worst - left_worst  # >0 => more room right
            span = max(1e-6, free_space_min_dist - self.stop_dist)
            lopsided = max(0.0, min(1.0, abs(imbalance) / span))
            # deliberately NOT floored at edge_correction_min_rad - unlike
            # the one-sided case below, a near-symmetric gap genuinely
            # wants no correction, and a floor here would make the robot
            # weave down every corridor
            magnitude = lopsided * self.edge_correction_max_rad
            return math.copysign(magnitude, -imbalance) if imbalance else 0.0
        if left_blocked == right_blocked:
            return 0.0  # both clear - let the whole-profile average decide

        blocked_dist = left_worst if left_blocked else right_worst
        span = max(1e-6, free_space_min_dist - self.stop_dist)
        severity = max(0.0, min(1.0, (free_space_min_dist - blocked_dist) / span))
        magnitude = self.edge_correction_min_rad + severity * (self.edge_correction_max_rad - self.edge_correction_min_rad)
        return -magnitude if left_blocked else magnitude  # away from whichever edge is blocked

    def _polar_bin(self, world_heading):
        """Bin index for an absolute heading (last_heading convention)."""
        frac = ((world_heading + math.pi) % (2 * math.pi)) / (2 * math.pi)
        return int(frac * self.polar_memory_bins) % self.polar_memory_bins

    def _polar_bin_heading(self, idx):
        """Centre heading of a bin - inverse of _polar_bin."""
        return normalize_angle(-math.pi + (idx + 0.5) * 2 * math.pi / self.polar_memory_bins)

    def _update_polar_memory(self, xy):
        """PHASE 1 - writes polar_memory and nothing else. No caller acts on
        the result; see the polar memory notes in __init__.

        Each depth_profile bin is mapped from its position across the frame
        to an absolute heading and stored. Sign convention: bin 0 is the
        leftmost columns, i.e. the robot's left, and last_heading is
        anticlockwise-positive (0=east - see on_rotation/matty.py), so a
        feature to the LEFT sits at a LARGER heading, hence the minus on
        the camera offset below. Same convention last_dir already uses."""
        if self.polar_memory_bins <= 0 or not self.depth_profile:
            return
        n = len(self.depth_profile)
        if n == 0:
            return
        half_fov = self.polar_profile_hfov / 2
        for i, d in enumerate(self.depth_profile):
            if d is None:
                continue
            if self.polar_memory_far_fill_m and d >= self.polar_memory_far_fill_m:
                continue  # far-mask fill, not a measurement - see __init__
            frac = (i + 0.5) / n * 2 - 1        # -1 (left edge) .. +1 (right edge)
            camera_offset = frac * half_fov     # +ve = to the robot's right
            world = normalize_angle(self.last_heading - camera_offset)
            self.polar_memory[self._polar_bin(world)] = (d, self.time, xy)

    def _polar_clearance(self, idx, xy):
        """Remembered clearance for a bin, or None if never seen or stale.
        Stale means either too old or observed from too far away - see the
        polar memory notes in __init__ for why both matter."""
        entry = self.polar_memory[idx] if 0 <= idx < len(self.polar_memory) else None
        if entry is None:
            return None
        dist, when, seen_from = entry
        if when is None or self.time - when > self.polar_memory_max_age:
            return None
        if math.hypot(xy[0] - seen_from[0], xy[1] - seen_from[1]) > self.polar_memory_max_travel_m:
            return None
        return dist

    def _polar_snapshot(self, xy):
        """[(bin_heading_rad, clearance_m or None)] for every bin, newest
        known value. Diagnostics in phase 1; the intended read path for a
        later phase."""
        return [(self._polar_bin_heading(i), self._polar_clearance(i, xy))
                for i in range(self.polar_memory_bins)]

    def _log_polar_memory(self, xy):
        """Throttled one-liner: how much of the 360deg is currently
        remembered, and where the roomiest remembered direction is. Purely
        so this can be judged from a real run (and replayed offline)
        before anything is allowed to act on it."""
        if self.polar_memory_bins <= 0 or self.polar_memory_log_interval.total_seconds() <= 0:
            return
        if (self._last_polar_log_time is not None
                and self.time - self._last_polar_log_time < self.polar_memory_log_interval):
            return
        self._last_polar_log_time = self.time
        snap = self._polar_snapshot(xy)
        known = [(h, d) for h, d in snap if d is not None]
        if not known:
            print(self.time, 'polar memory: nothing remembered yet')
            return
        best_h, best_d = max(known, key=lambda hd: hd[1])
        ahead = self._polar_clearance(self._polar_bin(self.last_heading), xy)
        print(self.time, 'polar memory: %d/%d bins known, best %.2fm at %+.0fdeg '
                          '(%+.0fdeg relative), ahead=%s' % (
              len(known), self.polar_memory_bins, best_d, math.degrees(best_h),
              math.degrees(normalize_angle(best_h - self.last_heading)),
              ('%.2fm' % ahead) if ahead is not None else 'unknown'))

    def _adaptive_speed(self, dt=None):
        """Cruising speed scaled by currently sensed clearance, within
        [min_speed, max_speed] (item 10 at top of file). max_speed
        outright if adaptive_speed is off (no rate limiting either in
        that case - an explicit opt-out of the adaptive behaviour
        entirely, not just its ceiling).

        The adaptive branch is rate-limited via _rate_limit_speed - see
        that method for why: unlike _rate_limit_steering (which already
        existed), this recomputed a brand new target EVERY cycle straight
        from the latest single L/C/R zone reading, with nothing smoothing
        it - a real gap, field-caught as Matty accelerating hard right
        after an avoidance maneuver ended and crashing in a still-tight
        space before the (deliberately debounced, close_confirm_frames-
        gated) hard-stop safety net could react."""
        if not self.adaptive_speed:
            return self.max_speed
        l_dist = self.left_dist if self.left_dist is not None else float('inf')
        r_dist = self.right_dist if self.right_dist is not None else float('inf')
        # convert the flanks to "centre-equivalent" room before taking the
        # minimum - see speed_side_clearance_factor in __init__
        if 0 < self.speed_side_clearance_factor < 1.0:
            l_dist /= self.speed_side_clearance_factor
            r_dist /= self.speed_side_clearance_factor
        clearance = min(self.last_obstacle, l_dist, r_dist, self.speed_clearance_ceiling)
        frac = clearance / self.speed_clearance_ceiling if self.speed_clearance_ceiling > 0 else 1.0
        target = max(self.min_speed, min(self.max_speed, self.min_speed + frac * (self.max_speed - self.min_speed)))
        return self._rate_limit_speed(target, dt)

    def _rate_limit_speed(self, target, dt):
        """Caps how fast _adaptive_speed()'s output can change per cycle -
        the speed counterpart of _rate_limit_steering below, which already
        existed for steering but had no equivalent here. Two related gaps
        this closes, both field-observed as Matty lurching to a much
        higher speed and crashing in a cramped space shortly after an
        avoidance maneuver ended:

        1. _adaptive_speed() has no smoothing of its own - it directly
        reflects whatever the MOST RECENT single L/C/R zone reading says.
        The corresponding safety reaction (stop_streak -> hard stop) is
        deliberately debounced over close_confirm_frames specifically so
        one noisy/optimistic frame can't cause a false trigger - but
        that same single noisy frame CAN spike commanded speed instantly,
        with nothing to prevent it and up to close_confirm_frames worth
        of confirmed-bad readings still needed before the hard stop
        catches up. Rate-limiting speed the same way steering already is
        removes that asymmetry.

        2. Right as an avoidance maneuver ends, the chassis has often
        just turned toward a very different heading - the zone reading
        CAN swing from "tight, near min_speed" to "wide open, near
        max_speed" in a single cycle for real, not noise. Still risky to
        accelerate into instantly with the least margin for anything
        unexpected right as a maneuver completes - and _update_adaptive_
        distances() computes THIS cycle's stop_dist/turning_dist from
        LAST cycle's _last_commanded_speed (deliberately - see that
        method), so a sudden speed jump also means the safety margin
        briefly lags a cycle behind the speed actually being commanded.
        Smoothing the speed keeps consecutive commanded speeds close
        together, keeping that one-cycle lag inconsequential instead of
        compounding with it.

        Deliberately NOT applied to the avoidance state machine's own
        fixed speeds (avoid_speed, backup_speed) - same reasoning
        _rate_limit_steering already documents for steering: those need
        to commit decisively the instant they're triggered, not ease
        into it. Baseline is _last_commanded_speed, updated once per
        cycle in on_pose2d regardless of which state produced it, so a
        transition INTO the adaptive branches (DRIVE/REALIGNING) ramps
        from whatever speed was actually just being commanded (e.g.
        avoid_speed or -backup_speed), not a stale value.
        max_speed_accel_mps2<=0 disables this outright, same as dt=None."""
        if self.max_speed_accel is None or dt is None:
            return target
        max_delta = self.max_speed_accel * dt.total_seconds()
        delta = max(-max_delta, min(max_delta, target - self._last_commanded_speed))
        return self._last_commanded_speed + delta

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

    def _current_heading(self):
        """Best available current-heading estimate for GPS bearing-
        following (_drive_steering) and scan-heading scoring (via
        _heading_to_bearing, used by _best_scan_heading), together with
        a short string identifying the source (diagnostics only - see
        _gps_status_line/_following_status). Priority:

          1. Compass (ESP32 magnetometer-fused yaw, converted +
             calibrated - see _compass_heading/_update_compass_
             calibration), if use_compass_heading is on AND a
             calibration is already known. Updated every pose2d cycle
             (100ms) straight from the IMU, entirely independent of GPS
             fix quality/availability - see item 15 at top of file for
             why this is what actually fixes the "circles under a tree"
             failure this method was written for.
          2. travel_heading (GPS fix differencing, unchanged from before
             item 15) - used before compass_offset has ever been
             learned (e.g. fresh boot taken straight into cover before
             any open-sky calibration lap), or with
             use_compass_heading=False.
          3. None - no usable heading yet; callers fall back to pure
             road-following (_drive_steering) or skip the bearing-nudge
             term (_best_scan_heading)."""
        if self.use_compass_heading and self.compass_offset is not None:
            raw_compass = self._compass_heading()
            if raw_compass is not None:
                return normalize_angle(raw_compass + self.compass_offset), 'compass'
        if self.travel_heading is not None:
            return self.travel_heading, 'gps-diff'
        return None, 'none'

    def _heading_to_bearing(self, imu_heading):
        """Converts an arbitrary recorded last_heading sample (OSGAR
        convention - e.g. one of _best_scan_heading's TURNING-sweep
        scan_samples) into the same compass convention as
        bearing_to_target, using whichever source _current_heading()
        would currently use for "right now": the calibrated compass
        conversion if known, else the older per-episode empirical
        heading_frame_offset snapshot (see _start_avoidance_cycle) as a
        fallback for before any compass calibration exists. Returns None
        if neither is available."""
        if self.use_compass_heading and self.compass_offset is not None:
            return normalize_angle(math.pi / 2 - self.compass_sign * imu_heading + self.compass_offset)
        if self.heading_frame_offset is not None:
            return normalize_angle(imu_heading + self.heading_frame_offset)
        return None

    def _bearing_distance_scale(self):
        """Extra multiplier (0..1) on GPS-bearing trust, on top of
        bearing_blend_road_frac's road-agreement scaling (item 16 at top of
        file) - fades toward 0 as target_dist shrinks toward
        waypoint_arrival_dist_m, instead of following bearing_to_target at
        full strength all the way to arrival.

        Rationale: bearing_to_target is recomputed from every GPS fix with
        NO smoothing at all (deliberately - see item 15's docstring), which
        is fine far from the target (a few meters of ordinary, non-RTK GPS
        position noise barely moves the bearing to something 50m away) but
        not close in - the SAME fixed position error produces a bearing
        error that grows as target_dist shrinks (roughly
        atan(position_error / target_dist)): already ~30deg at
        target_dist=10m for a few meters of noise, 45deg+ inside 5m -
        comparable to or bigger than turn_angle itself. This gets worse in
        exactly the situation item 15's compass work was written for: a
        waypoint near/under tree cover is also where GPS position noise is
        largest, so the two failure modes compound right where it matters
        most.

        roboorienteering/ro.py - this file's own cited precedent for the
        GPS-heading-following approach (see item 5) - already disabled its
        equivalent term below 20m of the target for this exact reason; that
        gate did not carry over when this file adapted the approach. This
        restores an equivalent, as a smooth fade rather than a hard cutoff
        (so it doesn't introduce its own steering discontinuity right at a
        threshold), tied to waypoint_arrival_dist_m as the near edge - no
        reason to trust bearing right as the robot is about to stop anyway,
        and one fewer independent magic number to tune.

        Returns 1.0 (no reduction) if there's no target_dist yet, or if
        bearing_near_target_dist_m <= waypoint_arrival_dist_m (feature
        disabled). Multiplies INTO the existing bearing_blend_road_frac
        weight - either one pulling weight down is enough to distrust the
        bearing term; this doesn't replace that check. Not field-tested -
        15.0 (the bearing_near_target_dist_m default) is a starting point,
        same caveat as the rest of this file's GPS code."""
        if self.target_dist is None or self.bearing_near_target_dist_m <= self.waypoint_arrival_dist_m:
            return 1.0
        span = self.bearing_near_target_dist_m - self.waypoint_arrival_dist_m
        return max(0.0, min(1.0, (self.target_dist - self.waypoint_arrival_dist_m) / span))

    def _drive_steering(self, dt=None):
        """Steering to use when NOT actively avoiding an obstacle - i.e.
        this only ever runs from a context where obstacle avoidance has
        already had first say (it is highest priority: it fully overrides
        this method's result by never calling it while avoiding). Three
        signals blend together, in order:

        1. last_dir (road mask, appearance-based "is this drivable") and
           a depth-based "is this open" component combine via
           free_space_weight - scaled down by _depth_confidence() when
           depth_profile is largely unknown this frame (a widespread
           sensor dropout, e.g. direct sunlight overwhelming the IR
           pattern on a glossy surface - field-observed - shouldn't just
           quietly stop contributing, it should hand more of the
           decision to last_dir instead) - into local_dir. Together
           these are the answer to "pick the smoothest path that's also
           marked drivable": smoothness/openness from depth, drivable-
           marking from the road mask, blended rather than either one
           alone deciding. The depth component itself is
           _free_space_steering() (whole-profile weighted average,
           clipped to +-turn_angle) plus _edge_bin_correction() (a
           sharper, severity-scaled nudge from just the leftmost/
           rightmost bins, so a problem confined to one edge doesn't get
           diluted away by an otherwise-open center - can grow past
           turn_angle for an urgent edge call), with the TOTAL then
           clipped to +-avoid_steering before blending in. This
           intentionally does NOT live in the avoidance state machine -
           it's a continuous control question ("how should ordinary
           cruising lean"), not a discrete one ("has something forced an
           unavoidable maneuver") - the state machine still owns exactly
           the latter, unchanged, and is NOT relaxed by low depth
           confidence (see _depth_confidence's docstring for why).
        2. GPS bearing still wins over local_dir whenever the mask shows
           little/no road on the side the bearing wants: bearing_blend_
           road_frac is the road-fraction (see on_nn_mask - the
           theoretical max is ~0.5 since the always-masked-out sky half
           counts toward the mean) at which the bearing gets full trust;
           below that it's scaled down proportionally, pure local_dir at
           0. The current heading used against bearing_to_target comes
           from _current_heading() (see item 15 at top of file) - the
           calibrated ESP32 compass once known, so this term stays
           usable under tree canopy where GPS fix differencing alone
           would not; falls back to travel_heading before that
           calibration exists. Also fades toward 0 as target_dist
           approaches waypoint_arrival_dist_m regardless of road_frac
           (_bearing_distance_scale, item 16) - GPS position noise turns
           into large bearing error close to the target, so trust in
           this term backs off there even when the road mask would
           otherwise hand it full weight. This is a first-pass
           heuristic, not field tuned - watch left_road_frac/
           right_road_frac against turn_streak false positives once you
           can test outside.
        3. The final result is rate-limited (_rate_limit_steering) so
           ordinary driving doesn't snap between corrections - this step
           only, never the avoidance maneuvers themselves."""
        local_dir = self.last_dir
        # scale the depth component's blend weight by how much of
        # depth_profile is actually valid right now - see
        # _depth_confidence. 1.0 (no reduction) in the ordinary case;
        # only pulls this down during a widespread sensor dropout, in
        # which case last_dir (RGB road extraction, unaffected by an IR
        # emitter being overwhelmed by sunlight) picks up the slack
        # instead of steering off of a component with nothing real left
        # to say.
        effective_free_space_weight = self.free_space_weight * self._depth_confidence()
        if effective_free_space_weight > 0:
            # whole-profile average clipped to the normal cruising ceiling
            # first, THEN the (potentially much larger, see
            # _edge_bin_correction) edge term added on top and the TOTAL
            # clipped to avoid_steering - so an urgent edge call can get
            # as assertive as a real avoidance turn, but never past it
            whole_profile = max(-self.turn_angle, min(self.turn_angle, self._free_space_steering()))
            depth_component = whole_profile + self._edge_bin_correction()
            depth_component = max(-self.avoid_steering, min(self.avoid_steering, depth_component))
            local_dir = (1 - effective_free_space_weight) * local_dir + effective_free_space_weight * depth_component

        current_heading, _source = self._current_heading()
        if not self.follow_gps_target or self.bearing_to_target is None or current_heading is None:
            return self._rate_limit_steering(local_dir, dt)  # no usable heading yet

        error = normalize_angle(current_heading - self.bearing_to_target)
        bearing_steering = max(-self.turn_angle, min(self.turn_angle, error))

        road_frac_that_way = self.left_road_frac if bearing_steering > 0 else self.right_road_frac
        if self.bearing_blend_road_frac > 0:
            weight = min(1.0, road_frac_that_way / self.bearing_blend_road_frac)
        else:
            weight = 1.0
        weight *= self._bearing_distance_scale()  # fade out approaching the target - see item 16
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
        _heading, source = self._current_heading()
        if source == 'none':
            return ('target set, waiting for compass calibration or GPS heading '
                     '(need >=%.1fm of movement)' % self.gps_heading_min_baseline_m)
        # near-target bearing fade (item 16) - only appended when it's
        # actually pulling weight down, so the common case (far from the
        # target) doesn't clutter every status line with "fade=100%"
        fade = self._bearing_distance_scale()
        fade_note = '' if fade >= 0.999 else ', bearing fading %.0f%% (near target)' % (fade * 100)
        if source == 'compass':
            if self.compass_last_calibrated is not None:
                age_sec = (self.time - self.compass_last_calibrated).total_seconds()
                return 'following (compass, offset=%.1fdeg, calibrated %.0fs ago)%s' % (
                    math.degrees(self.compass_offset), age_sec, fade_note)
            return 'following (compass)%s' % fade_note
        return 'following (GPS fix-differencing, compass not yet calibrated)%s' % fade_note

    def _gps_status_line(self, lat, lon):
        """One-line summary of everything on_qr_code/on_nmea_data know
        right now: own position, current heading (and its source - see
        _current_heading/item 15 at top of file), target and bearing/
        distance to it, what steering that would currently produce, and
        whether it's actually being applied (see _following_status)."""
        parts = ['pos=(%.6f,%.6f)' % (lat, lon)]
        current_heading, source = self._current_heading()
        if current_heading is not None:
            parts.append('heading=%.0fdeg(%s)' % (math.degrees(current_heading), source))
        else:
            parts.append('heading=unknown')
        if self.travel_heading is not None and source != 'gps-diff':
            # only worth calling out separately when it's NOT already the
            # value shown above - lets the field-calibration hand-turn
            # test (item 15) and general compass-vs-GPS sanity checking
            # compare the two without needing verbose mode
            parts.append('gps_travel_heading=%.0fdeg' % math.degrees(self.travel_heading))
        if self.compass_offset is not None:
            parts.append('compass_offset=%.1fdeg' % math.degrees(self.compass_offset))
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

        # Two field bugs fixed together here, both stemming from the same
        # root cause: this method used to redo its "which way, and start
        # a fresh sweep" decisions on EVERY call, including a re-entry
        # after an emergency-abort-to-backup WITHIN the same avoidance
        # episode - it had no notion of "already decided this episode".
        #
        # (1) Escape mode's turn_sign flip ran unconditionally every
        # re-entry. A single episode can re-enter TURNING several times
        # (abort, back up, retry...) - flipping every time means an EVEN
        # number of re-entries cancels out, landing back on the exact
        # direction that already wasn't working, while looking (from the
        # HUD/logs) like escape mode is doing something. Now the flip
        # happens ONCE per episode, at the moment escape mode is first
        # entered for it - re-entries keep whatever direction was already
        # decided, same hysteresis _choose_turn_sign() already uses for
        # its own non-escape picks.
        #
        # (2) scan_samples got reset to [] on every re-entry too - so a
        # good heading recorded early in a sweep (e.g. the very first
        # frame, still close to the pre-turn heading, if that heading
        # happened to already be open - a narrow-but-passable doorway is
        # exactly this case: center reads wide open before any turning
        # even starts) was silently discarded the instant an emergency
        # abort interrupted the sweep before _best_scan_heading() ever
        # got to score it. Repeat that a few times and the one genuinely
        # good sample never survives long enough to be picked - the robot
        # commits to whatever a series of truncated, abort-interrupted
        # partial sweeps happened to see instead, never "seeing" the
        # opening it started right next to. Now scan_samples/
        # scan_start_heading/scan_confident_streak only reset at the
        # start of a fresh episode, so they accumulate across re-entries
        # within one episode instead.
        fresh_episode_start = not self._turn_sign_committed
        if self._turn_sign_committed:
            pass  # keep self.turn_sign as already decided for this episode
        elif self.in_escape_mode:
            self.turn_sign = -self.turn_sign  # deliberately try the other side, ONCE for this episode
            self._turn_sign_committed = True
        else:
            self.turn_sign = self._choose_turn_sign()  # sets _turn_sign_committed itself

        if fresh_episode_start:
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

            def zone_severity(dist, turning, stopping):
                """0 right as `turning` is crossed, 1 once down at `stopping`."""
                span = max(1e-6, turning - stopping)
                return max(0.0, min(1.0, (turning - dist) / span))

            # Each zone is scored against ITS OWN thresholds (item 17) -
            # the sides now use side_turning_dist_factor/side_stop_dist_
            # factor, so measuring a side distance against the centre's
            # (larger) thresholds would systematically overstate how
            # urgent a flank reading is: a wall alongside at exactly the
            # side turning threshold would score as already most of the
            # way to a full-lock turn, which is precisely the
            # overcorrection item 13(a) exists to prevent.
            severity = max(
                zone_severity(self.last_obstacle, self.turning_dist, self.stop_dist),
                zone_severity(l_dist, self.turning_dist * self.side_turning_dist_factor,
                              self.stop_dist * self.side_stop_dist_factor),
                zone_severity(r_dist, self.turning_dist * self.side_turning_dist_factor,
                              self.stop_dist * self.side_stop_dist_factor))
        self.current_avoid_steering = self.min_avoid_steering + severity * (self.avoid_steering - self.min_avoid_steering)
        self.current_scan_min_sweep = self.min_scan_sweep + severity * (self.scan_min_sweep - self.min_scan_sweep)

        # bin-based feasibility cap: severity above answers "how urgent is
        # the trigger", a different question from "is there actually room
        # to swing turn_sign's way that far". If depth_profile's bins on
        # the side we're about to turn toward are NOT mostly open, cap the
        # amplitude down toward min_avoid_steering instead of committing
        # to the full severity-computed swing - avoids clipping something
        # on that flank that wasn't the primary L/C/R trigger. This can
        # only ever REDUCE current_avoid_steering, never increase it past
        # what severity already set - same "more cautious when uncertain,
        # never less" bias as the rest of this module. Not calibrated -
        # avoidance_bin_scale_floor is a starting point.
        #
        # NOT applied in escape mode - a real regression caught in the
        # field: this cap used to run unconditionally, silently cutting
        # escape mode's severity=1.0 full-commitment turn down to as
        # little as avoidance_bin_scale_floor whenever the bins (quite
        # plausibly, since escape mode by definition means milder
        # responses already weren't resolving this spot - a genuine
        # dead end reads as "not open" on BOTH sides, not just one)
        # showed the chosen side wasn't wide open. That directly undid
        # the "escape mode always gets full severity" guarantee two
        # paragraphs up, which is the one thing this module explicitly
        # promised never to compromise on once escape mode is reached.
        if not self.in_escape_mode:
            self.current_avoid_steering = max(
                self.min_avoid_steering, self.current_avoid_steering * self._avoidance_bin_scale())

    def _avoidance_bin_scale(self):
        """Fraction (avoidance_bin_scale_floor..1.0) of the severity-based
        current_avoid_steering to actually use, based on _bin_open_frac
        for the side turn_sign is about to swing toward. 1.0 = that side
        reads fully open, no reduction; avoidance_bin_scale_floor = that
        side reads entirely blocked/unknown."""
        open_frac = self._bin_open_frac(self.turn_sign)
        return self.avoidance_bin_scale_floor + (1 - self.avoidance_bin_scale_floor) * open_frac

    def _enter_drive(self):
        self.state = State.DRIVE
        self.saved_heading = None
        # a maneuver that ran to completion is not part of an abort loop
        # (see max_maneuver_aborts) - only consecutive aborts count
        self.maneuver_abort_streak = 0

    def _start_avoidance_cycle(self, xy):
        """Called once per fresh avoidance cycle (not on the re-entrant/
        aborted-mid-maneuver path). Tracks whether the robot is making
        real progress between cycles; after escape_after_cycles cycles
        with less than escape_progress_dist_m of net movement, switches
        into escape mode."""
        if self.saved_heading is not None:
            return  # already mid-cycle
        self.saved_heading = self.last_heading
        self._turn_sign_committed = False  # fresh episode - see _choose_turn_sign
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
                self.same_direction_repeats = 0  # actually moved - not a repeat of a stuck pattern
                self.maneuver_abort_streak = 0  # actually moved - past aborts no longer relevant
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

        # PHASE 1 bookkeeping (item 21) - deliberately before every early
        # return below, so the memory keeps building while the robot is
        # held stopped too. Writes polar_memory and prints; no decision in
        # this method or any other reads it.
        self._update_polar_memory(xy)
        self._log_polar_memory(xy)

        if not self.have_obstacle_data:
            # camera pipeline still booting (OAK-D Pro typically takes a
            # few seconds) - pose2d already flows from the platform at
            # this point, but last_obstacle/left_dist/right_dist are still
            # unset "assume clear" defaults, not a confirmed clear path.
            # Stay stopped rather than drive blind.
            self.send_speed_cmd(0, 0)
            return

        if self.depth_blind_active:
            # the camera is returning essentially nothing - see
            # _update_blind_state. Hold still: do NOT hand this to the
            # avoidance state machine, which would read the centre's 0.0
            # fail-safe as an obstacle and answer by reversing, i.e. by
            # moving in the one direction nothing watches at all. Placed
            # ahead of every other hold because it is the most fundamental
            # of them: the others act on what was sensed, this one applies
            # when nothing was.
            self._last_commanded_speed = 0.0
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

        # Helper flags for clearance. Judged against the SIDE threshold
        # (item 18d) - these decide "is there room on a flank to turn into
        # yet", which is the same question side_turning_dist_factor exists
        # to answer, so using the centre distance here left a dead band:
        # a flank between side_turning_dist and turning_dist counted as
        # neither blocked (so nothing re-triggered avoidance) nor clear
        # (so BACKING_UP would not release), and the robot kept reversing
        # until it hit max_backup_time. Measured in the 2026-08-20 20:25
        # U-shaped-area run: 122s of 284s spent BACKING_UP, averaging 2.1s
        # per backup against a 1.5s minimum.
        side_turning_dist = self.turning_dist * self.side_turning_dist_factor
        left_clear = self.left_dist is None or self.left_dist > side_turning_dist
        right_clear = self.right_dist is None or self.right_dist > side_turning_dist
        any_side_clear = left_clear or right_clear

        # CENTRALIZED SAFETY CHECK: If an emergency stop occurs during an active maneuver,
        # instantly abort the current state and switch to backing up.
        is_emergency = self.avoid_obstacles and self.stop_streak >= self.stop_confirm_frames
        if is_emergency and self.state in (State.TURNING, State.REALIGNING):
            print(self.time, 'Emergency stop during maneuver! Aborting to backup.')
            # This abort path bypasses BOTH stuck detectors - see
            # max_maneuver_aborts in __init__. Count the aborts here so a
            # TURNING -> abort -> BACKING_UP -> TURNING loop is at least
            # visible to something, and escalate once it has clearly failed
            # repeatedly rather than letting it repeat until the battery or
            # the operator ends it.
            self.maneuver_abort_streak += 1
            if self.maneuver_abort_streak >= self.max_maneuver_aborts:
                print(self.time, 'maneuver aborted %d times in this episode with no progress - '
                                  'forcing escape mode and re-deciding direction'
                       % self.maneuver_abort_streak)
                # Same "one source of truth, two ways to raise it" pattern
                # _choose_turn_sign already uses for its own repeat detector.
                self.escape_counter = max(self.escape_counter, self.escape_after_cycles)
                self.in_escape_mode = True
                # Clearing this is what actually lets the next _enter_turning
                # change anything: while it stays True, _enter_turning keeps
                # the already-committed turn_sign and escape mode's own flip
                # never runs, so every retry repeats the identical maneuver.
                # Deliberately gated behind max_maneuver_aborts rather than
                # done on every abort - the hysteresis it suspends exists to
                # stop frame-to-frame depth noise flipping the direction
                # mid-maneuver (see _choose_turn_sign), which is a different
                # situation from N consecutive confirmed failures.
                self._turn_sign_committed = False
                self.maneuver_abort_streak = 0
            # If we were turning, keep the steering angled to swing away in
            # the SAME direction (turn_sign) and amplitude
            # (current_avoid_steering) TURNING was actually using - not the
            # bare unsigned avoid_steering ceiling, which always points the
            # same physical way regardless of which side this cycle was
            # turning toward. Using the unsigned constant here meant that
            # whenever turn_sign was -1 (turning right), an abort mid-turn
            # would still back up angled as if turn_sign were +1 (left) -
            # swinging the chassis back toward whatever it had just started
            # turning away from instead of away from it. Otherwise (aborting
            # while REALIGNING, not actively turning) back straight.
            steer = self.turn_sign * self.current_avoid_steering if self.state == State.TURNING else 0
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
            # live, not cached - see scan_confident_margin in __init__ for
            # why a value frozen at boot would drift out of sync with
            # turning_dist once adaptive_distances is scaling it by speed
            scan_confident_dist = self.turning_dist * self.scan_confident_margin
            confident_clear = (self.last_obstacle >= scan_confident_dist
                                and l_dist >= scan_confident_dist
                                and r_dist >= scan_confident_dist)
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
            if self.turn_streak >= self.stop_confirm_frames:
                # The heading TURNING committed to has since become blocked
                # STRAIGHT AHEAD - REALIGNING drives forward toward it for up
                # to max_realign_time, and otherwise only the harder
                # is_emergency/stop_streak abort above could interrupt that.
                #
                # History worth keeping straight (items 17c/18a/19): this was
                # first gated on turn_streak, which fired continuously in a
                # corridor because turn_streak also confirms on a side zone -
                # REALIGNING then lasted a single 0.1s cycle every time and a
                # doorway became unpassable. It was narrowed to the centre
                # zone only, which fixed that but left REALIGNING with NO
                # awareness of its flanks at all - and REALIGNING is the one
                # forward-driving state that ignores the depth-based steering
                # blend entirely, holding up to realign_max_steering toward a
                # heading no matter what is beside it. Field log (2026-08-20
                # 20:42): locked at +30deg with the left flank at 0.54m and
                # accelerating, straight into a frontal collision.
                #
                # Back on turn_streak now, which is safe because
                # side_trigger_center_clear_factor (item 18c) already keeps a
                # side-only blockage from confirming while the way ahead is
                # open - i.e. the doorway that made this wrong the first time
                # no longer sets turn_streak at all. Uses stop_confirm_frames
                # rather than close_confirm_frames: abandoning a realign is
                # cheap and recoverable (DRIVE re-decides next cycle with the
                # full depth blend), so it should not wait as long as
                # committing to a turn does.
                #
                # Handing control back to DRIVE rather than re-entering
                # TURNING directly is deliberate: the very next cycle the
                # normal turn_streak branch below picks this up and runs the
                # ordinary, already-field-tested avoidance entry (including
                # _start_avoidance_cycle's progress/escape bookkeeping), so
                # this adds an exit rather than a new maneuver path.
                print(self.time, 'realign target blocked (%s), handing back to avoidance'
                       % ('ahead' if self.center_blocked_streak >= self.stop_confirm_frames else 'flank'))
                self._enter_drive()
                speed, steering_angle = self._adaptive_speed(dt), self._drive_steering(dt)
            elif abs(error) < self.realign_tolerance or elapsed > self.max_realign_time:
                print(self.time, 'realigned, resuming road following')
                self._enter_drive()
                speed, steering_angle = self._adaptive_speed(dt), self._drive_steering(dt)
            else:
                steering_angle = max(-self.realign_max_steering,
                                     min(self.realign_max_steering, error * self.realign_gain))
                speed = self._adaptive_speed(dt)

        elif is_emergency or (not self.avoid_obstacles and self.stop_streak >= self.stop_confirm_frames):
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
            speed, steering_angle = self._adaptive_speed(dt), self._drive_steering(dt)

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
