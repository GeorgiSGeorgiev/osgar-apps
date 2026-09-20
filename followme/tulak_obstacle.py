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

  27. Blindness detection no longer keys off the centre reading. It used
      to require "centre == 0.0", the sentinel an untrusted window
      reported - but obstdet3d_zones' center_fail_dist_m makes that value
      configurable, so raising it for a competition run (accepting that
      missing data no longer stops the robot) would ALSO have silently
      disabled blindness detection, and with it the blind hold and creep.
      Those two decisions must stay independent: one is "how much risk am
      I taking when the camera cannot see", the other is "can the camera
      see at all". _update_blind_state now uses depth_profile validity
      plus the ground band only, both of which are unaffected by the fail
      value. Same detections on the logs either way.

  26. Three fixes from the 2026-08-29 Stromovka analysis:

      (a) Course over ground from the receiver (use_gps_course). The GPS
      had been sending RMC once a second all along, carrying its own
      course, and osgar discarded every one - split_buffer only ever
      searched for GGA, so RMC never even left the serial buffer. Now
      parsed and merged into nmea_data as cog/sog. It beats fix
      differencing on steadiness roughly 3:1 (median change between
      consecutive 1Hz samples 2.83deg vs 8.43deg) and needs no baseline
      distance, no straight-line assumption and no reverse gate, since it
      is an instantaneous direction of travel rather than a chord averaged
      over whatever the robot did between two fixes. Used as a heading
      source below the compass, and preferred over travel_heading for
      teaching compass_offset. Only available while moving - 40% of logged
      fixes carried a usable cog - so travel_heading stays as the fallback.

      (b) min_real_frac in obstdet3d_zones - see there. Stops the far-mask
      supplying a confident distance for a window holding no measurement
      at all, which is what let run 125548 read 15.00m into a frontal
      collision. Measured on the logs: false-far centre readings 0.8% ->
      0.2% on that run, with the fail-safe rate rising 0.0-0.6pp and
      not moving at all on a healthy run.

      (c) The ground band now vetoes the blind hold
      (blind_ground_valid_frac). The centre and profile windows sit near
      the horizon and legitimately go blank facing open sky, which is not
      blindness; the ground band a metre ahead is present in any drivable
      scene. Run 130657 spent 14s creeping "blind" while the ground band
      read 60-62% valid the whole time. With the veto that run drops from
      2.3% blind in 9 episodes to 0.7% in 1, while the genuinely blind
      sunny-bridge run (ground band also at 0.0%) is unchanged at ~74%.

  25. Compass can now run without GPS correction at all
      (compass_offset_deg / compass_learn_offset - see __init__).
      compass_offset corrects declination plus a fixed mount rotation;
      both are constants, so learning them at runtime was only ever a way
      to avoid having to know them. Setting compass_offset_deg seeds the
      value at boot - the compass is then usable from the first cycle
      instead of after the ~36-58s needed to accumulate a straight,
      forward, quality-passing 5m GPS leg - and compass_learn_offset=False
      stops GPS refining it thereafter. GPS still supplies the target and
      the distance to it; it just stops being in the heading loop.

      +6.1deg is the measured starting point (constant term of the fit
      over 48 straight forward legs from the 2026-08-07/08-23 logs).
      Note the same fit found 5.6deg of error varying WITH heading, which
      is hard-iron distortion and cannot be removed by any single number
      here - that needs a magnetometer ellipsoid calibration on the ESP32.

  24. Blind behaviour is now a dial, not a switch (blind_creep_speed /
      blind_creep_max_dist_m - see __init__). Item 22 always stopped, and
      standing still is not automatically the safest option in context: a
      competition run that has to reach a waypoint cannot spend 3-18% of
      itself parked, which is what the 2026-08-23 runs measured. Setting
      blind_creep_speed > 0 keeps Matty moving forward at that speed
      instead, steered by whatever still works - _drive_steering degrades
      to the RGB road mask plus GPS bearing on its own, since
      _depth_confidence() reads 0 with an all-unknown profile and drops
      the depth term out of the blend. The road mask really does keep
      working through a stereo blackout: 0.20-0.22 drivable fraction
      during blackouts against 0.23-0.31 with healthy depth, on the same
      logs.

      What makes a blind crawl defensible is speed and nothing else. The
      2026-08-23 leg impact happened at 0.50m/s; the same contact at
      0.15m/s is a bumper tap. Keep blind_creep_speed at or below
      min_speed - above that it stops being a mitigation.

      Two bounds come with it. Creeping only happens from DRIVE, because
      going blind partway through an avoidance maneuver means something
      was already known to be there. And blind_creep_max_dist_m caps how
      far one blind stretch may travel; 0 means unlimited, which is both
      what "just keep going" asks for and what will eventually drive into
      water given a long enough blackout, so set a real budget on any
      route near a drop.

  23. Compass calibration was learning from curved legs. compass_offset is
      physically a near-constant (magnetic declination + fixed mount
      error), but the 2026-08-23 outdoor runs had it swinging +13.6..+67.8
      deg inside one run and -1.0..+21.0 deg in another - and NOT because
      of poor GPS: those runs were DGPS quality 2, 12 satellites, hdop
      0.55-1.08, 100% fix rate.

      The offset is learned from travel_heading, the bearing between two
      fixes gps_heading_min_baseline_m apart. That equals the robot's
      heading only if it drove roughly straight between them - and under
      continuous obstacle avoidance it does not, it curves, so the chord
      bearing differs from anywhere the robot was ever pointing. The
      existing quality/hdop gates are blind to this: the fix can be
      flawless while the path is an arc.

      _baseline_was_straight now requires the heading to have stayed within
      compass_calibration_max_heading_spread_deg across the whole baseline
      before that leg may teach anything, with the wander tracked
      incrementally per leg (_track_baseline_heading). Re-analysing the
      same logs offline with exactly this filter turns the same hardware
      and the same fixes into a stable median offset of +5.4deg / -2.3deg
      holding to about +-1deg over six minutes - the sensor was never the
      problem, the training data was.

      Rejected legs are counted and surfaced as cal_skipped in the GPS
      status line: if that climbs while compass_offset never settles, the
      robot simply is not driving straight long enough to calibrate, which
      is a different problem from a bad compass and should not be mistaken
      for one. Set the spread to 0 to disable the gate.

      A second, larger source of the same corruption, found while checking
      the above against 57 straight legs from the 2026-08-07/08-23 outdoor
      logs: REVERSE legs. Backing up in a straight line passes the spread
      test perfectly, but the GPS chord then points ~180deg away from where
      the robot was facing, so the leg teaches an offset that is half a
      turn wrong. Matty reverses for 30-40% of some runs. Measured over
      those legs: including reversals the offset averaged +12.9deg with
      outliers to +134.5deg; excluding them, +3.9deg with a maximum of
      +24.0deg. _baseline_was_straight now also rejects any leg containing
      reverse motion, and does so even when the spread gate is disabled -
      a backwards chord is wrong regardless of how straight it was.

      For reference, with reverse and curved legs both excluded, that same
      data fits offset = +6.1deg + 5.6deg*cos(heading - 19deg). The
      constant is a good match for Prague's magnetic declination; the
      heading-DEPENDENT term is residual hard-iron distortion, which no
      single scalar offset can correct - see the compass notes for what
      that would take.

  28. OSM route following. New optional mode, switched on entirely from
      the config JSON by adding an osm_router:OSMRouter module and linking
      its 'route_hint' output to this node - see
      config/matty-tulak-osm.json. With no such module wired, route_mode
      stays False and NOTHING in this file behaves differently; every
      branch that reads it is skipped. That is the only safety argument
      that matters here, because the avoidance behaviour this mode sits on
      top of took twenty-seven items above to get right.

      The problem it fixes is not in the follower, it is in the target.
      Item 5's GPS following steers at the final destination, and a
      bearing to a destination 200m away points straight through whatever
      is in between - which in a park is a lawn. Replayed against the real
      OSM data for Stromovka, the 2026-08-29 runs show exactly that: the
      median fix sits 1.25m from a mapped path, but three separate
      episodes have the robot 6-35m out on the grass, held for 20-60s at a
      time, and one whole run (125548) averaged ~14m off-path. Those are
      not fix noise; noise does not hold a direction for a minute.

      So the router plans a path over the mapped, drivable way network and
      hands this node a rolling aim point a few meters ahead ON that path.
      Every existing mechanism - bearing_to_target, the blend against
      last_dir, the depth terms, the whole avoidance state machine and its
      priority over all of the above - is reused unchanged. The four
      places this file actually differs:

      (a) on_route_hint sets target_lat/target_lon from the aim point
          instead of from a QR code, and takes 'arrived' from the router.
          on_qr_code is then not wired at all in route mode - the router
          reads the QR itself, since it is the thing that needs to plan.
      (b) _bearing_distance_scale reads the router's remaining-distance
          ALONG THE ROUTE rather than target_dist, which in route mode is
          only ever about one lookahead and would otherwise fade the
          bearing to nothing for the entire run.
      (c) _drive_steering takes the bearing weight from the router's
          'authority' instead of from the road-fraction gate. Low on a
          straight path (the RedRoad mask centres better than a 1-3m GPS
          fix), high approaching and leaving a junction (the mask has no
          opinion about WHICH branch leads to the target - that is the one
          question only the map can answer), high while recovering from a
          confirmed excursion. This is the answer to "do I trust the
          network or the compass more": neither, everywhere - each one
          where it is the better instrument.
      (d) on_pose2d caps forward speed on the router's 'hold', which is
          what makes the robot wait at the start line until the QR is
          read (route_hold_creep_speed, 0.0 = stand still) and stop on
          arrival / when lost. Deliberately a cap on the finished command
          rather than an early return, so obstacle avoidance keeps full
          authority underneath it.

      (e) on_emergency_stop now RECORDS the button state instead of only
          acting on it once, and on_pose2d holds speed at 0 for as long as
          it is engaged - ahead of every other hold in this file,
          including the blind hold, because a human with a finger on a
          button outranks anything sensed. This only matters with
          terminate_on_stop=False, which the OSM config uses so that the
          press/release cycle can act as a mode reset (release returns the
          router to WAITING - see the QR command protocol in
          osm_router.py). With terminate_on_stop=True, unchanged: the
          exception ends the run on the press exactly as before, and the
          new hold is never reached.

          The reason the hold has to exist at all: with the run no longer
          terminating, nothing else would stop on_pose2d commanding a
          speed again on the very next cycle. Whether the ESP32 refuses
          motion while its own EMERGENCY_STOP status is set is not
          something this file can verify, so it does not rely on it.

      Note what is NOT delegated: nothing about safety. The router never
      sees a depth reading and cannot relax a threshold, start a maneuver,
      or suppress a stop. Its worst possible failure is pointing the
      bearing somewhere unhelpful, which the unchanged avoidance layer
      handles the same way it handles a bad bearing today.

  29. The road mask cannot bring Matty back to the road, and this is
      measured, not assumed. Two numbers off the 2026-08-29 Stromovka
      logs, both reproducible via replay_osm_router.py:

      (a) While 1-8m off the planned corridor, the mask steered back
      toward it in only 40-48% of frames - worse than a coin flip - with a
      mean contribution of -0.7 to -1.0deg, i.e. very slightly AWAY. It is
      a lane-KEEPING sensor with no notion of WHICH lane. Once Matty is on
      the grass and the grass reads drivable, nothing in on_nn_mask ever
      says "the road is over there". That is the whole mechanism behind
      the field report of Matty turning toward a lawn and continuing onto
      it, and no amount of tuning the mask fixes it, because the mask is
      not wrong - it is answering a different question.

      (b) The mask degrades predictably with camera-to-road misalignment.
      Within 15deg of the road axis it fragments into multiple blobs in
      27% of frames and its whole-mask centroid (mask_center averages ALL
      drivable pixels) lands >10% of half-width from the largest blob in
      7%. At 45-60deg off-axis: 51% and 30%. That 4x rise is exactly the
      failure where the centroid sits between a real road blob and a lawn
      blob and points at neither - and 45-60deg off-axis is the state an
      avoidance turn leaves the robot in.

      Tested and REJECTED on the same data: taking the largest connected
      component, or the component nearest the bottom-centre of the frame,
      instead of the global centroid. Neither beat the plain centroid
      against a map-derived reference, including on fragmented frames.
      The centroid is not the problem; the absence of any road-anchored
      reference is. Noted so it is not re-attempted.

      Three consequences, all inside route mode (item 28), all inert
      without a router wired:

      - _mask_trust() fades last_dir out as the camera swings off the
        mapped road axis, between mask_trust_full_deg and
        mask_trust_none_deg. The mask says less exactly where it is
        measurably least reliable.
      - _route_corridor_bias() adds a slow, heavily smoothed cross-track
        + heading-error term in 'corridor' mode - ADDITIVE, not blended.
        A blend lets the mask cancel the only restoring signal available;
        an addition cannot be outvoted, it shifts the equilibrium the mask
        settles into while leaving it in charge of the fast corrections it
        is genuinely good at. Gains are deliberately gentle (3deg per
        metre, capped at 15deg): GPS is the LOW-frequency term here and
        must never turn a noisy fix into a sharp correction.
      - the route's steering ceiling is now situational rather than
        turn_angle everywhere. turn_angle (20deg) is the right ceiling for
        a cruising correction and the wrong one for a T junction: at 20deg
        Matty's turning radius is 0.16/tan(10deg) = 0.91m, so a 90deg turn
        needs 1.4m of arc and swings wide across the corner; at the 40deg
        the router asks for near a junction it is 0.44m. _rate_limit_
        steering also takes the router's rate as a RELAXATION only (never
        a tightening), because winding on 40deg at the cruising 30deg/s
        would consume most of the junction.

      Why this is safe to add on top of twenty-eight items of tuned
      behaviour: none of it can start, suppress or relax a maneuver.
      _drive_steering is only ever reached when the avoidance state
      machine has already declined to act, the depth thresholds are
      untouched, and every new term is bounded (mask_trust in [0.25,1],
      bias in +-15deg, steer limit <= the platform's own 45deg cap).

  All new config keys (route_hold_creep_speed, route_cross_track_gain_deg_per_m,
  route_heading_gain, route_bias_max_deg, route_bias_alpha, mask_trust_full_deg,
  mask_trust_none_deg, mask_trust_min, route_min_road_frac, route_veto_authority,
  escape_after_cycles, escape_progress_dist_m,
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
  blind_clear_frames, blind_creep_speed, blind_creep_max_dist_m,
  compass_calibration_max_heading_spread_deg)
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

import cv2
import numpy as np

from osgar.node import Node


class _ReadTrackingConfig(dict):
    """A config dict that remembers which keys were actually looked at.

    Deliberately a local copy rather than an import from osgar: this file
    and the osgar package are deployed to the robot separately, which is
    the very problem this class exists to catch (2026-09-12 - a config key
    added the evening before was read by nobody, the obstacle windows sat
    on the pavement 0.7 m ahead for seven runs, and nothing failed). An
    app that cannot start unless the osgar package is also current would
    trade a silent wrong answer for an obscure ImportError."""

    # osgar.record/osgar.replay inject this into every module's init when
    # the robot config has a top-level 'env'. It belongs to the framework,
    # not to any node, and most nodes never read it.
    FRAMEWORK_KEYS = frozenset(['env'])

    def __init__(self, data):
        super().__init__(data)
        self.read_keys = set()

    def get(self, key, default=None):
        self.read_keys.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read_keys.add(key)
        return super().__getitem__(key)

    def unread(self):
        return sorted(set(self.keys()) - self.read_keys - self.FRAMEWORK_KEYS)

from osgar.exceptions import EmergencyStopException


EARTH_RADIUS_M = 6371000


def mask_center(mask):
    if mask.max() == 0:
        return mask.shape[0] // 2, mask.shape[1] // 2
    assert mask.max() == 1, mask.max()
    indices = np.argwhere(mask == 1)  # shape (num_points, 2)
    return tuple(int(x) for x in indices.mean(axis=0))


def _band_center_x(mask, r0, r1):
    """Horizontal centre of the drivable mask within one row band only.

    Unweighted on purpose: the band IS the weighting. Returns None when the
    band holds no road, which is the caller's signal to keep whatever aim
    point it already had rather than snap to the frame centre."""
    sub = mask[r0:r1, :]
    xs = np.nonzero(sub)[1]
    if len(xs) == 0:
        return None
    return float(xs.mean())


def _largest_blob(mask):
    """Keep only the biggest connected region of the drivable mask.

    The failure this removes is not fragmentation of the road itself - on
    the 2026-09-12 runs the largest region already holds essentially all
    of the mask area in the median frame. It is the occasional separate
    patch (a sunlit strip of lawn, a gravel verge) that classifies as road
    and sits off to one side, where it pulls the centroid - and therefore
    the steering - toward ground the robot must not drive on. A second
    region is never the road the robot is standing on."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:               # background plus at most one region - nothing to drop
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest).astype(mask.dtype)


def weighted_mask_center_x(mask, sky_row):
    """Horizontal centre of the drivable mask, weighting the NEAR rows
    (bottom of the frame) more than the far ones.

    Why weight rather than crop (item 32). A hard "focus window" was the
    obvious idea and it measures WORSE: over 44623 recorded frames a fixed
    mid-frame band raised the frame-to-frame centroid jitter from 2.34px to
    2.61px and went completely empty - no steering signal at all - in 2.5%
    of frames. A weighting cannot go blind, and it does not stop the robot
    reacting to a road that is only visible far away or off to one side,
    which is exactly what a crop would.

    A ramp still earns its place: the drivable fraction rises from 0.03
    just under the sky cut to 0.81 at the bottom of the frame, so the far
    rows contribute little signal but carry most of the noise - they are
    where pitch error moves the horizon and where an out-of-distribution
    view (see item 29) invents road on a lawn. Weighting them down cuts
    jitter to 2.15px, an 8% improvement, and shifts the centroid by only
    0.95px at the median, so it does not re-decide ordinary steering.

    Falls back to the frame centre on an empty mask, same as
    mask_center()."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask.shape[1] / 2.0
    span = max(1, mask.shape[0] - sky_row)
    weights = (ys - sky_row) / float(span)
    total = weights.sum()
    if total <= 0:
        return float(xs.mean())
    return float((xs * weights).sum() / total)


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
    if m is None:
        # Decimal degrees carrying a hemisphere letter, prefixed or
        # suffixed: "50.1299558N, 14.3793860E", "N50.12, E14.38".
        # Deliberately tried AFTER both forms above so neither changes
        # behaviour - the plain-decimal branch already handles anything
        # without letters, and this pattern cannot match a DMS string
        # (the closing " sits between the seconds and the hemisphere).
        #
        # Field case (2026-09-04 CZU runs 174623/174729/175216): three
        # codes in this exact format decoded perfectly and were then
        # silently discarded as "no coordinates", so the robot ignored a
        # target that had been shown to it correctly.
        decimal = r'(?:([NSEW])\s*)?(-?\d{1,3}\.\d+)\s*(?:([NSEW])\s*)?'
        m2 = re.search(decimal + r',\s*' + decimal, text)
        if m2:
            pair = [(float(m2.group(2)), m2.group(1) or m2.group(3)),
                    (float(m2.group(5)), m2.group(4) or m2.group(6))]
            # the letters also say WHICH coordinate is which, so
            # "14.38E, 50.13N" is unambiguous - un-swap it. Without
            # letters, keep the conventional latitude-first order.
            if pair[0][1] in ('E', 'W') and pair[1][1] in ('N', 'S'):
                pair.reverse()
            (lat, lat_hemi), (lon, lon_hemi) = pair
            if lat_hemi == 'S':
                lat = -abs(lat)
            if lon_hemi == 'W':
                lon = -abs(lon)
            if abs(lat) <= 90 and abs(lon) <= 180:
                return lat, lon
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
        # a config key nothing reads means the config is newer than this
        # file - see _ReadTrackingConfig and the check at the end of __init__
        config = _ReadTrackingConfig(config)
        bus.register('desired_steering', 'missed_turn')

        # driving
        self.max_speed = config.get('max_speed', 0.5)
        self.turn_angle = math.radians(config.get('turn_angle_deg', 20))
        # Camera geometry behind the road mask - see _mask_fov_scale.
        # Facts about the lens and about which model the gain was tuned
        # on, not things to tune: 69 deg is the OAK-D Pro colour camera
        # across the whole 4:3 sensor, 54.6 deg is what the 224x224
        # robotourist blob saw (a 1:1 crop keeps the height and throws
        # away a quarter of the width: atan(tan(34.5 deg) * 0.75) * 2).
        self.camera_hfov_deg = config.get('camera_hfov_deg', 69.0)
        self.mask_reference_hfov_deg = config.get(
                'mask_reference_hfov_deg',
                math.degrees(math.atan(math.tan(math.radians(self.camera_hfov_deg) / 2) * 0.75)) * 2)
        self._mask_fov_reported = None

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
        # see on_emergency_stop - only consulted when terminate_on_stop is
        # False, since otherwise the run ends on the press
        self.emergency_stop_active = False

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
        # What a confirmed drop-off should DO when it is not fatal. False
        # keeps the historical behaviour (hold speed 0 until the reading
        # clears), which at a real edge never clears, because nothing about
        # the view changes while parked. True backs out along the retrace
        # queue instead - see on_ground_hazard.
        self.ground_hazard_retreat = config.get('ground_hazard_retreat', True)
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
        # Straightness gate on LEARNING a calibration point (item 23).
        # travel_heading is the bearing between two GPS fixes
        # gps_heading_min_baseline_m apart - which equals the robot's actual
        # heading ONLY if it drove roughly straight between them. Under
        # continuous obstacle avoidance it does not: it curves, and then the
        # chord bearing differs from where the robot was ever pointing, so
        # every such sample injects an error into compass_offset. The
        # existing quality/hdop gates cannot see this at all - the fix can
        # be perfect while the path is a arc.
        #
        # Field evidence (2026-08-23, DGPS quality 2, 12 sats, hdop
        # 0.55-1.08 - i.e. GPS as good as it gets): compass_offset still
        # swung +13.6..+67.8deg inside a single run, and -1.0..+21.0deg in
        # another, for a quantity that is physically a near-constant
        # (magnetic declination plus a fixed mount error). Re-analysing the
        # same logs offline while accepting ONLY legs where yaw stayed
        # within 8deg gave a stable median of +5.4deg / -2.3deg holding to
        # about +-1deg over six minutes. Same hardware, same fixes - the
        # sensor was never the problem, the training data was.
        #
        # So: require the heading to have stayed within this spread across
        # the whole baseline before the leg is allowed to teach anything.
        # 0 disables the gate (old behaviour - learn from every
        # quality-passing leg).
        #
        # 12deg is swept, not guessed - offset swing across the four
        # 2026-08-23 runs, replaying each at several thresholds:
        #     gate:      off     8deg    12deg   15deg   20deg
        #     130201:   22.1    NEVER     0.0     0.7     0.7
        #     130549:   54.1      9.3    14.1    20.5    41.8
        #     130944:    4.8      0.0     0.2     1.1     1.1
        #     125614:   12.9      0.0     2.6     4.6     7.0
        # 8deg is tighter where it works, but 130201 never drove straight
        # for a full baseline and so never calibrated AT ALL - which is
        # worse than a noisy offset, since the robot then falls back to
        # GPS fix-differencing for heading (exactly what item 15 replaced).
        # 12deg is the loosest setting at which every run still calibrates
        # while the swings stay collapsed. Cost: first calibration lands
        # around t=36-58s instead of 21-28s, so allow roughly a minute of
        # driving before compass heading is fully trustworthy.
        self.compass_calibration_max_spread = math.radians(
            config.get('compass_calibration_max_heading_spread_deg', 8.0))
        # circular min/max of last_heading since the current baseline anchor,
        # tracked incrementally relative to the first sample - see
        # _track_baseline_heading/_baseline_heading_spread
        self._baseline_ref_heading = None
        self._baseline_dmin = 0.0
        self._baseline_dmax = 0.0
        # Did the robot reverse anywhere in this baseline? A leg driven
        # BACKWARDS is perfectly straight by the spread test above, but its
        # GPS chord bearing points 180deg away from where the robot was
        # facing - so it teaches the compass an offset that is a half turn
        # wrong. Matty reverses for 30-40% of some runs, so this is not a
        # corner case. Measured over 57 straight legs from the 2026-08-07
        # and 2026-08-23 outdoor logs: including reverse legs the offset
        # averaged +12.9deg with outliers to +134.5deg; excluding them,
        # +3.9deg with a maximum of +24.0deg.
        self._baseline_had_reverse = False
        self.compass_cal_skipped = 0   # legs rejected as curved or reversed - diagnostics
        # --- fixed vs learned compass offset (item 25) ---
        # compass_offset corrects magnetic declination plus the fixed mount
        # rotation. Both are constants, so it does not HAVE to be learned at
        # runtime - learning it from GPS was only ever a way to avoid having
        # to know them. Two knobs to skip that entirely:
        #
        #   compass_offset_deg   seeds the offset at startup, so the compass
        #                        is usable from the first cycle instead of
        #                        after the ~36-58s it takes to accumulate a
        #                        straight, forward, quality-passing GPS leg.
        #   compass_learn_offset set False to stop GPS refining it at all,
        #                        leaving the compass to run open-loop.
        #
        # Together they answer "how does Matty steer on compass alone" -
        # no 5m baselines, no travel_heading, GPS reduced to supplying the
        # target and the distance to it.
        #
        # Fitted from 48 straight forward legs across the 2026-08-07 and
        # 08-23 outdoor logs, the constant part of the offset was +6.1deg,
        # which is a sensible starting value for compass_offset_deg here
        # (Prague declination is about +5.2degE, the rest being mount
        # rotation). The same fit found a further 5.6deg varying WITH
        # heading - hard-iron distortion, which no single number can
        # correct; removing that needs a magnetometer ellipsoid calibration
        # on the ESP32 side, not anything in this file.
        self.compass_learn_offset = config.get('compass_learn_offset', True)
        fixed_offset_deg = config.get('compass_offset_deg')
        self.compass_offset = (math.radians(fixed_offset_deg)
                                if fixed_offset_deg is not None else None)
        self.compass_last_calibrated = None  # self.time of the last calibration update, diagnostics only
        # --- hard-iron correction (item 30) ---
        # compass_offset above is a single number, and a single number
        # provably cannot fix this compass. Fitted over 485 straight-line
        # samples from the 2026-09-04 CZU runs against the receiver's own
        # course over ground:
        #
        #     error = +7.5 deg  +  10.9 deg * cos(heading - 6 deg)
        #
        # The constant is declination plus mount rotation - that is what
        # compass_offset_deg is for, and 6.1 was a good estimate of it. The
        # 10.9 deg term VARIES WITH HEADING, which is hard iron: something
        # ferrous on the robot itself, turning with it. No offset removes
        # it, and it is the dominant error - residual against the constant
        # alone is a median 7.7 deg / p90 14.1 deg, against constant plus
        # this term 2.6 deg / 7.3 deg. Three times better.
        #
        # Drift within a run was only +-3 deg over 150-300s, so the compass
        # is steady; it is simply biased in a direction-dependent way.
        #
        # Re-fit with fit_compass.py after ANY change to what is mounted on
        # the robot - the August logs gave 5.6 deg at phase 19 deg for the
        # same constant, so this term does move when the hardware does.
        # 0 disables (single-offset behaviour, as before).
        self.compass_hardiron = math.radians(config.get('compass_hardiron_deg', 0.0))
        self.compass_hardiron_phase = math.radians(config.get('compass_hardiron_phase_deg', 0.0))

        self.target_lat = None
        self.target_lon = None
        self.target_dist = None
        self.bearing_to_target = None
        self.last_gps_pos = None  # (lat, lon) of the last fix used as a heading baseline
        self.travel_heading = None  # radians, compass convention - see initial_bearing()

        # --- course over ground from the receiver (item 26) ---
        # The GPS was already sending RMC once a second carrying its own
        # course over ground, and osgar was discarding it (split_buffer only
        # ever searched for GGA). Now parsed and merged into nmea_data as
        # 'cog'/'sog' - see osgar/drivers/gps.py parse_rmc.
        #
        # This is a better heading than travel_heading in every respect: the
        # receiver derives it from its velocity solution rather than from
        # the chord between two fixes, so it needs no baseline distance, no
        # straight-line assumption and no reverse-motion gate - all of which
        # exist purely to make position-differencing survivable. Measured
        # over the 2026-08-29 logs, median change between consecutive 1Hz
        # samples: 2.83deg for cog against 8.43deg for a 1-second chord.
        #
        # It has one genuine limitation: course comes from velocity, so at a
        # standstill there is none. The receiver reports it empty then, and
        # gps_course_min_speed additionally ignores it below a speed where
        # it would be mostly noise. Only 40% of the logged fixes carried a
        # usable cog, precisely because Matty spends a lot of time stopped
        # or crawling - so this supplements travel_heading, it does not make
        # the fallback chain redundant.
        self.use_gps_course = config.get('use_gps_course', True)
        self.gps_course_min_speed = config.get('gps_course_min_speed', 0.2)
        self.gps_course = None       # radians, compass convention
        self.gps_course_time = None  # self.time it was received
        self.gps_course_max_age = datetime.timedelta(
            seconds=config.get('gps_course_max_age_sec', 3.0))
        self.waypoint_reached = False
        self.last_gps_log_time = None
        # Every fix, unconditionally - unlike last_gps_pos, which only
        # advances once the robot has covered gps_heading_min_baseline_m
        # and so can be several meters and several seconds stale by
        # design. on_route_hint needs the CURRENT position to turn an aim
        # point into a bearing at 10Hz, and must not use the baseline
        # anchor for that.
        self.last_fix = None

        # --- OSM route following (item 28 at top of file) ---
        # Everything below is inert until an osm_router:OSMRouter node is
        # wired into 'route_hint'. With no such node this file behaves
        # exactly as before: route_mode stays False and every branch that
        # reads it is skipped.
        #
        # What changes when it IS wired: target_lat/target_lon stop being
        # the final destination from a QR code and become a rolling aim
        # point a few meters ahead on a planned, mapped path. All of the
        # bearing machinery below is reused unchanged - the improvement is
        # entirely in WHERE the target is, not in how it is followed.
        self.route_mode = False
        self.route_state = 'none'
        self.route_authority = None    # 0..1, how much the router wants the bearing trusted
        self.route_remaining_m = None  # along-route distance to the FINAL destination
        self.route_cross_track_m = None
        # metres past the edge of the nearest mapped road - see
        # on_route_hint. None until the router reports one (and on
        # every log recorded before it existed).
        self.route_off_road_m = None
        self.route_road_halfwidth_m = None  # how wide the surface here is - see _route_corridor_bias
        self.route_turn_dir = None          # radians, +left, signed angle of the next turn
        self.route_turn_dist_m = None       # metres to it
        self.route_exit_bearing = None      # radians, compass - the direction the route LEAVES that turn
        self.route_hold = None         # None | 'creep' | 'stop' - see on_pose2d
        self.route_guidance_mode = None    # 'corridor' | 'junction' | 'recovery'
        self.route_road_bearing = None     # radians, compass - which way the mapped path runs here
        self.route_steer_limit = None      # radians, this situation's steering ceiling
        self.route_steer_rate = None       # rad/s, this situation's rate ceiling
        self.route_speed_limit = None      # m/s, or None
        self.route_plan_seq = None

        # --- corridor bias (item 29) ---
        # In 'corridor' mode the route is applied as a small ADDITIVE bias
        # on top of the mask/depth steering rather than blended against it.
        # The reason is measured, not stylistic: over the 2026-08-29
        # Stromovka logs, while 1-8m off the planned corridor the road mask
        # steered back toward it only 40-48% of the time (mean contribution
        # -0.7 to -1.0deg, i.e. very slightly AWAY). The mask is a
        # lane-KEEPING sensor with no notion of which lane, so it supplies
        # no restoring signal at all - and a blend lets it dilute or cancel
        # the one signal that does. A bias cannot be outvoted; it shifts
        # the equilibrium the mask settles into.
        #
        # Gains are deliberately gentle. GPS position noise is bounded but
        # real (1-3m), so this must never translate a noisy fix into a
        # sharp correction - it is the LOW-FREQUENCY term, and the mask
        # keeps the fast one.
        self.route_cross_track_gain = math.radians(
            config.get('route_cross_track_gain_deg_per_m', 3.0))
        self.route_heading_gain = config.get('route_heading_gain', 0.35)
        self.route_bias_max = math.radians(config.get('route_bias_max_deg', 15.0))
        self.route_bias_alpha = config.get('route_bias_alpha', 0.15)
        self._route_bias = 0.0

        # --- how much the road mask is worth right now (item 29) ---
        # Measured on the same logs: with the camera within 15deg of the
        # road axis the mask fragments into multiple blobs in 27% of frames
        # and its whole-mask centroid lands >10% of half-width away from
        # the largest blob in 7%. At 45-60deg off-axis those become 51% and
        # 30% - a 4x increase in exactly the failure where the centroid
        # sits between a real road blob and a lawn blob, pointing at
        # neither. That is the state an avoidance turn leaves the robot in.
        # So the mask's contribution is faded out by how far the camera is
        # pointing off the mapped road axis, and the route bias above picks
        # up what it drops.
        self.mask_trust_full_deg = config.get('mask_trust_full_deg', 25.0)
        self.mask_trust_none_deg = config.get('mask_trust_none_deg', 60.0)
        self.mask_trust_min = config.get('mask_trust_min', 0.25)
        # Safety net replacing the road-fraction gate in junction mode: if
        # the mask sees essentially NO drivable surface the way the route
        # wants to go, something is wrong (bad fix, wrong branch, map
        # error) and the route should not command a hard turn into it.
        # Deliberately a veto at a very low threshold rather than the old
        # proportional gate, which would have blocked legitimate turns
        # toward a branch sitting at the edge of the frame.
        # Lateral half of the mask fade - see _mask_trust. Anchored to
        # what a real path allows: on a 3m park path, 1.2m off the mapped
        # centreline still has the robot on it; 3m off does not, whatever
        # the mask thinks it can see. 0 disables (heading error only).
        self.mask_trust_cross_full_m = config.get('mask_trust_cross_full_m', 0.0)
        self.mask_trust_cross_none_m = config.get('mask_trust_cross_none_m', 3.0)
        # How much of the steering the route may take over once the robot
        # is off the mapped way - see the end of _drive_steering's
        # corridor branch. Reached at mask_trust_cross_none_m, scaled
        # linearly from 0 at mask_trust_cross_full_m. 0 disables (the
        # corridor stays a pure additive bias, as before).
        self.route_offroad_authority = config.get('route_offroad_authority', 0.0)
        # Half-width the cross-track gain is expressed FOR - see
        # _route_corridor_bias. 0 disables the tube normalisation.
        self.tube_reference_halfwidth_m = config.get('tube_reference_halfwidth_m', 0.0)
        # --- junction turn hint (item 39) - see _route_turn_hint ---
        # Ceiling on the additive push toward the next turn, the distance
        # over which it ramps in, and the turn angle at which it reaches
        # full strength. 0 disables.
        self.route_turn_hint_max = math.radians(config.get('route_turn_hint_max_deg', 0.0))
        self.route_turn_hint_lead_m = config.get('route_turn_hint_lead_m', 10.0)
        self.route_turn_hint_full_angle = math.radians(
            config.get('route_turn_hint_full_angle_deg', 80.0))
        # How much harder the hint may push once actually AT the fork,
        # where a turn has to be committed rather than merely suggested.
        # Applied in the junction branch only.
        self.junction_hint_gain = config.get('junction_hint_gain', 3.0)
        # close the hint on heading toward the route's exit bearing - see
        # _route_turn_hint. False keeps the open-loop turn-angle hint.
        self.route_turn_closed_loop = config.get('route_turn_closed_loop', False)
        # --- asymmetric damping of the outward depth nudge (item 38) ---
        # See _damp_outward. Starts well inside the road, because it is
        # not a correction - it only declines to push further out - and
        # because the cross-track's SIGN is reliable at a scale where its
        # magnitude is not. 0 disables.
        self.outward_damp_start_m = config.get('outward_damp_start_m', 0.0)
        self.outward_damp_full_m = config.get('outward_damp_full_m', 1.5)
        self.outward_damp_max = config.get('outward_damp_max', 0.8)
        # cross-track past which the map breaks the tie on which way an
        # avoidance turn goes - see _road_side_preference. 0 disables.
        self.turn_road_side_min_cross_m = config.get('turn_road_side_min_cross_m', 0.0)
        # Avoidance side choice - see _around_obstacle_side and
        # _choose_turn_sign. Pass an in-corridor object on the side it is
        # not on; otherwise prefer a side that is clearly more open.
        self.turn_around_obstacle = config.get('turn_around_obstacle', False)
        self.turn_around_min_lateral_m = config.get('turn_around_min_lateral_m', 0.05)
        # How far ahead an in-corridor object may be and still decide the
        # side, as a multiple of turning_dist - see _around_obstacle_side.
        self.turn_around_lead_factor = config.get('turn_around_lead_factor', 1.0)
        self.turn_clearance_margin_m = config.get('turn_clearance_margin_m', 0.0)
        # Clearance past which "more open" stops being a reason to prefer
        # a side, so the road mask breaks the tie instead - see
        # _choose_turn_sign. 0 = no cap (extra room always wins).
        self.turn_clearance_comfort_m = config.get('turn_clearance_comfort_m', 0.0)
        # Road-mask veto on those two depth-only picks - see
        # _road_frac_veto. Both are fractions of the road mask's own
        # half-frame mean, whose ceiling is 0.5 because the top half of
        # the frame is zeroed in on_nn_mask; 0.05 is therefore "a tenth
        # of what a fully drivable side reads". 0 disables the veto.
        self.turn_side_min_road_frac = config.get('turn_side_min_road_frac', 0.0)
        self.turn_side_road_frac_margin = config.get('turn_side_road_frac_margin', 0.10)
        # Fraction of the router's cruise authority the aim point gets in
        # corridor mode - see _drive_steering. 0 restores the previous
        # bias-only behaviour.
        self.corridor_aim_frac = config.get('corridor_aim_frac', 0.0)
        # How much of the route's junction authority an urgent depth edge
        # takes back - see _drive_steering's junction branch. 0 restores
        # the previous behaviour (the route keeps full authority through a
        # fork whatever is beside the robot); 1.0 lets a full-strength
        # edge correction silence the route entirely.
        self.junction_edge_yield = config.get('junction_edge_yield', 0.0)
        # Speed ceiling for REALIGNING, as a fraction of what
        # _adaptive_speed would otherwise allow. REALIGNING is the one
        # forward-driving state that ignores the depth-based steering
        # blend entirely - it holds a heading regardless of what is beside
        # it - so it is the state with the least right to run at full
        # speed. 1.0 = unchanged.
        self.realign_speed_factor = config.get('realign_speed_factor', 1.0)
        # --- leaving the road is a terminal event (item 37) ---
        # At Robotour, driving off a mapped road ends the run. That makes
        # it the same KIND of event as a collision, not a preference to be
        # traded against comfort - and the two therefore deserve the same
        # shape of response, which until now only obstacles had:
        #
        #   obstacle:  turning_dist -> react   stop_dist -> hard limit
        #                             ...and speed scales with clearance
        #   off-road:  offroad_full_m -> react   offroad_hold_m -> hard limit
        #                             ...and speed scales with off_road_m
        #
        # Thresholds from the measured off_road_m distribution over the
        # 2026-09-05 runs (p50/p75 = 0.00, p90 0.71, p95 1.27, p99 2.48):
        # 0.5m is past the noise floor without being past the first 12% of
        # excursions; 2.0m is the top 1.3% and is unambiguous.
        self.offroad_full_m = config.get('offroad_full_m', 0.5)
        self.offroad_none_m = config.get('offroad_none_m', 2.0)
        # Speed at a full-scale excursion. Speed is what converts a
        # steering error into METRES of grass, so this is the single most
        # direct lever there is on how much road-leaving actually costs -
        # and it was the one thing the old design had no opinion about at
        # all: speed came only from obstacle clearance, so the robot held
        # 0.5 m/s while drifting. Defaults to min_speed.
        self.offroad_speed = config.get('offroad_speed', self.min_speed)
        # Hard limit, the stop_dist analogue: past this, capped at
        # offroad_speed whatever anything else says. NOT a full stop -
        # a stop cannot recover, and the only way back onto the road is
        # to drive there. 0 disables.
        self.offroad_hold_m = config.get('offroad_hold_m', 0.0)
        self.offroad_hold_active = False
        self.route_min_road_frac = config.get('route_min_road_frac', 0.02)
        # what the route's authority is cut to when that veto fires - not
        # zero, because "the mask sees nothing that way" is also what a
        # blinded or badly-lit frame looks like, and the map is then the
        # better of two poor witnesses
        self.route_veto_authority = config.get('route_veto_authority', 0.3)
        # What "hold: creep" means for this robot. 0.0 = stand still until
        # the router has a plan (i.e. until the start QR has been read),
        # which is the safe default: crawling forward before knowing where
        # the target is can only take the robot somewhere it then has to
        # come back from. Set it to a small value (<= min_speed) if you
        # would rather Matty inch forward while waiting.
        self.route_hold_creep_speed = config.get('route_hold_creep_speed', 0.0)

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
        # The ground band is the "is the camera alive at all?" test. The
        # centre/profile windows sit near the horizon, so they legitimately
        # go blank whenever the robot faces open sky or a long view - which
        # is NOT blindness, and holding for it wastes time. The ground band
        # looks at the floor a metre or two ahead, which is present in
        # every scene Matty can drive in; if it still returns data, the
        # camera is working.
        #
        # Both 2026-08-29 mid-run "blind" episodes, side by side:
        #   130657 t=576-591: upper frame 65% -> 2.1% valid, GROUND STAYED
        #     60-62% the whole time. Camera fine, robot just facing open
        #     distance - 14s of creeping for nothing.
        #   133946 (sunny bridge): upper 62% -> 0.1% AND ground 0.5% ->
        #     0.0%. Genuinely nothing anywhere - water, bright smooth deck
        #     and low sun give stereo no texture at all.
        # Only the second is blindness. This threshold separates them.
        self.blind_ground_valid_frac = config.get('blind_ground_valid_frac', 0.10)
        self.ground_valid_frac = None  # latest from on_ground_hazard, kept even when the hazard reaction is off
        self.blind_confirm_frames = config.get('blind_confirm_frames', 3)
        self.blind_clear_frames = config.get('blind_clear_frames', 3)
        self.depth_blind_active = False
        self.blind_streak = 0
        self.blind_clear_streak = 0

        # --- what to DO while blind (item 24) - three settings, from most
        # to least conservative:
        #
        #   blind_hold_enabled=False
        #       No blind handling at all. The centre's 0.0 fail-safe goes
        #       to the state machine as before, which reads it as an
        #       obstacle and reverses. Restores pre-item-22 behaviour.
        #   blind_hold_enabled=True, blind_creep_speed=0
        #       Stand still until depth returns (item 22's behaviour).
        #   blind_hold_enabled=True, blind_creep_speed>0
        #       Keep going forward at that speed instead of stopping.
        #
        # The creep exists because standing still is not always the safest
        # option in context: a run that must reach a waypoint cannot spend
        # 3-18% of itself parked (measured over the 2026-08-23 runs), and
        # a robot stopped in the open is not obviously better off than one
        # moving at a crawl. What makes the crawl defensible is speed
        # alone - the 2026-08-23 leg impact happened at 0.50m/s, where the
        # same contact at 0.15m/s is a bumper tap. Anything above
        # min_speed defeats the point, so keep it at or below that.
        #
        # Steering while creeping comes from _drive_steering(), which
        # degrades correctly on its own: with depth_profile all-unknown
        # _depth_confidence() returns 0, so the depth term drops out of
        # the blend and what remains is the RGB road mask plus the GPS
        # bearing. Both keep working while stereo does not - measured on
        # the 2026-08-23 logs, the road mask stayed at a 0.20-0.22
        # drivable fraction during blackouts against 0.23-0.31 when depth
        # was healthy.
        #
        # Two safety bounds, both deliberate:
        #  - creeping only happens from DRIVE. Going blind midway through
        #    an avoidance maneuver means something was already known to be
        #    there, and blind-driving forward into it is exactly wrong; in
        #    any other state the robot holds regardless of this setting.
        #  - blind_creep_max_dist_m caps how far one blind stretch may
        #    travel before it stops anyway. 0 = unlimited, which is what
        #    "just keep going" means and also what will drive into a river
        #    given a long enough blackout - set a real budget if the route
        #    goes anywhere near water or a drop.
        self.blind_creep_speed = config.get('blind_creep_speed', 0.0)
        self.blind_creep_max_dist_m = config.get('blind_creep_max_dist_m', 0.0)
        self._blind_anchor_xy = None      # where the current blind stretch began
        self._blind_budget_spent = False  # so the "budget exhausted" line prints once
        # the OAK pipeline takes several seconds to boot, during which
        # last_obstacle/left_dist/right_dist above still hold their
        # "assume clear" init values - stay stopped in on_pose2d until the
        # first real obstacle_zones frame arrives instead of driving on
        # that default (see on_pose2d and on_obstacle_zones)
        self.have_obstacle_data = False

        # road following
        # Weight near rows over far ones when locating the road centre -
        # see weighted_mask_center_x. False restores the plain centroid of
        # every drivable pixel below the sky cut.
        self.mask_row_weighting = config.get('mask_row_weighting', True)
        # --- road-mask post-processing (item: cobblestone instability) ---
        # Measured on the 2026-09-12 Stromovka runs, restricted to frames
        # where the robot was provably ON a mapped road (OSM off_road <
        # 0.5 m), so this is the network's behaviour on the road and not a
        # count of how often it was in the grass:
        #
        #   surface        frames   empty mask   centroid jitter p95
        #   asphalt         10871         2.7%                 0.031
        #   compacted        9002         2.7%                 0.024
        #   sett (cobbles)   3412        11.6%                 0.134
        #   paving_stones    4966        14.3%                 0.095
        #
        # So the report is real: on cobbles the mask goes blank 4x more
        # often and its centroid moves 4x further between consecutive
        # frames. Since on_nn_mask turns the centroid straight into
        # last_dir with no memory at all, that jitter is steering.
        #
        # Two independent failures needing two different answers:
        #   mask_center_alpha - an EMA on the centroid, so a single bad
        #     frame cannot swing the wheel. 1.0 disables it (old
        #     behaviour). At 10 fps, 0.4 is a ~0.25 s time constant:
        #     slower than the jitter, far faster than a real corner.
        #   mask_hold_sec - an EMPTY mask currently falls back to the
        #     frame centre, i.e. "drive straight", which on a curve steers
        #     off the road. Holding the last good centroid for a short
        #     while is strictly better: the road did not move in 0.5 s.
        #     0 disables it.
        #   mask_largest_blob - keep only the biggest connected region, so
        #     a patch of lawn that momentarily classifies as road cannot
        #     drag the centroid sideways.
        self.mask_center_alpha = config.get('mask_center_alpha', 1.0)
        self.mask_hold_sec = config.get('mask_hold_sec', 0.0)
        self.mask_largest_blob = config.get('mask_largest_blob', False)
        self.mask_min_frac = config.get('mask_min_frac', 0.0)
        self._mask_center_ema = None      # smoothed centre_x, pixels
        self._mask_last_good = None       # (time, centre_x) of the last trusted frame
        self.mask_held = False            # diagnostics: is the held value driving?
        self.steer_debug = {}            # diagnostics - see _drive_steering

        # --- aim point: how far ahead the mask is read (item: too low) ---
        # The centroid is currently weighted toward the BOTTOM of the frame,
        # i.e. toward the ground just past the bumper. Scored against the
        # mapped road's own direction over 47147 frames, moving the aim up
        # is a real trade, not a free win:
        #
        #   blend k   agree%   agree_off%   corr    (k = weight on the
        #      0.00     66.8         74.8   0.05     mid band, 1-k on the
        #      0.25     62.8         73.0   0.09     bottom band)
        #      0.50     60.9         70.6   0.13
        #      1.00     57.2         66.0   0.19
        #
        # Reading up the frame doubles-to-quadruples correlation with where
        # the road GOES (anticipation, which is the stated point) and costs
        # sign agreement with which way the road IS - and that second column
        # is the one that recovers the robot from a verge. Since leaving the
        # road ends a Robotour run, the default stays at 0 and this is a
        # knob to try deliberately on a test drive, not a silent change.
        # 0.25 is the value to try first: -1.8 points of off-road agreement
        # for nearly double the anticipation.
        self.mask_aim_blend = config.get('mask_aim_blend', 0.0)
        self.mask_aim_rows = tuple(config.get('mask_aim_rows', [0.55, 0.75]))
        self.mask_pos_rows = tuple(config.get('mask_pos_rows', [0.78, 1.0]))

        # --- road lost / camera-side off-road veto ---
        self.road_lost_frames = config.get('road_lost_frames', 0)
        self.road_lost_speed = config.get('road_lost_speed', 0.2)
        self.road_blank_streak = 0
        self.road_lost = False
        self.road_ahead_frac = 1.0
        # fades the DEPTH steering terms out as the strip ahead stops being
        # road - the camera-side twin of _route_offroad_frac, which only
        # works in route mode and so was inert for 80.9% of the session
        # Named by what the ROAD-AHEAD FRACTION means, not by the output,
        # because the first version of this read the two bounds the wrong way
        # round and silently returned 0 for every frame - a fade whose bounds
        # can be swapped without anything complaining is a fade that will be.
        # on_frac must be > off_frac or the whole thing is disabled.
        self.camera_offroad_on_frac = config.get('camera_offroad_on_frac', 0.0)
        self.camera_offroad_off_frac = config.get('camera_offroad_off_frac', 0.0)

        # --- grey (invalid) depth bins in the edge correction ---
        # _edge_bin_correction used to score an invalid edge bin as an
        # obstacle at 0.0 m - the most urgent reading possible - so a grey bin
        # produced the full edge_correction_max_deg push away from its side.
        # _free_space_steering two methods above it skips the same bins as
        # unknown; the two disagreed about what "invalid" means.
        #
        # Measured on the 2026-09-14 CZU runs (21 logs, replayed through this
        # class): 1267 steering pushes of 12 deg or more came from a grey edge
        # bin, and in 69% of them the road network showed road in that very
        # bearing. The typical cause is the most harmless scene there is - an
        # open road running off into the distance, whose far pixels are all
        # far-mask fill and therefore "no real measurement". The more open the
        # road on one side, the more likely that side's bin goes grey, and the
        # harder Matty was pushed AWAY from it. Run 16:58:58 left the road that
        # way (t=29.5 s: +35 deg left against a mask asking right).
        #
        # A grey bin is not evidence of nothing, though: a plain white wall
        # close enough to kill the stereo match reads exactly the same. The
        # road network separates the two, because it does not mark walls or
        # obstacles as road (checked on the 16:50:09 alley and the home run).
        # So:
        #   'legacy' - grey = obstacle at 0 m (previous behaviour)
        #   'mask'   - grey over visible road is ignored; grey over anything
        #              the network does not call road still blocks
        #   'open'   - grey is always ignored
        # In the alley every wall push came from a REAL close reading over
        # non-road columns, and every grey push was over road - so 'mask'
        # keeps the wall avoidance and removes only the artefact.
        self.edge_grey_mode = config.get('edge_grey_mode', 'legacy')
        self.edge_grey_road_frac = config.get('edge_grey_road_frac', 0.15)
        # Real readings far off to the side are not a flank threat. With
        # adaptive distances free_space_min_dist grows with speed, and an edge
        # bin sits ~29 deg off axis, so a bush 5 m away at 2.4 m lateral was
        # "blocking" and pushed 12+ deg: 493 of 1838 real-reading pushes on
        # 09-14 were from 5 m or further. Only readings whose lateral offset
        # d*sin(bearing) is within this many metres count. 0 disables.
        self.edge_lateral_max_m = config.get('edge_lateral_max_m', 0.0)
        self.mask_bin_rows = tuple(config.get('mask_bin_rows', [0.50, 0.80]))
        self.profile_bin_road = []        # mask road fraction inside each depth bin's bearing band
        self.branch_road_left = 0.0       # far-band road on the left / right thirds - see _arrow_turn_hint
        self.branch_road_right = 0.0
        self.branch_dir_left = None       # steering angle toward the road in that outer third
        self.branch_dir_right = None

        # --- route guidance style: 'legacy' or 'arrow' ---
        # 'legacy' is the previous behaviour: in junction/recovery mode the
        # router's bearing gets route_authority (0.85 / 0.90) of the steering,
        # and the turn hint is triggered and released by GPS position.
        #
        # Measured on 09-14 that is what put Matty on the grass in runs
        # 16:58:58, 17:18:03, 17:21:48 and the repeated 17:34 tries. While
        # the camera saw the robot squarely on a road, GPS+map placed it more
        # than 2 m off in 14% of fixes, and at the 17:34 spot in 66-100%.
        # Recovery mode then commanded 0.90 x 30 deg toward that biased line
        # - straight off the real road. The position-triggered hint circled
        # for 35 s in 17:40:22 because the robot never "passed" a junction
        # point 3 m from where it really was (and the router on the robot was
        # older than this repo and never published exit_bearing_deg, so the
        # heading closed loop below never ran in the field).
        #
        # 'arrow' treats the map the way a person treats a satnav arrow:
        #   - the road network always drives; GPS bearing never takes the
        #     wheel while the camera sees road (route_bearing_authority_max)
        #   - a planned turn is armed within route_turn_hint_lead_m of the
        #     junction, but only steers toward a branch the network can
        #     actually see on that side
        #   - it is closed on COMPASS heading and releases when Matty faces
        #     the exit bearing, or after route_turn_timeout_m of odometry -
        #     never on GPS progress, so it neither circles nor quits when the
        #     projection jumps past the junction (17:12:27, t=58.6 s)
        #   - turns sharper than route_turn_max_turn_deg (U-turns from a
        #     replan, 17:26:15 t=96 s) are not steered at all
        self.route_guidance_style = config.get('route_guidance_style', 'legacy')
        self.route_bearing_authority_max = config.get('route_bearing_authority_max', 1.0)
        self.route_turn_max_turn = math.radians(config.get('route_turn_max_turn_deg', 180.0))
        self.route_turn_branch_min_frac = config.get('route_turn_branch_min_frac', 0.05)
        self.route_turn_branch_full_frac = config.get('route_turn_branch_full_frac', 0.20)
        self.route_turn_done = math.radians(config.get('route_turn_done_deg', 20.0))
        self.route_turn_timeout_m = config.get('route_turn_timeout_m', 20.0)
        # The arrow's pull ramps from 0 when armed (route_turn_hint_lead_m out)
        # to full once the ODOMETRY estimate of the remaining distance is
        # inside this radius - the GPS error the turn has to tolerate. 0 = no
        # ramp (full pull from arming, the second-pass behaviour).
        self.route_turn_near_m = config.get('route_turn_near_m', 0.0)
        # junction/recovery modes raised the steering rate limit to the
        # router's 90 deg/s, which is where the 40-49 deg/s yaw swings at the
        # 17:18:03 curb came from; False keeps max_steering_rate_deg_s always
        self.route_rate_boost = config.get('route_rate_boost', True)
        self._arrow = None                # the armed turn, see _arrow_turn_hint
        self._arrow_done_key = None

        # --- road lost: retreat along the path just driven ---
        # A blank mask for this long while cruising means Matty is no longer
        # looking at a road - off it already, or blinded (auto-exposure on a
        # bright sky). Crawling forward at road_lost_speed kept driving into
        # whatever was ahead (17:12:27 ending: into the bushes). The path just
        # driven is the one place known to be road, so reverse along the
        # retrace buffer until the network sees road again, at most
        # road_lost_retreat_max_m and road_lost_max_retreats times, then hold.
        # No GPS involved. 0 disables.
        self.road_lost_retreat_sec = config.get('road_lost_retreat_sec', 0.0)
        self.road_lost_retreat_max_m = config.get('road_lost_retreat_max_m', 3.0)
        self.road_found_frames = config.get('road_found_frames', 5)
        self.road_lost_max_retreats = config.get('road_lost_max_retreats', 2)
        # A hold ends only when the mask shows road again - and a robot
        # standing still keeps showing the network the same picture, so on
        # sunlit cobblestone (2026-09-19) it never did. After holding this
        # long, start a fresh retreat + scout cycle (a new view for the
        # network), at most road_hold_max_retries times until 5 s of road
        # resets the count. 0 = hold for good, as before.
        self.road_hold_retry_sec = config.get('road_hold_retry_sec', 0.0)
        self.road_hold_max_retries = config.get('road_hold_max_retries', 0)

        # --- heading without the magnetometer (2026-09-15 test8) ---
        # 'compass' (previous) or 'odometry_gps': odometry heading plus an
        # offset learned from GPS course on straight stretches - see
        # heading_estimator.py. The compass error measured against GPS course
        # was 44*cos(course - 32) deg on 09-15 (p90 55 deg) and 32*cos(course
        # - 39) on 09-14, against the 10.3 deg hard-iron term configured; the
        # estimator scored p50 2.5 / p90 9 deg out of sample on the same
        # stretches. With it, _current_heading() never falls back to the
        # compass: GPS course, else None (callers already handle None).
        self.heading_source = config.get('heading_source', 'compass')
        self.heading_est = None
        if self.heading_source == 'odometry_gps':
            from heading_estimator import OdoGpsHeading
            self.heading_est = OdoGpsHeading(alpha=config.get('heading_offset_alpha', 0.5),
                                             min_baseline_m=config.get('heading_min_baseline_m', 4.0),
                                             valid_travel_m=config.get('heading_valid_travel_m', 60.0))
        self._odom_heading = None
        self._odo_hist = deque()             # (odometry travel m, odometry heading) over the last few metres
        self._odo_travel = 0.0
        self._odo_hist_xy = None
        # How much of a planned turn is still to do. 'exit_bearing'
        # (previous): absolute heading against the router's exit bearing -
        # i.e. against the compass. 'odometry': the map's turn angle
        # (turn_rel_deg, else turn_dir_deg) against the odometry heading
        # change since the approach. 171614 t=73.5: the compass read 53 deg
        # driving north on a road mapped at 358, "exit bearing 41" was 12 deg
        # away, and the right turn was released 10.8 m before the junction.
        self.route_turn_frame = config.get('route_turn_frame', 'exit_bearing')
        self.route_turn_rel = None
        # At the junction (inside route_turn_near_m by odometry, or past it)
        # steer from the turn still to do - gain x remaining angle, capped -
        # rather than only toward the bearing of the branch in the camera. A
        # hairpin's branch is outside the field of view until it is behind
        # Matty: at 171920 the branch pull peaked at +8 deg and the 136 deg
        # turn became a 15 s arc into the parking apron. Still only toward
        # road visible on that side. 0 disables.
        self.route_turn_commit_gain = config.get('route_turn_commit_gain', 0.0)
        self.route_turn_commit_max = math.radians(config.get('route_turn_commit_max_deg', 35.0))
        # ...only for turns at least this sharp (a gentler branch is visible
        # ahead, and the branch pull reaches it without leaving the road), and
        # only from route_turn_commit_window_m past the odometry estimate of
        # the junction to route_turn_near_m before it. Without the window a
        # stale arm pulled -35 deg for 60 s in the 171614 replay.
        self.route_turn_commit_min = math.radians(config.get('route_turn_commit_min_deg', 90.0))
        self.route_turn_commit_window_m = config.get('route_turn_commit_window_m', 6.0)
        # Countdown to a turn by ODOMETRY (odometry frame only). From the
        # first time the router reports the turn within route_turn_track_m,
        # the junction's position is the median of (router distance + forward
        # odometry) and the distance still to go is that minus forward
        # odometry - so a jump of the GPS projection cannot fire or skip the
        # turn. 170349 t=374.9: the router's distance went 12.1 -> 2.8 ->
        # "passed" in two updates while Matty drove 1 m.
        self.route_turn_track_m = config.get('route_turn_track_m', 25.0)
        self.route_turn_track_pass_m = config.get('route_turn_track_pass_m', 8.0)
        # The odometry target is the approach heading (last 3 m) plus the
        # map's turn angle, so the approach has to have been DRIVEN: at least
        # this far since the turn was first reported. Before that only an
        # odo+gps heading against the exit bearing can arm it; with neither
        # (171614 t=6: plan made 1.1 m from a junction, nothing driven, no
        # heading) the turn is not armed.
        self.route_turn_approach_m = config.get('route_turn_approach_m', 3.0)
        # abandon once the heading has swung this much further from the exit
        # than it was when armed - the robot is doing something else
        self.route_turn_abandon_extra = math.radians(config.get('route_turn_abandon_extra_deg', 60.0))
        # Arm only while the heading (odo+gps) runs within this of the route
        # direction the router reported with the turn - i.e. Matty is on the
        # approach, not facing away from it. After the 171614 re-plan at
        # t=110 the new route started BEHIND the robot (route 177 deg, robot
        # 349 deg); the turn armed from the exit bearing anyway and pulled
        # -25 deg for 20 s toward a branch that was behind it. 0 disables.
        self.route_turn_align_max = math.radians(config.get('route_turn_align_max_deg', 0.0))
        # Heading sources the guard above trusts. odo+gps needs a 4 m GPS
        # baseline first, so right after a start it is not there yet - at
        # 164121 t=69 (started 3 m from a -113 deg junction, the start
        # manoeuvre already facing the exit) the heading was the receiver's
        # course 132 deg against a route running 38 deg, the guard never
        # looked, the turn armed for another 105 deg and pulled Matty onto
        # the lawn. The course over ground is good enough to tell "facing
        # the approach" from "facing 90 deg away from it".
        self.route_turn_align_sources = tuple(config.get('route_turn_align_sources', ['odo+gps']))
        # Refuse a turn whose own numbers disagree: the route itself turns
        # back at that junction (|turn_dir| over route_turn_max_turn_deg), or
        # the rotation still to do from the exit bearing is a U-turn or
        # differs from the router's turn by more than this. After the
        # missed-turn re-plan at 171129 t=76 the new route started with a
        # U-turn (turn_dir -180) that the router's chords called +45; the
        # exit bearing said 161 deg to the RIGHT, it armed, and the map-only
        # pull took Matty onto the grass. 0 disables.
        self.route_turn_arm_max_mismatch = math.radians(config.get('route_turn_arm_max_mismatch_deg', 0.0))
        # How much of the turn still counts as "facing the exit". The
        # tolerance is the SMALLER of route_turn_done_deg and this fraction
        # of the turn, floored at route_turn_done_min_deg. At 0.5 a 38 deg
        # turn was released with 19 deg still to do - half of it - and the
        # road follower took over pointing between the two branches
        # (161958 t=30.9, and all six turns of the 173022 run).
        self.route_turn_done_frac = config.get('route_turn_done_frac', 0.5)
        self.route_turn_done_min = math.radians(config.get('route_turn_done_min_deg', 0.0))
        # Turning where the camera sees no branch at all. The mud track at
        # 163151 never rose above 0.16 road fraction in the far band on the
        # turn side, so the branch pull stayed under 5 deg and Matty drove
        # past a turn it had correctly armed and counted down. This steers by
        # the map alone, but only inside the window below, only while the
        # camera still sees road where Matty IS, and only as far as this
        # angle. 0 disables.
        self.route_turn_blind = math.radians(config.get('route_turn_blind_deg', 0.0))
        self.route_turn_blind_window_m = config.get('route_turn_blind_window_m', 3.0)
        # ...and only for turns at least this sharp. A sharp branch points
        # behind the camera's view until Matty is half way round, so the map
        # is all there is; a mild one is visible, and blind steering there
        # only added GPS timing error (161640 +91 deg: behind the railing).
        # 0 = every turn, as before.
        self.route_turn_blind_min_turn = math.radians(config.get('route_turn_blind_min_turn_deg', 0.0))
        # ...and only while the camera sees at least this much road straight
        # ahead, i.e. the robot is demonstrably still ON a road when it starts
        # the turn. Defaults to the off-road threshold.
        self.route_turn_blind_min_ahead = config.get('route_turn_blind_min_ahead',
                                                     config.get('camera_offroad_on_frac', 0.45))
        # A turn abandoned with more than this fraction of it still to do is
        # reported to the router (missed_turn) so it can plan another way
        # round. Needs the app.missed_turn -> osm_router.missed_turn link.
        self.replan_on_missed_turn = config.get('replan_on_missed_turn', False)
        self.route_turn_missed_frac = config.get('route_turn_missed_frac', 0.5)
        # Never turn INTO the side the mask says is not road when the other
        # side is road - back up instead, or go round on the road side if the
        # depth profile shows room past the obstacle. The red tram-stop post
        # at 161958 t=100 read left 2.15 m (grass) and right 0.68 m (the
        # post): depth alone chose the grass, while the mask had 0.97 road on
        # the right and 0.00 on the left.
        self.avoid_road_side_veto = config.get('avoid_road_side_veto', False)
        self._forced_turn_sign = None
        # Scouting after the road-lost retreats are spent: creep toward the
        # last direction the camera saw road in, instead of holding on the
        # spot until a human moves the robot (161321 t=201, 165432 t=305).
        # 0 disables and keeps the hold.
        self.road_scout_m = config.get('road_scout_m', 0.0)
        self.road_scout_speed = config.get('road_scout_speed', 0.15)
        self.road_scout_max = math.radians(config.get('road_scout_max_deg', 35.0))
        self.road_scout_tries = config.get('road_scout_tries', 1)
        self.road_scout_min_frac = config.get('road_scout_min_frac', 0.3)
        # only scout where the map agrees the robot is still on a road, and
        # only with the depth clear this far ahead
        self.road_scout_on_road_m = config.get('road_scout_on_road_m', 1.0)
        self.road_scout_clear_m = config.get('road_scout_clear_m', 1.5)
        self._road_scout = None
        self._road_scouts = 0
        self._road_last_dir = 0.0
        self._turn_track = None
        self._turn_done_keys = set()
        self._turn_done_seq = None
        self._odo_fwd = 0.0                   # signed forward odometry, m (backing up counts down)
        # odometry heading summed cycle by cycle, never wrapped: a 136 deg
        # hairpin plus a 45 deg drift the other way is 181 deg still to do,
        # not -179 (170349 replay t=422: the wrapped angle pulled -35 deg,
        # away from the turn)
        self._odo_unwrapped = 0.0
        self._odo_unwrap_prev = None
        # A "turn" smaller than this (chord to chord) is not armed: a +3 deg
        # arm turned the arrow into a heading hold that pulled +16 deg as
        # soon as anything else moved the heading (165320 replay t=54).
        self.route_turn_min_rel = math.radians(config.get('route_turn_min_rel_deg', 0.0))
        # Speed caps from GPS-derived state. Measured over the 09-15 runs
        # (2728 s moving): full speed 17% of the time; the router's junction
        # cap bound 32%, the GPS off-road cap 26%, the recovery cap 23%, depth
        # clearance only 1.2%. 'legacy' applies the router's cap always;
        # 'camera' applies it only when the camera doubts the road ahead
        # (_camera_offroad_frac > 0) or an armed turn is within
        # route_turn_slow_m by odometry.
        self.route_speed_gate = config.get('route_speed_gate', 'legacy')
        self.route_turn_slow_m = config.get('route_turn_slow_m', 6.0)
        # GPS off_road_m only counts when the camera agrees - it feeds the
        # off-road speed cap, the mask-trust cut and the depth-weight cut.
        # While the camera saw road ahead it cut mask trust on 43% of cycles.
        self.route_offroad_camera_gate = config.get('route_offroad_camera_gate', False)
        self.road_lost_since = None
        self.road_found_streak = 0
        self._road_retreat = None
        self._road_retreats = 0
        self._road_hold = False
        self._road_hold_since = None
        self._road_hold_retries = 0
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
        # --- tail swing (item 31) ---
        # Matty is CENTRE-ARTICULATED (matty.py: radius =
        # (0.32/2)/tan(joint/2), joint in the middle), so steering one way
        # swings the rear body the OTHER way. An avoidance turn away from
        # an obstacle therefore sweeps the tail TOWARD it - and by the time
        # the robot is drawing level, the forward-looking depth zones have
        # already lost sight of it.
        #
        # Measured over the 2026-08-29 and 2026-09-04 runs, on forward
        # cycles commanding 15deg or more:
        #                       tail side <0.8m     ...seen live
        #     TURNING              67.9%               36.6%
        #     REALIGNING           41.7%                7.1%
        # i.e. in a third of hard-turn cycles there was something the tail
        # was swinging into that the camera could no longer see. Median
        # remembered clearance 0.43m, minimum 0.32m. That is the reported
        # "turns hard right after bypassing an obstacle and the rear wheels
        # catch it", and it is a flipping risk, not just a scrape.
        #
        # Two parts. A short per-flank memory supplies what the live zones
        # cannot (nothing looks sideways or back), and the steering
        # amplitude is then capped so the swept corner stays clear.
        self.tail_swing_enabled = config.get('tail_swing_enabled', True)
        # Chassis geometry, MEASURED 2026-09-05 (see _corner_swing_limit):
        # boxes 17.5x22.5cm with a 14.5cm gap and the joint at its centre,
        # 3cm bumpers, wheels adding 6.5cm each side -> 35.5cm wide,
        # 55.5cm long. Everything below follows from those numbers; change
        # them only if the robot changes.
        self.rear_corner_radius_m = config.get('rear_corner_radius_m', 0.329)
        self.rear_corner_angle = math.radians(config.get('rear_corner_angle_deg', 32.6))
        self.swing_margin_m = config.get('swing_margin_m', 0.10)
        self.swing_min_steering = math.radians(config.get('swing_min_steering_deg', 10))
        # The side zones report RANGE to something in a forward-DIAGONAL
        # sector, not lateral clearance. obstdet3d_zones' left/right
        # columns sit roughly 11-29 deg off the axis, so a reading of R
        # metres is about R*sin(20deg) laterally and R*cos(20deg) ahead.
        # Using the range directly as clearance - which is the obvious
        # mistake here - would overstate the room by about 3x.
        # Half-angle from the optical axis to the CENTRE of a side zone.
        # Derived, not guessed: obstdet3d_zones' left/right columns are
        # centred 180px from the frame centre of 640, and the OAK-D Pro
        # mono HFOV is ~80deg (Luxonis list VFOV 55, which this project
        # already uses, alongside HFOV 80) -> 180 * 80/640 = 22.5deg. The
        # config's own polar_profile_hfov_deg=66 over free_space_cols
        # 60..580 implies a full HFOV of 81deg, which corroborates it.
        self.side_zone_bearing = math.radians(config.get('side_zone_bearing_deg', 22.5))
        # ...and the INNER edge of that sector, which is what the
        # range->lateral conversion actually has to use - see
        # on_obstacle_zones. obstdet3d_zones' side columns run 60-220 and
        # 420-580 of 640, so the inner edges sit 100px from the centre:
        # 100 * 81/640 = 12.7deg. Defaults to side_zone_bearing_deg (the
        # previous, over-optimistic behaviour) so this cannot change
        # anything until it is configured.
        self.side_zone_inner_bearing = math.radians(
            config.get('side_zone_inner_bearing_deg', math.degrees(self.side_zone_bearing)))
        # --- side thresholds in LATERAL units (item 34) ---
        # See on_obstacle_zones. False restores the old range-vs-distance
        # comparison and its side_*_dist_factor knobs.
        self.side_thresholds_lateral = config.get('side_thresholds_lateral', True)
        # Anchored to the body: half-width 0.1775m plus a margin. The stop
        # is the last resort; the turn trigger fires earlier and is still
        # suppressed by side_trigger_center_clear_factor when the way
        # ahead is open, which is what keeps doorways passable.
        self.side_lateral_stop_m = config.get('side_lateral_stop_m', 0.25)
        self.side_lateral_turn_m = config.get('side_lateral_turn_m', 0.32)
        # Effective distance from the articulation joint to the outer rear
        # corner - what actually sweeps. NOT measured on the real chassis:
        # 0.35m is half the wheelbase plus an assumed body overhang. Check
        # it against the real robot; too small silently disables the guard,
        # too large just makes turns more conservative in tight spots.
        # Never clamp below this: the robot still has to be able to
        # maneuver out of a tight spot, and a guard that forbids turning
        # entirely would just trade a scrape for being stuck. Below this,
        # the existing side hard-stop and backup logic take over.
        # How long, and how far, a flank reading stays relevant. Both
        # bounds matter: the memory describes the world in the robot's own
        # frame, so it stops being true once the robot has moved on.
        self.flank_memory_time = datetime.timedelta(seconds=config.get('flank_memory_sec', 6.0))
        self.flank_memory_dist_m = config.get('flank_memory_dist_m', 2.0)
        self._flank_history = {-1: deque(), 1: deque()}  # key: +1 = left side, -1 = right
        self._odom_heading = 0.0   # pose2d heading, same frame as its xy - see _update_flank_memory
        # Only obstacles that are now BESIDE the body may limit the swing -
        # see _lateral_clearance. The side zones look ~22 deg AHEAD, so a
        # live reading is always something the robot has not reached yet;
        # the rear corner can only sweep it once the robot has driven level
        # with it. Until 2026-09-11 every side reading counted, including
        # the very pole an avoidance turn was trying to get away from: in
        # the test5 pole loops every TURNING cycle sat at exactly the
        # 10 deg swing floor, rotating Matty ~8 deg per attempt, so it
        # never cleared and backed up again - 4-10 times per pole.
        # False restores the old any-reading-counts behaviour.
        self.swing_beside_only = config.get('swing_beside_only', False)
        self.swing_forward_max_m = config.get('swing_forward_max_m', 0.05)
        self.swing_behind_max_m = config.get('swing_behind_max_m', 0.8)

        # --- forced-joint detection (item 35) - see on_joint_angle ---
        self.joint_force_detect = config.get('joint_force_detect', True)
        # Absolute ceiling: matty.py clips commands to max_steering_deg
        # (45), so 55 leaves 10 deg of headroom past anything commandable.
        # Measured across a full day: runs without contact peak at 44-51
        # deg (ordinary overshoot), the three runs WITH contact reached 53,
        # 58 and 65.
        self.joint_force_deg = config.get('joint_force_deg', 55.0)
        self.joint_force_confirm_frames = config.get('joint_force_confirm_frames', 3)
        self.joint_force_window = datetime.timedelta(
            seconds=config.get('joint_force_window_sec', 2.0))
        self._commanded_joint_history = deque()
        self.joint_force_streak = 0
        self.last_joint_angle = 0.0
        # Slower than backup_speed on purpose - see _enter_joint_escape.
        self.joint_force_escape_speed = config.get('joint_force_escape_speed', 0.12)
        self.joint_force_escape = datetime.timedelta(
            seconds=config.get('joint_force_escape_sec', 4.0))
        self.max_joint_escapes = config.get('max_joint_escapes', 2)
        self.joint_escape_active = False
        self.joint_escape_started = None
        self.joint_escape_count = 0

        # --- the robot's own corridor, from the profile bins (item 36) ---
        # The L/C/R zones cannot answer "is something in the way" for a
        # robot this wide, for two reasons that are both pure geometry:
        #
        #   - the CENTRE zone is narrower than the robot. cols 260-380 of
        #     640 at an ~81deg full HFOV is +-6.2deg, i.e. +-0.11m at 1m
        #     and +-0.16m at 1.5m, against a half-width of 0.1775m. At the
        #     distances that matter the robot is WIDER than the window
        #     watching for things it will hit.
        #   - between the centre and side zones there is nothing at all.
        #     cols 220-260 and 380-420 belong to no zone - two ~4deg gaps
        #     sitting exactly where an object 1-2m ahead has to be to be
        #     struck head-on. A thin pole lands in one of them and no
        #     trigger in this file ever sees it.
        #
        # And where the side zones DO see something, they mis-scale it:
        # side_zone_bearing converts range to lateral clearance using the
        # zone's NOMINAL CENTRE bearing (22.5deg), but the zone spans
        # roughly 9-26deg. An object at its inner edge gets its clearance
        # overstated by sin(22.5)/sin(9) = 2.4x. That is the 2026-09-05
        # handrail: R read 1.14m, which the side test scored as 0.44m of
        # lateral room against a 0.32m threshold, while the post was
        # really about 0.18m off the centreline - already inside the
        # robot's own width.
        #
        # depth_profile has neither problem. Its bins tile free_space_cols
        # CONTIGUOUSLY (no gaps) and finely (9 bins over ~66deg, ~7deg
        # each - narrow enough that a pole fills a quarter of one, where
        # the same pole is 10% of a zone), and _update_polar_memory
        # already maps a bin index to its camera bearing. So the honest
        # test is available from data this file already receives: convert
        # each bin to (lateral, forward) and ask whether anything sits
        # inside the swept corridor the robot is about to occupy.
        #
        # This ADDS a trigger; it removes none. corridor_check=False
        # restores exactly the previous behaviour.
        self.corridor_check = config.get('corridor_check', True)
        # Half the width actually swept. rear_corner_radius_m/
        # rear_corner_angle_deg already encode the chassis: the body
        # half-width is 0.1775m and the rear corner reaches 0.329m when
        # articulated. Use the body half-width plus a margin - the corner
        # sweep is _articulated_steering_limit's job, not this one's.
        self.corridor_half_width_m = config.get('corridor_half_width_m', 0.1775)
        self.corridor_margin_m = config.get('corridor_margin_m', 0.08)
        # continuous push that moves an in-corridor object out of the
        # robot's swept width - see _corridor_repulsion. 0 disables.
        self.corridor_repel_gain = config.get('corridor_repel_gain', 0.0)
        self.corridor_repel_centre_m = config.get('corridor_repel_centre_m', 0.08)
        self._repel_side = 0
        # Bins farther than this laterally cannot be hit head-on and are
        # left entirely to the side zones / edge nudge, so widening the
        # corridor is the single knob if this proves too twitchy.
        self.corridor_dist = None         # recomputed per frame - diagnostics/HUD
        # see _corridor_hold_dist / on_obstacle_zones - how long a close
        # corridor reading is held after the bin stops reporting it. 1
        # disables the hold (only the current frame counts).
        self._corridor_history = deque(maxlen=max(1, config.get('corridor_hold_frames', 1)))

        # --- near/low obstacle from the ground band (item 36b) ---
        # obstdet3d_zones' min_ground_dist - see that module's docstring.
        # The stereo returns nothing at all inside 0.30m and only 1.9% of
        # its pixels inside 0.7m, so an obstacle that gets close does not
        # produce a closer reading, it produces NO reading and the window
        # falls back to whatever is visible past it. The ground band is
        # the last channel that still sees it, because a thing standing on
        # the floor occupies the floor's pixels. Debounced here, same
        # pattern as every other streak in this file.
        self.ground_near_confirm_frames = config.get('ground_near_confirm_frames', 2)
        self.ground_near_streak = 0
        self.ground_near_active = False

        self.max_bumper_hits = config.get('max_bumper_hits', 3)
        self.bumper_hit_streak = 0
        self.bumper_stop_active = False
        self._last_xy = (0.0, 0.0)  # updated every on_pose2d cycle - see _on_bumper_hit

        # Nothing below may read config - see _ReadTrackingConfig. This is
        # the same guard obstdet3d_zones carries, for the same reason: on
        # 2026-09-12 a config deployed ahead of the code it configures cost
        # a whole test session, and the only symptom was a setting quietly
        # not being there.
        unread = config.unread()
        if unread:
            raise ValueError(
                    'tulak_obstacle: this config sets %d key(s) that this version of the '
                    'code never reads: %s. The config is newer than the app (or a key is '
                    'misspelt) - those settings are being ignored, not applied. Refusing to '
                    'start rather than drive with settings that are silently absent.'
                    % (len(unread), ', '.join(unread)))

    def send_speed_cmd(self, speed, steering_angle):
        return self.bus.publish(
            'desired_steering',
            [round(speed * 1000), round(math.degrees(steering_angle) * 100)]
        )

    def on_emergency_stop(self, data):
        # Tracked as state, not just acted on once, so the hold survives
        # the next on_pose2d cycle. With terminate_on_stop=True (the
        # default, and what the non-route config uses) the exception below
        # ends the run before that can matter and nothing changes; it is
        # terminate_on_stop=False - which the OSM config uses so the
        # press/release cycle can act as a mode reset - that needs the
        # software to keep itself stopped rather than trusting the state
        # machine not to command a speed on the very next cycle.
        self.emergency_stop_active = bool(data)
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
        # A contact is the strongest possible evidence that this heading
        # does not work, and it was the one kind of evidence nothing
        # recorded. Backing off straight and handing back to DRIVE leaves
        # the heading unchanged, so the next cycle re-decides from the
        # same inputs and drives into the same thing - which is exactly
        # what the 2026-09-05 dormitory sequence was: four front contacts
        # in seven seconds, each followed by a 1s retreat and another
        # approach, until max_bumper_hits ended the run. The obstacle
        # there (a kerb) is below the depth camera's downward view and
        # cannot be seen at all, so nothing in the sensing path was ever
        # going to break that loop; the memory of having hit it is the
        # only thing that can.
        #
        # failed_headings is already consulted by _best_scan_heading, and
        # is already cleared the moment the robot makes real progress
        # (see _start_avoidance_cycle), so this cannot poison anything
        # beyond the spot it was learned in.
        if not rear and self.last_heading is not None:
            self.failed_headings.append(self.last_heading)
        # ...and a SECOND contact in the same spot is by definition
        # "milder responses are not resolving this", which is the
        # definition of escape mode. Waiting for the third only meant
        # waiting for the run to end.
        if self.bumper_hit_streak >= 2 and not self.in_escape_mode:
            print(self.time, 'second bumper contact with no progress - forcing escape mode')
            self.escape_counter = max(self.escape_counter, self.escape_after_cycles)
            self.in_escape_mode = True
            self._turn_sign_committed = False   # let the next turn re-decide the side
        if rear:
            self._enter_turning()
        else:
            self._enter_backing_up(0, use_retrace=False)

    def on_joint_angle(self, data):
        """The articulation angle the platform actually reports, against the
        one it was told to hold. A large persistent gap means the joint is
        being FORCED - the rear body twisted relative to the front by
        something the robot has run onto - and it is the only direct
        measurement of load on Matty's single central joint.

        This is what the pillar climb looks like (2026-09-05 run 083121,
        the last three seconds before the operator's emergency stop):

            t=271.3  pitch 19.4 deg, joint -34, speed 0.50   <- full speed
            t=272.2  pitch 20.9 deg, joint  +7
            t=273.0  joint +64.2 while commanded +5          <- forced open
            t=273.8  joint +52.3
            t=274.4  emergency stop

        Note what is NOT in that trace: roll peaked at 8.4 deg. On an
        articulated chassis the rear can be climbing something while the
        box holding the IMU stays level, so a roll threshold does not see
        this at all - I looked for it there first and missed it. Across the
        whole day the peak joint angle separates cleanly: 65 deg on this
        run and 58/53 deg on the two other runs with contact, against
        44-51 deg on every run without - and the platform can only ever
        command 45.

        Treated as physical contact, because that is what it is, and
        handled by the same path as a bumper: stop, back off straight,
        count it, and give up for good if it keeps happening. Continuing
        to drive is what turns a wheel touching a pillar into a wheel
        climbing one."""
        if not self.joint_force_detect:
            return
        # Tested against an ABSOLUTE ceiling, not against the current
        # command. Comparing the two directly was the first thing I tried
        # and it is wrong: two ordinary servo artifacts look identical to
        # forcing, and both appear in these logs - the joint LAGGING a
        # newly raised command (-13 measured while +45 had just been asked
        # for) and the joint still SETTLING after a command falls back
        # toward zero. Either produces a 30-45 deg discrepancy with nothing
        # touching the robot, and a relative test fired on every run of the
        # day because of it.
        #
        # Neither artifact can push the joint PAST what the platform is
        # able to ask for: matty.py clips every command to max_steering_deg
        # (45). So a reading beyond that ceiling can only have come from
        # the world pushing the rear body around.
        self.last_joint_angle = data[0] / 100.0
        self.joint_force_streak = self.joint_force_streak + 1 \
            if abs(self.last_joint_angle) > self.joint_force_deg else 0
        if self.joint_force_streak == self.joint_force_confirm_frames and not self.joint_escape_active:
            print(self.time, 'JOINT FORCED to %.0f deg - past the %.0f deg the platform can even '
                              'command. The rear is jammed or climbing something.'
                   % (self.last_joint_angle, self.joint_force_deg))
            self._enter_joint_escape()

    def _enter_joint_escape(self):
        """Dedicated escape for a forced joint. Deliberately NOT the bumper
        reaction, and deliberately not a wider turn.

        Why reversing, and only reversing. The joint is PASSIVE (see
        matty.py's header) - it is steered by driving the wheels
        differentially, not by a servo - and send_speed() transmits the
        steering angle with whatever speed it was given, so at zero speed
        there is no differential and the joint cannot move at all.
        Stopping therefore does not straighten it; it only stops adding
        energy to the climb. The single motion that undoes running up onto
        something is backing off the way you came.

        Why NOT a wider turn afterwards, which is the intuitive next move.
        The rear outer corner's lateral reach is r*sin(phi0 + gamma) with
        r = 0.329m and phi0 = 32.6deg for this chassis, so it grows the
        harder you steer: 0.23m at 10deg, 0.32m at 45deg. A wider turn
        beside an obstacle sweeps the rear FURTHER into it - it is what put
        the wheel on the pillar in the first place. The turn that follows
        must be narrower, after gaining distance, which is exactly what
        _articulated_steering_limit already enforces.

        Three bounds, because an escape that is not working must stop
        rather than repeat:
          - slower than an ordinary backup: the joint is loaded, and with
            it forced to 65deg the robot reverses along a tight arc rather
            than straight however it is steered, so the actual path is not
            known.
          - steering commanded to 0, which is the largest unwinding error
            the differential can act on.
          - a hard deadline. If the joint has not come back under the
            ceiling within joint_force_escape_sec, this is a jam that
            reversing cannot solve, and continuing to pull against it is
            precisely how the one central joint gets damaged. Stop and
            leave it to a human."""
        self.joint_escape_active = True
        self.joint_escape_started = self.time
        self.joint_escape_count += 1
        self.send_speed_cmd(0, 0)
        print(self.time, 'joint escape %d/%d: reversing at %.2f m/s with the joint commanded '
                          'straight, for at most %.0fs'
               % (self.joint_escape_count, self.max_joint_escapes,
                  self.joint_force_escape_speed, self.joint_force_escape.total_seconds()))

    def _joint_escape_command(self):
        """Returns (speed, steering) while escaping, or None once free.
        Raises if it cannot free itself, or has had to try too often."""
        if abs(self.last_joint_angle) <= self.joint_force_deg:
            print(self.time, 'joint back to %.0f deg - freed, resuming' % self.last_joint_angle)
            self.joint_escape_active = False
            self.joint_force_streak = 0
            self._enter_backing_up(0, use_retrace=False)  # keep backing off a little
            return None
        if self.time - self.joint_escape_started > self.joint_force_escape:
            print(self.time, 'joint STILL forced at %.0f deg after %.0fs of reversing - this is a '
                              'jam that backing off cannot clear. Stopping rather than keep '
                              'pulling against the joint.'
                   % (self.last_joint_angle, self.joint_force_escape.total_seconds()))
            self.bumper_stop_active = True
            raise EmergencyStopException()
        if self.joint_escape_count > self.max_joint_escapes:
            print(self.time, 'joint forced %d times - stopping for good' % self.joint_escape_count)
            self.bumper_stop_active = True
            raise EmergencyStopException()
        return -self.joint_force_escape_speed, 0.0

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

    def _route_hold_is_absolute(self):
        """True when the router's hold means "do not move at all", as
        opposed to "crawl". 'stop' always does; 'creep' does too whenever
        route_hold_creep_speed is 0, which is the default and the normal
        setting - see the hold handling in on_pose2d for why this has to
        pre-empt the avoidance state machine rather than cap its output."""
        if self.route_hold == 'stop':
            return True
        return self.route_hold == 'creep' and self.route_hold_creep_speed <= 0

    def on_route_hint(self, data):
        """Aim point and trust level from osgar-apps/followme/osm_router.py
        (see item 28 at top of file). Arrives at the pose2d rate.

        The whole integration is this method plus four small reads of the
        flags it sets - deliberately, because everything downstream of
        "here is a bearing to follow" in this file is field-tuned and had
        no reason to change. What the router replaces is only the CHOICE
        of target: instead of the final destination, which a bearing
        points at straight through whatever lawn lies between, the target
        becomes a point a few meters ahead on a mapped path.

        data keys: state, lat/lon (the aim point, None when there is no
        active route), remaining_m (to the real destination, ALONG the
        route), authority (0..1), cross_track_m, off_route, arrived, hold
        ('creep'|'stop'|None - see on_pose2d)."""
        self.route_mode = True
        self.route_state = data.get('state', 'unknown')
        self.route_authority = data.get('authority')
        self.route_remaining_m = data.get('remaining_m')
        self.route_cross_track_m = data.get('cross_track_m')
        # metres PAST THE EDGE of the nearest mapped road (0 = on a road,
        # None = unknown/stale) - see osm_router's _gps_update. Absent on
        # logs recorded before it existed, which is what keeps the
        # cross-track fallback in _route_offroad_frac necessary.
        self.route_off_road_m = data.get('off_road_m')
        self.route_road_halfwidth_m = data.get('road_halfwidth_m')
        # signed angle of the next real turn and how far to it - the only
        # part of the plan a constant GPS offset cannot corrupt. See
        # _route_turn_hint.
        td = data.get('turn_dir_deg')
        self.route_turn_dir = math.radians(td) if td is not None else None
        self.route_turn_dist_m = data.get('turn_dist_m')
        eb = data.get('exit_bearing_deg')
        self.route_exit_bearing = math.radians(eb) if eb is not None else None
        # the same turn measured between chords a few metres either side of
        # the node (osm_router turn_rel_probe_m) - see route_turn_frame
        tr = data.get('turn_rel_deg')
        self.route_turn_rel = math.radians(tr) if tr is not None else None
        self.route_hold = data.get('hold')
        self.route_guidance_mode = data.get('mode')
        self.route_speed_limit = data.get('speed_limit')
        bearing_deg = data.get('road_bearing_deg')
        self.route_road_bearing = math.radians(bearing_deg) if bearing_deg is not None else None
        limit_deg = data.get('steer_limit_deg')
        self.route_steer_limit = math.radians(limit_deg) if limit_deg is not None else None
        rate_deg = data.get('steer_rate_deg_s')
        self.route_steer_rate = math.radians(rate_deg) if rate_deg is not None else None
        plan_seq = data.get('plan_seq')
        if plan_seq != self.route_plan_seq:
            # a fresh plan - whatever bias had built up was accumulated
            # against a route that no longer exists
            self._route_bias = 0.0
            self.route_plan_seq = plan_seq

        lat, lon = data.get('lat'), data.get('lon')
        if lat is None or lon is None:
            # no route to follow right now (waiting for the start QR, mid
            # re-plan, arrived, lost). Clear the target so _drive_steering
            # falls back to pure road following rather than steering at a
            # stale aim point; on_pose2d's hold handling decides whether
            # the robot may move at all.
            self.target_lat = self.target_lon = None
            self.bearing_to_target = None
            self.target_dist = None
        else:
            self.target_lat, self.target_lon = lat, lon
            if self.last_fix is not None:
                # recomputed here rather than waiting for on_nmea_data: the
                # aim point moves with every pose2d cycle while fixes only
                # arrive at 1Hz, and a bearing to last cycle's aim point is
                # exactly the lag this is meant to avoid at a junction
                self.target_dist = haversine_distance(*self.last_fix, lat, lon)
                self.bearing_to_target = initial_bearing(*self.last_fix, lat, lon)
        self.waypoint_reached = bool(data.get('arrived'))

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

    def _apply_compass_calibration(self, raw_compass):
        """raw geometric compass heading -> corrected true bearing.

        Two terms: the learned/configured constant (declination plus mount
        rotation) and the hard-iron sinusoid (see compass_hardiron in
        __init__). The sinusoid is a function of TRUE heading, which is
        what we are solving for, so it is evaluated at the
        constant-corrected heading - one fixed-point step, which is ample
        when the amplitude is ~11 deg and its derivative correspondingly
        small."""
        heading = normalize_angle(raw_compass + self.compass_offset)
        if self.compass_hardiron:
            heading = normalize_angle(
                heading + self.compass_hardiron * math.cos(heading - self.compass_hardiron_phase))
        return heading

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

    def _track_baseline_heading(self):
        """Accumulate how much last_heading has wandered since the current
        GPS baseline anchor was set. Called once per pose2d cycle.

        Kept as an incremental circular min/max around the first sample of
        the window rather than a list of samples - a baseline can span many
        seconds at 10Hz, and only the extremes matter. A turn beyond +-180deg
        would alias, but such a leg is nowhere near straight and gets
        rejected on the spread anyway."""
        if self._last_commanded_speed < -0.01:
            self._baseline_had_reverse = True
        if self._baseline_ref_heading is None:
            self._baseline_ref_heading = self.last_heading
            self._baseline_dmin = 0.0
            self._baseline_dmax = 0.0
            return
        delta = normalize_angle(self.last_heading - self._baseline_ref_heading)
        self._baseline_dmin = min(self._baseline_dmin, delta)
        self._baseline_dmax = max(self._baseline_dmax, delta)

    def _reset_baseline_heading(self):
        """Start a fresh window - called whenever the GPS baseline anchor
        moves, so each leg is judged only on its own heading history."""
        self._baseline_ref_heading = self.last_heading
        self._baseline_dmin = 0.0
        self._baseline_dmax = 0.0
        self._baseline_had_reverse = False

    def _baseline_heading_spread(self):
        """Total heading wander across the current baseline, radians."""
        return self._baseline_dmax - self._baseline_dmin

    def _baseline_was_straight(self):
        """Is this leg's GPS chord bearing usable as a stand-in for where
        the robot was actually pointing? Two ways it can fail:

          - the robot turned during the leg, so the chord is not any
            heading it actually held (compass_calibration_max_spread)
          - the robot REVERSED during the leg, in which case the chord
            points roughly opposite to where it was facing. This one is
            invisible to the spread test - backing up in a straight line
            is perfectly "straight" - and teaching it to the compass is
            worth a half turn of error."""
        if self._baseline_had_reverse:
            return False
        if self.compass_calibration_max_spread <= 0:
            return True  # spread gate disabled (the reverse check still applies)
        if self._baseline_ref_heading is None:
            return False  # no heading history for this leg - don't guess
        return self._baseline_heading_spread() <= self.compass_calibration_max_spread

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

    def _update_gps_course(self, data):
        """Take the receiver's own course over ground from the merged RMC
        fields, when it is trustworthy - see use_gps_course in __init__.

        Gated on speed twice over: the receiver leaves cog empty at a
        standstill (course is derived from velocity, and there is none),
        and gps_course_min_speed additionally rejects the crawl range where
        what it does report is mostly noise. Also requires the robot to be
        driving FORWARD - course over ground is the direction of travel, so
        while reversing it points a half turn away from where the robot
        faces, exactly the error that corrupted compass calibration from
        position chords."""
        if not self.use_gps_course:
            return
        cog, sog = data.get('cog'), data.get('sog')
        if cog is None or sog is None or sog < self.gps_course_min_speed:
            return
        if self._last_commanded_speed <= 0.01:
            return
        self.gps_course = math.radians(cog) % (2 * math.pi)
        self.gps_course_time = self.time

    def _fresh_gps_course(self):
        """gps_course if recent enough to still describe where the robot is
        pointed, else None. Stale course is worse than none - it keeps
        asserting the last direction of travel after a turn has begun."""
        if self.gps_course is None or self.gps_course_time is None:
            return None
        if self.time - self.gps_course_time > self.gps_course_max_age:
            return None
        return self.gps_course

    def on_nmea_data(self, data):
        self._update_gps_course(data)
        lat, lon = data.get('lat'), data.get('lon')
        if lat is not None and lon is not None:
            if data.get('lat_dir') == 'S':
                lat = -lat
            if data.get('lon_dir') == 'W':
                lon = -lon
            self.last_fix = (lat, lon)
            if self.heading_est is not None and self.time is not None:
                self.heading_est.update_fix(self.time.total_seconds(), lat, lon)

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
                    # Prefer the receiver's course over ground for teaching
                    # the compass: it is an instantaneous direction of
                    # travel matched to an instantaneous compass reading,
                    # so it needs none of the straightness/forward gates
                    # that exist only because a chord between two fixes
                    # averages over whatever the robot did in between
                    # (item 26). Those gates still apply when falling back
                    # to travel_heading.
                    course = self._fresh_gps_course()
                    straight = self._baseline_was_straight()
                    if (self.use_compass_heading and self.compass_learn_offset
                            and self._gps_fix_quality_ok(data) and course is not None):
                        self._update_compass_calibration(course)
                    elif (self.use_compass_heading and self.compass_learn_offset
                            and self._gps_fix_quality_ok(data) and straight):
                        # right when travel_heading is at its most
                        # trustworthy (baseline-confirmed, quality-gated,
                        # AND driven straight - see _baseline_was_straight;
                        # without that last one a curved leg's chord bearing
                        # gets taught to the compass as if it were a heading)
                        self._update_compass_calibration(self.travel_heading)
                    elif self.use_compass_heading and self.compass_learn_offset and not straight:
                        self.compass_cal_skipped += 1
                    # each leg is judged on its own heading history
                    self._reset_baseline_heading()
            else:
                self.last_gps_pos = (lat, lon)
                self._reset_baseline_heading()  # first anchor - start the window here
            if not had_heading and self.travel_heading is not None:
                print(self.time, 'GPS heading established: %.0f deg' % math.degrees(self.travel_heading))

            if self.target_lat is not None:
                self.target_dist = haversine_distance(lat, lon, self.target_lat, self.target_lon)
                self.bearing_to_target = initial_bearing(lat, lon, self.target_lat, self.target_lon)
                # In route mode target_lat/lon is a rolling aim point a few
                # meters ahead, NOT the destination - so target_dist is
                # always about one lookahead and would trip this check on
                # the very first fix. Arrival is the router's call there
                # (it knows the distance REMAINING ALONG THE ROUTE, which
                # is the honest measure); it arrives via on_route_hint.
                if (not self.route_mode and not self.waypoint_reached
                        and self.target_dist < self.waypoint_arrival_dist_m):
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
        if self.side_thresholds_lateral:
            # The side zones look DIAGONALLY forward (their columns sit
            # ~22.5deg off axis), so their reading is a RANGE, not the
            # lateral clearance beside the robot. Comparing that range
            # against a distance threshold - which is what the factors
            # below do - silently overstates the room by 1/sin(22.5) = 2.6x.
            #
            # Measured 2026-09-05 with the shipped factors: the side stop
            # fired at 0.45m of range, which is 0.17m of LATERAL clearance
            # against a half-width of 0.1775m. The obstacle was already
            # inside the robot's own width - the side hard-stop could not
            # fire before contact, by geometry, at any speed. That is the
            # 15 front-bumper hits on poles and columns in those logs, and
            # neither the zones nor the finer depth_profile bins were at
            # fault: both reported 1-4m because the pole sat in the last
            # 20deg of the sector where a metre of range is a hand's width
            # of clearance.
            #
            # So the sides are judged in the units the geometry is in.
            # Thresholds are anchored to the half-width plus a margin, and
            # the trade against narrow passages is measured, not guessed:
            #     lateral   fires (% fwd)  bumper hits caught in time
            #      0.20m        2.2%            3/15   (~ the old behaviour)
            #      0.25m        4.5%            3/15
            #      0.32m        6.7%            6/15
            #      0.36m        7.9%            8/15
            #
            # Shipped at stop 0.25 / turn 0.32. It IS a trade: replayed
            # over the 09-05 runs, state changes go 9.9 -> 12.2 per minute
            # and time spent reversing 5.2% -> 9.7%, all of it driven by
            # the STOP threshold (the turn threshold barely moves either
            # number). More manoeuvring is the right way round to be wrong
            # when the alternative is climbing a pole - the same argument
            # item 19 made - but side_lateral_stop_m is the single knob if
            # it proves too twitchy in the field.
            # INNER edge of the zone, not its centre. The conversion has
            # to answer "how close could this thing be to my flank", and
            # the zone spans a sector - an object anywhere in it could be
            # at the inner edge, which is the smallest lateral offset a
            # given range can correspond to. Using the centre bearing
            # instead assumes the object sits exactly mid-sector and
            # overstates the room by sin(centre)/sin(inner) - 2.4x for
            # the shipped 22.5/9 deg geometry, which is how a handrail
            # post 0.18m off the centreline was scored as 0.44m of
            # clearance and driven into (2026-09-05 run 153832).
            side_lateral = math.sin(self.side_zone_inner_bearing)
            l_side, r_side = l_dist * side_lateral, r_dist * side_lateral
            side_stop_dist, side_turning_dist = self.side_lateral_stop_m, self.side_lateral_turn_m
        else:
            l_side, r_side = l_dist, r_dist
            side_stop_dist = self.stop_dist * self.side_stop_dist_factor
            side_turning_dist = self.turning_dist * self.side_turning_dist_factor
        # The robot's own corridor - the head-on channel the zones cannot
        # provide (see _corridor_scan / corridor_check). Judged against
        # the CENTRE thresholds, because that is what it is: something
        # the robot is about to drive into, not something beside it.
        corridor = self._corridor_scan()
        corridor_fwd = corridor[0] if corridor else None
        self.corridor_dist = corridor_fwd
        # A thin object does not stop existing because one stereo frame
        # missed it, and a pole is exactly the case where it does. In the
        # 2026-09-06 rollover the corridor read 0.55m, then 2.65m, then
        # 0.35m, then 2.36m on consecutive frames as the pole moved
        # between bins and in and out of a clean stereo match - so a
        # stop_confirm_frames streak of CONSECUTIVE close readings never
        # completed, and nothing fired. Holding the smallest recent
        # reading turns an intermittent detection into a usable one,
        # while still expiring on its own once the object is genuinely
        # gone. Deliberately short: this is a hold, not a memory.
        self._corridor_history.append(corridor_fwd)
        corridor_fwd = self._corridor_hold_dist()
        corridor_stop = corridor_fwd is not None and corridor_fwd < self.stop_dist
        corridor_blocked = corridor_fwd is not None and corridor_fwd < self.turning_dist

        stop_now = (center < self.stop_dist or l_side < side_stop_dist or r_side < side_stop_dist
                    or corridor_stop or self.ground_near_active)
        self.stop_streak = self.stop_streak + 1 if stop_now else 0

        # The robot is only "clear" if the center AND both sides are further
        # than their respective thresholds - except that a side-only
        # blockage is ignored while the way ahead is clearly open, which is
        # what a doorway/gap looks like (see
        # side_trigger_center_clear_factor in __init__)
        # corridor_blocked joins center_blocked rather than side_blocked
        # on purpose: it must NOT be suppressed by
        # side_trigger_center_clear_factor below. That suppression exists
        # so a doorway's frames do not abort a passage the robot is lined
        # up to drive through - but a corridor hit is by construction
        # something in the gap itself, which is the one case where
        # "carry on, the way ahead is open" is wrong.
        center_blocked = center < self.turning_dist or corridor_blocked
        side_blocked = (l_side < side_turning_dist) or (r_side < side_turning_dist)
        if side_blocked and self.side_trigger_center_clear_factor > 0:
            if center > self.turning_dist * self.side_trigger_center_clear_factor:
                side_blocked = False
        is_blocked = center_blocked or side_blocked

        self.turn_streak = self.turn_streak + 1 if is_blocked else 0
        # see center_blocked_streak in __init__ - deliberately centre only
        # (the corridor counts as "ahead" here too, same reasoning as
        # center_blocked above)
        blocked_ahead = center < self.turning_dist or corridor_blocked
        self.center_blocked_streak = self.center_blocked_streak + 1 if blocked_ahead else 0

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
        # NOT keyed off the centre reading any more. It used to test
        # "centre == 0.0", the sentinel _dist returns for an untrusted
        # window - but center_fail_dist_m makes that value configurable,
        # and raising it for a competition run would have silently
        # disabled blindness detection altogether, which is the one thing
        # that must not depend on how forgiving the fail-safe is set to
        # be. depth_profile covers the same columns and more, and the
        # ground band below is the real "is the camera alive" test.
        if self.depth_profile:
            valid_frac = sum(1 for d in self.depth_profile if d is not None) / len(self.depth_profile)
        else:
            valid_frac = 0.0
        # If the ground band still sees the floor, the camera is working and
        # this is a far/open scene rather than blindness - see
        # blind_ground_valid_frac. Unknown (never received, e.g. the
        # ground_hazard link not wired) deliberately does NOT veto, so
        # behaviour is unchanged where that signal is unavailable.
        ground_alive = (self.ground_valid_frac is not None
                        and self.ground_valid_frac >= self.blind_ground_valid_frac)
        blind_now = valid_frac < self.blind_profile_valid_frac and not ground_alive

        if blind_now:
            self.blind_streak += 1
            self.blind_clear_streak = 0
        else:
            self.blind_clear_streak += 1
            self.blind_streak = 0

        if not self.depth_blind_active and self.blind_streak >= self.blind_confirm_frames:
            self.depth_blind_active = True
            # a fresh blind stretch gets a fresh distance budget - see
            # blind_creep_max_dist_m
            self._blind_anchor_xy = None
            self._blind_budget_spent = False
            plan = ('creeping forward at %.2f m/s' % self.blind_creep_speed
                    if self.blind_creep_speed > 0 else 'holding still')
            print(self.time, 'depth camera is blind (centre invalid, %.0f%% of the profile unknown) - '
                              '%s until it recovers' % (100 * (1 - valid_frac), plan))
        elif self.depth_blind_active and self.blind_clear_streak >= self.blind_clear_frames:
            self.depth_blind_active = False
            self._blind_anchor_xy = None
            self._blind_budget_spent = False
            print(self.time, 'depth data recovered (%.0f%% of the profile valid) - resuming'
                   % (100 * valid_frac))

    def on_ground_hazard(self, data):
        """data is [hazard_bool, ground_valid_frac, ground_dist] for this
        frame from ObstacleDetector3DZones - the streak/debounce logic
        lives here, same pattern as stop_streak/turn_streak above, rather
        than in the sensing module. valid_frac/dist are only for
        diagnostics/logging, not part of the trigger decision itself."""
        # 4-element form since obstdet3d_zones grew min_ground_dist; older
        # logs and older sensing modules publish three, and must keep
        # replaying unchanged
        hazard, valid_frac, dist, *rest = data
        near = bool(rest[0]) if rest else False
        # Debounced separately from the drop-off streak below, and NOT
        # gated on enable_ground_hazard: this is a close-obstacle signal,
        # not a drop-off reaction, and the two switches mean different
        # things. See ground_near_confirm_frames in __init__.
        self.ground_near_streak = self.ground_near_streak + 1 if near else 0
        was_near = self.ground_near_active
        self.ground_near_active = self.ground_near_streak >= self.ground_near_confirm_frames
        if self.ground_near_active and not was_near:
            print(self.time, 'near obstacle on the ground band (%.2fm) - treating as a close '
                              'obstacle ahead' % dist)
        # kept regardless of enable_ground_hazard: the drop-off REACTION is
        # what that switch turns off, but the ground band's valid fraction
        # is also the blind detector's "is the camera alive" signal (see
        # blind_ground_valid_frac) and that must keep working either way
        self.ground_valid_frac = valid_frac
        if not self.enable_ground_hazard:
            return  # ObstacleDetector3DZones still computes/publishes it - just ignored here
        self.ground_hazard_streak = self.ground_hazard_streak + 1 if hazard else 0

        if self.ground_hazard_streak >= self.ground_hazard_confirm_frames:
            if not self.ground_hazard_active:
                print(self.time, 'possible drop-off/staircase detected - stopping',
                      'ground_valid_frac', valid_frac, 'ground_dist', dist)
            self.ground_hazard_active = True
            self.send_speed_cmd(0, 0)
            if self.terminate_on_ground_hazard:
                raise EmergencyStopException()
            if self.ground_hazard_retreat and self.state != State.BACKING_UP:
                # Stopping alone does not undo a drop-off. The robot got
                # here by driving forward and is now parked with its nose
                # over the edge; the ground band still reads "no floor", so
                # the hazard never clears and the old behaviour - hold
                # speed 0 and re-test the same unchanging view every cycle
                # - is a permanent freeze at the worst possible place. The
                # 2026-09-12 17:23 rollover is what happens when it is not
                # a freeze but a drive-through: ground_hazard was True for
                # 22 consecutive frames (2.87 s, ground band receding
                # 2.46 m -> 6.35 m) while the robot ACCELERATED 0.25 ->
                # 0.50 m/s, because enable_ground_hazard was false and
                # this whole handler returned early. With the detector
                # enabled and confirm_frames=3 the stop lands at t=133.55,
                # 2.96 s before the chassis passed 80 deg of roll.
                #
                # Retrace, not a blind straight reverse: the path just
                # driven is the one piece of ground known to hold the
                # robot up.
                print(self.time, 'drop-off confirmed - retreating along the path just driven')
                self._enter_backing_up(0, use_retrace=True)
        elif self.ground_hazard_streak == 0:
            if self.ground_hazard_active:
                print(self.time, 'ground hazard cleared, resuming')
            self.ground_hazard_active = False

    def _camera_offroad_frac(self):
        """0..1 - how much the CAMERA thinks the robot is leaving the road,
        from the road fraction in the strip straight ahead. The camera-side
        twin of _route_offroad_frac.

        Why it has to exist separately. _route_offroad_frac is the existing,
        already-measured answer to "the depth terms push off-corridor harder
        than the mask pushes back" - and it returns 0 outside route mode. On
        2026-09-12 the router was in state `free` for 80.9% of the session,
        so that mitigation never ran for four fifths of the driving,
        including every one of the grass excursions in run 16:57:47.

        This one needs no route, no plan and no GPS - only the mask the road
        follower is already computing. 0 (both bounds zero) disables it and
        the legacy path is untouched."""
        on, off = self.camera_offroad_on_frac, self.camera_offroad_off_frac
        if on <= off:
            return 0.0          # disabled, or bounds the wrong way round
        # road ahead >= on  -> 0 (believe the depth terms)
        # road ahead <= off -> 1 (they have no business steering here)
        return max(0.0, min(1.0, (on - self.road_ahead_frac) / (on - off)))

    def _road_lost_speed_cap(self):
        """Speed ceiling while the road mask has been blank long enough to
        count as lost - see on_nn_mask. None when it has not.

        This is the honest reaction to a long dropout, and it is what the
        smoothing CANNOT be asked to do. In run 16:57:47 the mask went
        completely blank at t=95.2 and stayed blank for 6.1 s while the app
        held +0.50 m/s and last_dir = 0.0 - because an empty mask falls back
        to the frame centre, which the steering reads as 'road dead ahead'.
        The robot covered about 3 m of lawn in a straight line on that
        reading, and finished 3.8 m off the mapped path."""
        if not self.road_lost:
            return None
        return self.road_lost_speed

    def _mask_fov_scale(self, width, height):
        """Factor putting this mask's horizontal offsets back into the
        field the road-following gain was tuned on.

        A mask does NOT always span the camera's field.
        Camera.requestOutput() defaults to ImgResizeMode.CROP, so the NN
        gets the largest centred region of the sensor with the aspect its
        input asks for:

            224x224 (1:1) -> full height, three quarters of the width
            640x480 (4:3) -> the whole sensor

        So when the 640x480 redroad-v2 blob replaced the 224x224
        robotourist one, the same road at the same real bearing started
        producing a SMALLER normalised offset - the field it is measured
        against grew from 54.6 to 69 degrees - and the steering built on
        it quietly lost a quarter of its authority. Nothing in the config
        would have shown that; the only visible change was the blob path.

        Working in tangents rather than degrees because that is what the
        pinhole projection is linear in: a point at bearing b sits at
        tan(b)/tan(hfov/2) across the half-frame, so the conversion
        between two fields is the ratio of their half-tangents. For the
        two blobs above that is 1.33, and for any 1:1 mask it is exactly
        1.0, which is why this needs no config change to reproduce the
        behaviour every earlier run was tuned with.

        Deliberately applied to the offset rather than to turn_angle:
        turn_angle is also the ceiling on the depth-based steering
        (_free_space_steering and the route limits), and those are
        measured in real angles already - widening them because the
        camera feeding a different sensor got wider would be wrong."""
        aspect = width / float(height)
        tan_full = math.tan(math.radians(self.camera_hfov_deg) / 2)
        sensor_aspect = 4.0 / 3.0
        if aspect >= sensor_aspect:
            tan_mask = tan_full                      # bound by the width
        else:
            tan_mask = tan_full * aspect / sensor_aspect   # bound by the height
        tan_ref = math.tan(math.radians(self.mask_reference_hfov_deg) / 2)
        scale = tan_mask / tan_ref if tan_ref > 0 else 1.0
        if self._mask_fov_reported != (width, height):
            self._mask_fov_reported = (width, height)
            print(self.time, 'road mask %dx%d spans %.1f deg horizontally, '
                             'steering offsets scaled by %.3f'
                  % (width, height, math.degrees(math.atan(tan_mask)) * 2, scale))
        return scale

    def on_nn_mask(self, data):
        mask = data.copy()  # never modify the shared buffer in place
        height, width = mask.shape
        mask[:height // 2, :] = 0  # ignore sky/horizon in the top half

        if self.mask_largest_blob:
            mask = _largest_blob(mask)

        road_frac = float(mask[height // 2:, :].mean())
        trusted = road_frac >= self.mask_min_frac and mask.max() > 0
        if self.mask_row_weighting:
            center_x = weighted_mask_center_x(mask, height // 2)
            center_y = height * 3 // 4  # only used for the viewer crosshair
        else:
            center_y, center_x = mask_center(mask)

        # Blend in an aim point read further up the frame - see
        # mask_aim_blend in __init__ for the measured trade-off. The bottom
        # band answers "how far off the path am I", the mid band answers
        # "where does the path go"; they are different questions and the
        # scoring says each wins a different metric, so this mixes rather
        # than replaces. 0 (default) leaves center_x exactly as before.
        if self.mask_aim_blend > 0:
            a0, a1 = (int(height * f) for f in self.mask_aim_rows)
            aim_x = _band_center_x(mask, a0, a1)
            if aim_x is not None:
                center_x = (1 - self.mask_aim_blend) * center_x + self.mask_aim_blend * aim_x
                center_y = (a0 + a1) // 2

        # An untrusted frame must not be read as "road dead ahead" - that is
        # what the frame-centre fallback silently means, and on a curve it
        # steers off the road. Reuse the last trusted centre instead, but
        # only for mask_hold_sec: a mask that has been blank for longer than
        # that is not a dropout, it is the robot no longer looking at a
        # road, and pretending otherwise would drive on stale information.
        self.mask_held = False
        if trusted:
            self._mask_last_good = (self.time, center_x)
        elif self.mask_hold_sec > 0 and self._mask_last_good is not None:
            held_at, held_x = self._mask_last_good
            if self.time is not None and \
                    (self.time - held_at).total_seconds() <= self.mask_hold_sec:
                center_x = held_x
                self.mask_held = True

        # EMA last, so it smooths whichever centre survived the gate above
        if self.mask_center_alpha < 1.0:
            if self._mask_center_ema is None:
                self._mask_center_ema = center_x
            else:
                self._mask_center_ema += self.mask_center_alpha * (center_x - self._mask_center_ema)
            center_x = self._mask_center_ema

        half = width / 2
        dead = (width // 16) / half  # same dead-zone width as before, as a fraction of half-width

        offset = (center_x - half) / half  # -1 (mask hugging left edge) .. +1 (right edge)
        # ... and back into the field the gain was tuned on, so swapping
        # the blob for one with a different input aspect does not silently
        # change the steering - see _mask_fov_scale
        offset *= self._mask_fov_scale(width, height)
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

        # --- is the strip the robot is about to enter actually road? ---
        # This is the useful half of the "anti-mask" idea (item: everything
        # that is not road should push Matty away). Measured over 47147
        # frames of the 2026-09-12 session, the repulsion FORM of that idea
        # - a signed push away from non-road pixels - was worse than the
        # plain centroid at the job that matters (sign agreement with the
        # mapped road direction 56.3% against 65.4%, and 63.8% against
        # 73.4% while already off the road), so it is not used to steer.
        #
        # Read as a VETO it earns its place, because it answers a question
        # nothing else on the robot answers without GPS: at a threshold of
        # 0.30 it fires on 19.1% of frames with 64.9% of those genuinely
        # more than 0.5 m off the mapped road. That is not good enough to
        # steer on and is plenty to stop HANDING THE DEPTH TERMS THE WHEEL
        # - see _camera_offroad_frac and _drive_steering.
        r0, r1 = int(height * 0.70), int(height * 0.85)
        c0, c1 = int(width * 0.40), int(width * 0.60)
        self.road_ahead_frac = float(mask[r0:r1, c0:c1].mean())
        if self.road_ahead_frac >= self.road_scout_min_frac:
            # the steering that pointed at the road while it was still
            # visible - where a road-lost scout goes looking (road_scout_m)
            self._road_last_dir = self.last_dir

        # road fraction inside each depth bin's bearing band, for judging grey
        # bins (see edge_grey_mode) - mapped through the same pinhole geometry
        # _mask_fov_scale uses, so a bin and its mask columns look the same way
        n_bins = len(self.depth_profile) if self.depth_profile else 0
        if n_bins:
            aspect = width / float(height)
            tan_mask = math.tan(math.radians(self.camera_hfov_deg) / 2)
            if aspect < 4.0 / 3.0:
                tan_mask *= aspect / (4.0 / 3.0)
            b0, b1 = (int(height * f) for f in self.mask_bin_rows)
            band = mask[b0:b1, :]
            half_fov = self.polar_profile_hfov / 2
            bins = []
            for i in range(n_bins):
                e0 = -half_fov + self.polar_profile_hfov * i / n_bins
                e1 = -half_fov + self.polar_profile_hfov * (i + 1) / n_bins
                c_lo = int((math.tan(e0) / tan_mask + 1) / 2 * width)
                c_hi = int((math.tan(e1) / tan_mask + 1) / 2 * width)
                c_lo, c_hi = max(0, min(width - 1, c_lo)), max(1, min(width, c_hi))
                sub = band[:, c_lo:c_hi]
                bins.append(float(sub.mean()) if sub.size else 0.0)
            self.profile_bin_road = bins
        # a branch leaving to either side shows up in the far band's outer
        # thirds - how much road is there, and WHERE it is. _arrow_turn_hint
        # steers toward that road's own centroid rather than by a fixed angle,
        # so on a wide road or plaza a planned turn moves Matty to that side of
        # the road it is already on, and on a narrow path it does nothing at
        # all until the branch is actually in the picture.
        f0, f1 = int(height * 0.50), int(height * 0.72)
        third = width // 3
        far_left, far_right = mask[f0:f1, :third], mask[f0:f1, width - third:]
        self.branch_road_left = float(far_left.mean())
        self.branch_road_right = float(far_right.mean())
        fov = self._mask_fov_scale(width, height)
        dead_zone = (width // 16) / (width / 2.0)

        def _dir_to(xs_image):
            if len(xs_image) == 0:
                return None
            off = (float(xs_image.mean()) - width / 2.0) / (width / 2.0) * fov
            if abs(off) <= dead_zone:
                return 0.0
            return -math.copysign(min(1.0, (abs(off) - dead_zone) / (1 - dead_zone)) * self.turn_angle, off)
        self.branch_dir_left = _dir_to(np.nonzero(far_left)[1])
        self.branch_dir_right = _dir_to(np.nonzero(far_right)[1] + (width - third))

        # --- have we lost the road entirely, and for how long? ---
        # The dropout the driver notices is 1-2 frames long, and that is
        # genuinely the median (2 frames). But it is not where the time
        # goes: of 425 dropout episodes, those longer than 1 s account for
        # 80.8% of all blank frames, and the longest ran 17.4 s. No filter
        # can help there - there is no road in the picture to smooth - and
        # extrapolating a turn through 17 s of nothing is worse than
        # admitting the road is gone. See _road_lost_speed_cap.
        self.road_blank_streak = 0 if trusted else self.road_blank_streak + 1
        was_lost = self.road_lost
        self.road_lost = (self.road_lost_frames > 0
                          and self.road_blank_streak >= self.road_lost_frames)
        if self.road_lost and not was_lost:
            print(self.time, 'road mask blank for %d frames - road lost, capping speed at %.2f m/s'
                  % (self.road_blank_streak, self.road_lost_speed))
        elif was_lost and not self.road_lost:
            print(self.time, 'road mask back after %d blank frames' % self.road_blank_streak)
        if self.road_lost:
            if self.road_lost_since is None:
                self.road_lost_since = self.time
        else:
            self.road_lost_since = None
        self.road_found_streak = self.road_found_streak + 1 if trusted else 0
        if self.road_found_streak >= 50:
            self._road_retreats = 0       # 5 s of road again - a new episode may retreat afresh
            self._road_hold_retries = 0

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

        if self._forced_turn_sign is not None:
            # the caller already decided (see _only_clear_side_is_grass)
            pick, self._forced_turn_sign = self._forced_turn_sign, None
            self._turn_sign_committed = True
            self.last_turn_sign_choice = pick
            return pick
        if self._turn_sign_committed:
            current_side = left if self.turn_sign > 0 else right
            other_side = right if self.turn_sign > 0 else left
            if other_side > current_side + self.turn_sign_switch_margin_m:
                return -self.turn_sign
            return self.turn_sign
        self._turn_sign_committed = True

        left_clear = left > self.turning_dist
        right_clear = right > self.turning_dist
        around = self._around_obstacle_side(left, right)
        if around is not None:
            # something is in the robot's own swept width - go round it on
            # the side it is NOT on, which is the shortest way clear
            pick = self._road_frac_veto(around, left, right)
        elif left_clear and not right_clear:
            pick = 1
        elif right_clear and not left_clear:
            pick = -1
        elif (left_clear and right_clear and self.turn_clearance_margin_m > 0
              and abs(left - right) > self.turn_clearance_margin_m
              and (self.turn_clearance_comfort_m <= 0
                   or min(left, right) < self.turn_clearance_comfort_m)):
            # both passable, one clearly more so - "clear" is judged at
            # turning_dist, which threw away the difference between 1.6m
            # and 2.3m and let the road mask decide (172130 t=34.6).
            #
            # Only while the tighter side is still tight, though. Extra
            # metres stop buying anything once both sides are comfortably
            # wider than the robot, and the cost of ignoring the mask
            # does not go away with them: at 175010 t=82.8 this preferred
            # 4.02m over 3.44m and turned away from the side the mask
            # liked - 3.44m was never the constraint. Past
            # turn_clearance_comfort_m the question stops being "which
            # side fits" and becomes "which side is road", which is the
            # mask's to answer.
            pick = self._road_frac_veto(1 if left > right else -1, left, right)
        elif self._road_side_preference() is not None:
            # Both sides physically passable - let the MAP break the tie
            # before the road mask gets to (below). At Robotour leaving
            # the road ends the run, and this decision commits the robot
            # to driving that way for seconds at avoid_speed, which makes
            # it the single most expensive place to get the side wrong.
            #
            # It was being decided without the map at all: measured over
            # the 2026-09-05 runs, of 133 avoidance turns started while
            # already more than 0.5m off the planned centreline, 62 (47%)
            # turned FURTHER off - a coin flip. The road mask cannot fix
            # this either; measured on cycles already off the road, it
            # reports MORE drivable surface on the outward side (0.253)
            # than the inward one (0.217), because that is exactly when
            # it is calling the grass a road.
            #
            # Only ever used as a TIE-BREAK between two sides depth has
            # already called passable, so it cannot steer the robot into
            # anything - and _bin_override_turn_sign still gets the last
            # word below if the bins disagree strongly.
            pick = self._road_side_preference()
        # both clear, or both blocked by depth alone - break the tie
        # using whichever side still looks more like road
        elif abs(self.left_road_frac - self.right_road_frac) > 0.02:
            pick = 1 if self.left_road_frac > self.right_road_frac else -1
        else:
            # still tied - fall back to raw clearance, default left
            pick = 1 if left >= right else -1
        if around is None:
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

    def _only_clear_side_is_grass(self):
        """True when depth calls exactly one side passable and the mask says
        that side is not road while the other one is - see
        avoid_road_side_veto in __init__. Returns (True, road_side) or
        (False, 0)."""
        if not self.avoid_road_side_veto or self.turn_side_min_road_frac <= 0:
            return False, 0
        left = self.left_dist if self.left_dist is not None else float('inf')
        right = self.right_dist if self.right_dist is not None else float('inf')
        left_clear = left > self.turning_dist
        right_clear = right > self.turning_dist
        if left_clear == right_clear:
            return False, 0                 # both or neither - the usual logic decides
        clear_frac = self.left_road_frac if left_clear else self.right_road_frac
        other_frac = self.right_road_frac if left_clear else self.left_road_frac
        if (clear_frac < self.turn_side_min_road_frac
                and other_frac > clear_frac + self.turn_side_road_frac_margin):
            return True, (-1 if left_clear else 1)
        return False, 0

    def _road_side_has_gap(self, side):
        """Room past the obstacle on that side: an outermost depth bin that
        reads far, or reads nothing at all. A bin with no return is open
        ground as far as this is concerned - same reading as
        edge_grey_mode 'open', and it is what a thin post looks like from
        the side (161958: bins at +23 and +31 deg empty while the post sat
        at +15 deg, 1.2 m)."""
        profile = self.depth_profile or []
        if len(profile) < 3:
            return False
        edge = profile[-2:] if side < 0 else profile[:2]
        return any(d is None or d > self.turning_dist * 1.5 for d in edge)

    def _profile_bin_bearing(self, i, n):
        """Camera bearing of profile bin i, radians, positive to the
        robot's RIGHT. Same mapping _update_polar_memory already uses -
        bin 0 is the leftmost columns and polar_profile_hfov is the field
        the whole profile spans - kept in one place so the two cannot
        drift apart."""
        frac = (i + 0.5) / n * 2 - 1        # -1 (left edge) .. +1 (right edge)
        return frac * (self.polar_profile_hfov / 2)

    def _corridor_scan(self):
        """Closest FORWARD distance to anything inside the corridor the
        robot is about to sweep, and which side of the centreline it sits
        on: (forward_m, lateral_m, bearing_rad) or None.

        See corridor_check in __init__ for why the L/C/R zones cannot
        answer this. Each profile bin is a range at a known bearing, so
        the decomposition is exact:

            lateral = d * sin(bearing)      how far off the centreline
            forward = d * cos(bearing)      how far ahead

        and a bin counts only when |lateral| is inside the robot's own
        half-width plus a margin. Everything wider is genuinely passable
        and is left to the side zones and the edge nudge, which is what
        keeps a doorway a doorway: a frame 0.4m off the centreline is
        outside the corridor and contributes nothing here, exactly as
        before.

        Unknown bins (None) are skipped rather than treated as blocked -
        the opposite of _edge_bin_correction's "assume worst" bias, and
        deliberately so. This feeds a hard stop, and a bin goes unknown
        far too often (sun on a glossy surface, sky, a low-texture wall)
        for missing data to be allowed to stop the robot outright; the
        centre zone's own fail_value already owns that decision. This
        channel only ever reports something it actually MEASURED."""
        if not self.corridor_check or not self.depth_profile:
            return None
        n = len(self.depth_profile)
        if n == 0:
            return None
        half = self.corridor_half_width_m + self.corridor_margin_m
        far_m = self.polar_memory_far_fill_m
        best = None
        for i, d in enumerate(self.depth_profile):
            if d is None or d <= 0:
                continue
            if far_m and d >= far_m:
                continue        # far-mask fill, not a measurement
            bearing = self._profile_bin_bearing(i, n)
            lateral = d * math.sin(bearing)
            if abs(lateral) > half:
                continue
            forward = d * math.cos(bearing)
            if best is None or forward < best[0]:
                best = (forward, lateral, bearing)
        return best

    def _road_side_preference(self):
        """+1 (left) / -1 (right) for the side that leads back toward the
        mapped way, or None when the robot is close enough to the planned
        line for the question not to arise.

        Deliberately gated on a cross-track well past the noise: the
        SIGN of cross_track is reliable, but only once its magnitude is
        clearly outside the metre-scale error in the fix and the mapped
        centreline - see _damp_outward for why the magnitude itself
        cannot be trusted for finer decisions."""
        if self.turn_road_side_min_cross_m <= 0 or self.route_cross_track_m is None:
            return None
        if abs(self.route_cross_track_m) < self.turn_road_side_min_cross_m:
            return None
        # cross_track > 0 = robot right of the route, so the way back is
        # left, which is positive steering (see _route_corridor_bias)
        return 1 if self.route_cross_track_m > 0 else -1

    def _corridor_hold_dist(self):
        """Smallest corridor reading over the last corridor_hold_frames -
        see corridor_hold_frames in __init__. None when nothing has been
        measured in the corridor recently."""
        vals = [d for d in self._corridor_history if d is not None]
        return min(vals) if vals else None

    def _around_obstacle_side(self, left, right):
        """+1/-1 to pass the closest in-corridor object on the side it is
        NOT on, or None when there is no such object, it is dead-centre,
        it is still far off, or that side is itself blocked."""
        if not self.turn_around_obstacle:
            return None
        hit = self._corridor_scan()
        if hit is None:
            return None
        fwd, lat, _bearing = hit
        # turning_dist, not the free_space lead - this overrules the road
        # mask, so it may only speak for an object close enough to be the
        # reason the robot is turning at all. Measured on 2026-09-11: the
        # lead-margin reach (1.98m) let objects at 1.6-1.9m decide the
        # side while the centre zone was reading 1.5-2.0m clear, i.e. the
        # turn had been triggered by a SIDE zone and this was answering
        # about something else entirely - 5 of the 10 picks that went
        # against the mask came from exactly that. Every genuine close-in
        # case in those runs (0.49-0.88m, the poles this was built for)
        # is well inside turning_dist and unaffected.
        if fwd > self.turning_dist * self.turn_around_lead_factor or abs(lat) < self.turn_around_min_lateral_m:
            return None
        away = 1 if lat > 0 else -1              # object on the right -> go left
        room = left if away > 0 else right
        return away if room > self.turning_dist else None

    def _road_frac_veto(self, pick, left, right):
        """Refuse a side the road mask says is essentially NOT road, when
        the other side clearly is and is itself passable. Returns the
        side to actually turn toward (pick, or -pick when vetoed).

        Guards the two branches above that decide on DEPTH ALONE - the
        in-corridor "go round it" pick and the "one side is clearly more
        open" pick. Both were added for the 2026-09-11 pole incidents and
        both sit BEFORE the left_road_frac/right_road_frac tie-break, so
        without this the mask has no say at all in those cases. Measured
        over the 2026-09-11 test5 logs with the current config: those two
        branches decide 27 of 40 avoidance side choices, and 12 of the 40
        picks land on the side the mask reports as LESS road. That is not
        a detail - one avoidance maneuver displaces the robot sideways by
        a median 0.36m, p90 0.80m, max 1.16m (measured from the same
        runs' odometry), which on a 0.7m path is the whole road. At
        Robotour leaving the road ends the run, so a side choice made
        without the mask is a real way to lose one.

        Why "essentially no road" and not simply "less road": the depth
        branches exist because the mask was picking badly. At 172130
        t=34.6 the mask preferred the side the pole was on (0.304 vs
        0.096) and that is exactly the loop the user reported. A veto on
        any mask disagreement would hand that failure straight back. So
        this only fires where the mask is not expressing a preference but
        a near-absence: the chosen side below turn_side_min_road_frac
        while the other is ahead by turn_side_road_frac_margin. On the
        2026-09-11 logs at 0.05/0.10 that is 1 of 40 decisions (174824
        t=55.2, left 0.015 vs right 0.135) - a backstop, not a change of
        policy.

        The flip is also required to be physically possible: the other
        side must read past turning_dist, the same test the depth
        branches themselves use. A side the depth zones call blocked is
        never chosen here, whatever the mask thinks of it - vetoing a
        turn must never become a way to steer into something.

        Both fracs default to 0.5 before any mask has arrived, so with no
        mask the difference is 0 and this cannot fire."""
        if self.turn_side_min_road_frac <= 0:
            return pick
        picked_frac = self.left_road_frac if pick > 0 else self.right_road_frac
        other_frac = self.right_road_frac if pick > 0 else self.left_road_frac
        if picked_frac >= self.turn_side_min_road_frac:
            return pick
        if other_frac - picked_frac < self.turn_side_road_frac_margin:
            return pick        # mask sees little road either way - no opinion
        other_room = right if pick > 0 else left
        if other_room <= self.turning_dist:
            return pick        # the only alternative is blocked - depth wins
        print(self.time, 'road mask says the %s side is not road (%.3f vs %.3f) '
                          '- avoiding to the %s instead'
              % ('left' if pick > 0 else 'right', picked_frac, other_frac,
                 'right' if pick > 0 else 'left'))
        return -pick

    def _corridor_repulsion(self):
        """Steering (radians, +left) that moves an object in the robot's
        own swept width OUT of it, starting as soon as it is inside the
        distance the bins treat as "open".

        The continuous depth steering had no term that pushes AWAY from
        anything. _free_space_steering only pulls TOWARD open bins, and a
        single blocked bin in the middle of an otherwise open profile
        leaves that weighted average dead centre; _edge_bin_correction
        only watches the outermost bin on each side. So a pole that sits
        in the interior bins produces no steering at all until the
        discrete maneuver fires. Field case, 2026-09-11 run 172130
        t=33.0-33.4: the pole moved from bin 0 into bins 1-3 at 0.6m, the
        edge term lost it (bin 0 now read 2.2m, past the pole), and the
        commanded steering over the last 1.5s was -0.4, -6.2, -6.2 deg -
        straight on - before the backup.

        Geometric rather than tuned: the object has to move by
        (half-width + margin - |lateral|) sideways within `forward`
        metres, and the command is proportional to that required angle.
        Away from the side it is on; near dead-centre the previous choice
        is held (hysteresis) and otherwise the side with more open bins
        wins."""
        if self.corridor_repel_gain <= 0:
            return 0.0
        hit = self._corridor_scan()
        if hit is None:
            self._repel_side = 0
            return 0.0
        fwd, lat, _bearing = hit
        if fwd >= self.turning_dist * self.free_space_lead_margin:
            self._repel_side = 0
            return 0.0
        half = self.corridor_half_width_m + self.corridor_margin_m
        shift = max(0.0, half - abs(lat))
        if abs(lat) > self.corridor_repel_centre_m:
            side = 1 if lat > 0 else -1          # object on the right -> steer left
        elif self._repel_side:
            side = self._repel_side
        else:
            side = 1 if self._bin_open_frac(1) >= self._bin_open_frac(-1) else -1
        self._repel_side = side
        mag = min(self.edge_correction_max_rad,
                  self.corridor_repel_gain * math.atan2(shift, max(0.05, fwd)))
        return side * mag

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

        n_prof = len(self.depth_profile)

        def worst(edge, first_index):
            # See edge_grey_mode / edge_lateral_max_m in __init__. Returns inf
            # when nothing on this edge counts, which reads as "not blocked".
            vals = []
            for k, d in enumerate(edge):
                i = first_index + k
                if d is None:
                    if self.edge_grey_mode == 'open':
                        continue
                    if (self.edge_grey_mode == 'mask' and i < len(self.profile_bin_road)
                            and self.profile_bin_road[i] >= self.edge_grey_road_frac):
                        continue      # grey over visible road: far-fill artefact, not an obstacle
                    vals.append(0.0)  # grey over no road: could be a textureless wall - keep blocking
                    continue
                if self.edge_lateral_max_m > 0:
                    lateral = d * abs(math.sin(self._profile_bin_bearing(i, n_prof)))
                    if lateral > self.edge_lateral_max_m:
                        continue      # passes wide of the robot - not a flank threat
                vals.append(d)
            return min(vals) if vals else float('inf')
        left_worst = worst(left_edge, 0)
        right_worst = worst(right_edge, n_prof - self.edge_bins_watched)
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

    def _blind_command(self, xy, dt):
        """(speed, steering) to use while the depth camera is blind - see
        the blind-behaviour notes in __init__. Returns a dead stop unless
        blind_creep_speed is set AND creeping is currently allowed."""
        if self.blind_creep_speed <= 0:
            return 0.0, 0.0
        if self.state != State.DRIVE:
            # blind partway through an avoidance maneuver: whatever
            # triggered it was real, and it is still out there
            return 0.0, 0.0
        if self._blind_anchor_xy is None:
            self._blind_anchor_xy = xy
        if self.blind_creep_max_dist_m > 0:
            travelled = math.hypot(xy[0] - self._blind_anchor_xy[0],
                                   xy[1] - self._blind_anchor_xy[1])
            if travelled >= self.blind_creep_max_dist_m:
                if not self._blind_budget_spent:
                    self._blind_budget_spent = True
                    print(self.time, 'blind creep budget spent (%.1fm with no usable depth) - '
                                      'stopping until it recovers' % travelled)
                return 0.0, 0.0
        # steering degrades to road mask + GPS bearing by itself here -
        # _depth_confidence() is 0 with an all-unknown profile, so the
        # depth term drops out of _drive_steering's blend
        return min(self.blind_creep_speed, self.max_speed), self._drive_steering(dt)

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

    def _update_flank_memory(self, xy):
        """Remember the closest thing seen on each flank recently - see the
        tail-swing notes in __init__. Expires on BOTH age and travel: this
        is a statement about the robot's immediate surroundings, and it
        stops being true once the robot has driven past."""
        if not self.tail_swing_enabled:
            return
        h = self._odom_heading
        for side, dist in ((1, self.left_dist), (-1, self.right_dist)):
            history = self._flank_history[side]
            if dist is not None:
                # where the thing IS, in the odometry frame, so it can be
                # re-expressed relative to wherever the robot has moved to
                a = h + side * self.side_zone_bearing
                point = (xy[0] + dist * math.cos(a), xy[1] + dist * math.sin(a))
                history.append((self.time, xy, dist, point))
            while history and (self.time - history[0][0] > self.flank_memory_time
                                or math.hypot(xy[0] - history[0][1][0],
                                              xy[1] - history[0][1][1]) > self.flank_memory_dist_m):
                history.popleft()

    def _flank_clearance(self, side):
        """Closest thing known on one flank (+1 left, -1 right): the smaller
        of what the zone sees now and what was seen recently. The memory is
        the whole point - nothing on this robot looks sideways or back, so
        once the robot draws level with something the live zone has already
        lost it."""
        live = self.left_dist if side > 0 else self.right_dist
        remembered = min((d for _t, _xy, d, _p in self._flank_history[side]), default=None)
        known = [d for d in (live, remembered) if d is not None]
        return min(known) if known else None

    def _lateral_clearance(self, side):
        """Flank RANGE converted to the lateral offset that actually
        matters for the swing - see side_zone_bearing in __init__."""
        if not self.swing_beside_only:
            rng = self._flank_clearance(side)
            return None if rng is None else rng * math.sin(self.side_zone_bearing)
        # Each remembered point re-expressed in the body frame NOW: only
        # those level with the body (not still ahead, not long passed) on
        # the requested side can be swept by the rear corner.
        xy, h = self._last_xy, self._odom_heading
        c, sn = math.cos(h), math.sin(h)
        best = None
        for history in self._flank_history.values():
            for _t, _xy, _d, pt in history:
                vx, vy = pt[0] - xy[0], pt[1] - xy[1]
                fwd = vx * c + vy * sn
                lat = -vx * sn + vy * c          # +ve = left
                if fwd > self.swing_forward_max_m or fwd < -self.swing_behind_max_m:
                    continue
                if (lat > 0) != (side > 0):
                    continue
                if best is None or abs(lat) < best:
                    best = abs(lat)
        return best

    def _articulated_steering_limit(self, steering_sign):
        """Largest |steering| (radians) this chassis can be asked for
        without sweeping something beside it, or None for no limit.

        TWO separate hazards, and they are on OPPOSITE sides:

        INSIDE the turn - turning TOWARD something close. The whole robot
            curves that way, and the REAR, still level with the obstacle
            after the front has passed it, is carried into it. This is the
            reported incident: a pole passed on the left, the camera lost
            it, the right zone read clear, Matty turned hard right and the
            rear-right wheel caught the pole and started to climb it -
            loading the single central articulation joint in exactly the
            way it should never be loaded. Encroachment over a body length
            L at radius R is about L^2/(2R), and R = a/tan(gamma/2) with
            a = half the wheelbase, so it grows as tan(gamma/2).

        OUTSIDE the turn - the rear overhang swinging out the other way,
            the ordinary long-vehicle tail swing, growing as sin(gamma).

        Both are measured on the logs, on hard-steer forward cycles:
                            inside <0.8m   outside <0.8m
            TURNING             63.4%          67.9%
            REALIGNING          62.5%          41.8%
        and on the inside the camera could still see it in only 33% of
        cases (7.8% while REALIGNING). So the limit is the tighter of the
        two - guarding only one side would leave the reported failure
        wide open, and I had it on the wrong side until the field report
        said which wheel actually hit.

        Positive steering is LEFT, so the inside of the turn is the left
        flank for positive steering."""
        if not self.tail_swing_enabled:
            return None
        inside = 1 if steering_sign > 0 else -1
        limits = [self._corner_swing_limit(self._lateral_clearance(inside)),
                  self._corner_swing_limit(self._lateral_clearance(-inside))]
        limits = [l for l in limits if l is not None]
        return min(limits) if limits else None

    def _corner_swing_limit(self, clearance_m):
        """Steering ceiling (radians) from the rear outer corner sweeping
        sideways, given the LATERAL clearance beside the robot.

        Measured chassis (2026-09-05): the joint sits at the centre of the
        14.5cm gap between two 17.5cm boxes, the rear bumper is 3cm past
        the rear box, and the wheels take the total width to 35.5cm. So

            joint -> rear bumper   L = 0.2775 m
            half width             W = 0.1775 m
            joint -> rear corner   r = hypot(L, W) = 0.329 m
            corner off the axis    phi0 = atan(W/L) = 32.6 deg

        Articulating by gamma rotates the rear body about that joint, so
        the corner's lateral extent from the centreline is

            extent(gamma) = r * sin(phi0 + gamma)

        which is exactly W at gamma=0 (the corner IS the body edge there)
        and peaks at r when phi0+gamma = 90 deg, i.e. 57 deg of
        articulation. Beyond the body edge that is 0.152 m at the peak and
        0.144 m at the platform's own 45 deg limit - not a rounding error,
        which is why a pole passed cleanly by the front can still be
        caught by the rear wheel.

        Inverting: gamma <= asin((clearance - margin)/r) - phi0.

        Applied to BOTH flanks (see _articulated_steering_limit). Exactly
        which corner swings which way depends on how the articulation
        change is shared between front and rear wheels, which depends on
        grip and is not something this code can know; the symmetric,
        conservative reading is the honest one and costs little, since the
        limit only binds within about half a metre."""
        if clearance_m is None or self.rear_corner_radius_m <= 0:
            return None
        room = clearance_m - self.swing_margin_m
        if room >= self.rear_corner_radius_m:
            return None                      # geometry cannot reach that far
        if room <= self.rear_corner_radius_m * math.sin(self.rear_corner_angle):
            # already inside the static body width plus margin - nothing to
            # do but keep the floor so the robot can still manoeuvre
            return self.swing_min_steering
        limit = math.asin(room / self.rear_corner_radius_m) - self.rear_corner_angle
        return max(self.swing_min_steering, limit)

    def _limit_tail_swing(self, steering):
        """Clamp a commanded steering angle to what the chassis has room for.
        Only ever reduces the magnitude, never flips the sign or adds
        steering - so it cannot invent a maneuver, only soften one."""
        limit = self._articulated_steering_limit(steering)
        if limit is None or abs(steering) <= limit:
            return steering
        return math.copysign(limit, steering)

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
        # The corridor scan is the only channel that sees a thin object in
        # the robot's own swept width (see _corridor_scan), and it was
        # feeding the TRIGGERS while the speed was still being set by the
        # L/C/R zones alone. Field case, 2026-09-06 run 122857 t=45-47:
        # the corridor read 0.35m while C/L/R read 2.3-3.8m, so
        # _adaptive_speed saw no reason to slow and REALIGNING drove into
        # a pole at the full 0.50 m/s - the reproduced rollover. Speed
        # must be set by the closest thing that can actually be hit.
        corridor = self._corridor_hold_dist()
        clearance = min(self.last_obstacle, l_dist, r_dist, self.speed_clearance_ceiling)
        if corridor is not None:
            clearance = min(clearance, corridor)
        frac = clearance / self.speed_clearance_ceiling if self.speed_clearance_ceiling > 0 else 1.0
        target = max(self.min_speed, min(self.max_speed, self.min_speed + frac * (self.max_speed - self.min_speed)))
        target = min(target, self._offroad_speed_cap())
        # ...and by "I cannot see the road at all any more", which is a
        # different question from both obstacle clearance and map position,
        # and the only one of the three that was answerable during the
        # 6.1 s blind run onto the lawn in 16:57:47 - see _road_lost_speed_cap.
        lost_cap = self._road_lost_speed_cap()
        if lost_cap is not None:
            target = min(target, lost_cap)
        return self._rate_limit_speed(target, dt)

    def _offroad_speed_cap(self):
        """Speed ceiling from how far off the mapped road the robot is -
        the corridor's counterpart to the obstacle-clearance term above,
        and the reason this method now takes a min() of two independent
        questions instead of answering only one.

        Until now speed came exclusively from obstacle clearance, so the
        robot held max_speed while drifting onto a lawn: measured over the
        2026-09-01 runs it spent 444s more than 1.5m off the corridor at a
        mean 0.44 m/s, steering back at a net couple of degrees. Speed is
        what turns a steering error into METRES of grass, so leaving it
        untouched meant the one quantity that decides the cost of the
        failure was the one quantity nothing regulated.

        Ramps from max_speed at offroad_full_m down to offroad_speed at
        offroad_none_m, and is pinned at offroad_speed past
        offroad_hold_m. Only ever a CAP - it cannot make the robot go
        faster than the obstacle logic allows, and an avoidance maneuver
        running at its own fixed speed never calls this at all."""
        offroad = self._route_offroad_frac()
        hold = (self.offroad_hold_m > 0 and self.route_off_road_m is not None
                and self.route_off_road_m > self.offroad_hold_m)
        if hold != self.offroad_hold_active:
            self.offroad_hold_active = hold
            if hold:
                print(self.time, 'OFF THE ROAD by %.1fm (past offroad_hold_m %.1fm) - crawling at '
                                  '%.2f m/s with the route steering back'
                       % (self.route_off_road_m, self.offroad_hold_m, self.offroad_speed))
            else:
                print(self.time, 'back within %.1fm of a mapped road - releasing the off-road crawl'
                       % self.offroad_hold_m)
        if hold:
            return self.offroad_speed
        if offroad <= 0:
            return self.max_speed
        return self.max_speed + offroad * (self.offroad_speed - self.max_speed)

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
        rate = self.max_steering_rate
        if self.route_mode and self.route_steer_rate and (
                self.route_rate_boost or self.route_guidance_style != 'arrow'):
            # A junction turn has to be wound on inside the junction. At
            # the cruising 30deg/s default, reaching 40deg takes 1.3s -
            # 0.65m at cruising speed, most of the way across the fork -
            # so the router raises this while approaching one. Only ever
            # taken as a RELAXATION (max), never a tightening: the
            # smoothness this limit exists to protect is a property of
            # ordinary cruising, and the router has no business making
            # cruising twitchier than it is configured to be.
            rate = max(rate, self.route_steer_rate)
        max_delta = rate * dt.total_seconds()
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
        if self.heading_est is not None:
            # heading_source 'odometry_gps' - see __init__. Never the compass.
            estimate = self.heading_est.heading()
            if estimate is not None:
                return estimate, 'odo+gps'
            course = self._fresh_gps_course()
            if course is not None:
                return course, 'gps-course'
            return None, 'none'
        if self.use_compass_heading and self.compass_offset is not None:
            raw_compass = self._compass_heading()
            if raw_compass is not None:
                return self._apply_compass_calibration(raw_compass), 'compass'
        # receiver's own course over ground - preferred over fix
        # differencing, which it beats on steadiness by roughly 3x and
        # needs no baseline, no straight-line assumption and no reverse
        # gate (item 26). Only available while actually moving, hence the
        # travel_heading fallback below rather than a replacement.
        course = self._fresh_gps_course()
        if course is not None:
            return course, 'gps-course'
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
            return self._apply_compass_calibration(
                normalize_angle(math.pi / 2 - self.compass_sign * imu_heading))
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
        # In route mode target_dist is the distance to a rolling aim point
        # a lookahead ahead (~8m), not to the destination - feeding that in
        # here would fade the bearing out permanently, for the whole run,
        # on a route that is nowhere near finished. The honest measure of
        # "how close am I to the end" is the router's remaining distance
        # ALONG the route, which is what this uses instead.
        dist = self.route_remaining_m if self.route_mode else self.target_dist
        if dist is None or self.bearing_near_target_dist_m <= self.waypoint_arrival_dist_m:
            return 1.0
        span = self.bearing_near_target_dist_m - self.waypoint_arrival_dist_m
        return max(0.0, min(1.0, (dist - self.waypoint_arrival_dist_m) / span))

    def _route_heading_error(self):
        """How far the robot is pointing off the mapped road axis, in
        radians, positive when the robot faces clockwise (right) of the
        road. None when either side of the comparison is unknown."""
        if not self.route_mode or self.route_road_bearing is None:
            return None
        current_heading, _source = self._current_heading()
        if current_heading is None:
            return None
        return normalize_angle(current_heading - self.route_road_bearing)

    def _mask_trust(self):
        """0..1 weight on the road mask, from how far the camera is
        pointing off the mapped road axis AND how far the robot has
        drifted sideways off the mapped way - see mask_trust_full_deg and
        mask_trust_cross_full_m in __init__. 1.0 (unchanged) outside route
        mode or with no road axis known, so this cannot affect the legacy
        path. The two fades are combined by taking the smaller, not by
        multiplying: each is a sufficient reason on its own to stop
        believing the mask, and multiplying would let two moderate doubts
        compound into a certainty neither one earns."""
        trust = 1.0
        error = self._route_heading_error()
        if error is not None:
            error_deg = abs(math.degrees(error))
            if error_deg > self.mask_trust_full_deg:
                span = self.mask_trust_none_deg - self.mask_trust_full_deg
                if span <= 0:
                    trust = self.mask_trust_min
                else:
                    fade = min(1.0, (error_deg - self.mask_trust_full_deg) / span)
                    trust = 1.0 - fade * (1.0 - self.mask_trust_min)
        # Heading error alone does not describe the failure that actually
        # cost the 2026-09-05 runs. Driving PARALLEL to the road but
        # metres to the side of it has near-zero heading error, so the
        # mask kept full trust for 444s (15% of the session) while the
        # robot tracked a lawn: over those cycles the mask contributed a
        # net 0.85deg AWAY from the corridor, the free-space term 1.6deg
        # away and the edge term 4.8deg away, against 6.6deg of restoring
        # bias - a net 2.2deg, which at 0.4m/s needs about 160 seconds to
        # recover 2.5m. That is the reported "drove 2m off the road,
        # parallel, for a very long time".
        #
        # The mask is not wrong, it is answering a different question: it
        # is a lane-KEEPING sensor with no notion of WHICH lane, and
        # metres off a mapped way on open grass it will happily keep the
        # verge it is centred on. Cross-track is the only signal that
        # knows the difference, so it belongs here too.
        # 0 disables (old behaviour: heading error only).
        offroad = self._route_offroad_frac()
        if offroad > 0:
            trust = min(trust, 1.0 - offroad * (1.0 - self.mask_trust_min))
        return trust

    def _damp_outward(self, depth_steering):
        """Damp only the half of the depth nudge that points FURTHER from
        the planned way. Asymmetric on purpose - it never pulls inward, it
        only declines to push outward, so it cannot fight the robot's
        legitimate use of the road's width.

        Why this shape, and not the obvious "hold the centreline harder".
        Under Robotour rules leaving the road ends the run, so the
        temptation is to raise route_cross_track_gain until the robot hugs
        the mapped centreline. Measured, that is wrong: restricted to
        cycles PROVABLY on a mapped road (off_road_m == 0) the cross-track
        still runs p50 0.43m, p75 0.81m, p90 1.37m, and 38% of all forward
        driving is on a road while more than 0.5m off the planned line.
        Most of that is not control error - it is road width plus a
        centreline known to about a metre. A gain high enough to remove it
        would spend the run steering toward a line whose position is not
        known that well, and on a 2.4m footway that pushes the robot off
        the far side.

        What CAN be removed is the systematic outward bias in the signals
        the robot controls. Measured over on-road cycles more than 0.3m
        off-centre, the local terms push AWAY from the centreline every
        cycle: edge bins +0.77deg, free-space +0.53deg, mask +0.43deg,
        +1.73deg in total. The depth half of that is the part with no
        defence: "more open" beside a path is the lawn, and unlike the
        mask it is not even trying to answer "is this a road".

        So the outward component fades with how far off-centre the robot
        already is, and is restored in full by proximity - at urgency 1
        (a flank down at stop_dist) this returns depth_steering
        untouched. An obstacle can always still be avoided outward; only
        a PREFERENCE for the open side is suppressed.

        Direction, not magnitude, is what this keys off - the sign of
        cross_track survives the metre-scale error that makes its
        magnitude unusable for centring."""
        if self.outward_damp_start_m <= 0 or self.route_cross_track_m is None:
            return depth_steering
        cross = self.route_cross_track_m
        # positive steering is LEFT; cross_track > 0 means the robot is
        # RIGHT of the route, so steering further right (negative) is
        # outward. outward_sign is the steering sign that increases |cross|
        outward_sign = -1.0 if cross > 0 else 1.0
        if depth_steering * outward_sign <= 0:
            return depth_steering            # pointing back - never damped
        span = max(1e-6, self.outward_damp_full_m - self.outward_damp_start_m)
        excess = (abs(cross) - self.outward_damp_start_m) / span
        damp = max(0.0, min(1.0, excess)) * self.outward_damp_max
        if self.edge_correction_max_rad > 0:
            urgency = min(1.0, abs(self._edge_bin_correction()) / self.edge_correction_max_rad)
            damp *= 1.0 - urgency
        return depth_steering * (1.0 - damp)

    def _route_offroad_frac(self):
        """_route_offroad_frac_gps, capped by _camera_offroad_frac when
        route_offroad_camera_gate is on: a biased fix alone no longer slows
        Matty or discounts the mask while the camera sees road ahead."""
        gps = self._route_offroad_frac_gps()
        if self.route_offroad_camera_gate and gps > 0:
            return min(gps, self._camera_offroad_frac())
        return gps

    def _route_offroad_frac_gps(self):
        """0..1 - how far the robot has drifted off the mapped way, as a
        fraction of "still on it" (mask_trust_cross_full_m) to "certainly
        not" (mask_trust_cross_none_m). 0 outside route mode, with no
        cross-track, or with the check disabled, so nothing that reads
        this can affect the legacy path.

        Kept as its own number rather than folded into _mask_trust
        because two different things need it, and only one of them is
        about the mask - see _drive_steering."""
        if self.route_off_road_m is not None and self.offroad_none_m > 0:
            # Preferred: metres past the EDGE of the nearest mapped road.
            # Distance to a centreline is a different question - a 5m
            # service road tolerates 2.5m of it, a 2.4m Stromovka footway
            # does not - and measured over the 2026-09-05 runs this one
            # has a real zero: p50 and p75 are both 0.00m (provably on a
            # road), against a cross-track that is never zero because a
            # robot is never exactly on a centreline. p90 0.71, p95 1.27,
            # p99 2.48.
            #
            # It is also measured against the NEAREST way, not the planned
            # one, so drifting onto a legal parallel path reads as 0 -
            # correctly, since that is a re-planning question, not an
            # off-road one.
            off = self.route_off_road_m
            if off <= self.offroad_full_m:
                return 0.0
            span = self.offroad_none_m - self.offroad_full_m
            if span <= 0:
                return 1.0
            return min(1.0, (off - self.offroad_full_m) / span)
        # Fallback for logs/configs without off_road_m: cross-track to the
        # plan. Coarser (it cannot tell a wide road from a lawn) but it is
        # what this used to key off, so nothing regresses without it.
        if self.mask_trust_cross_full_m <= 0 or self.route_cross_track_m is None:
            return 0.0
        cross = abs(self.route_cross_track_m)
        if cross <= self.mask_trust_cross_full_m:
            return 0.0
        span = self.mask_trust_cross_none_m - self.mask_trust_cross_full_m
        if span <= 0:
            return 1.0
        return min(1.0, (cross - self.mask_trust_cross_full_m) / span)

    def _route_turn_hint(self):
        """A small, EARLY additive push toward the way the route turns
        next. Radians, positive left; 0 when there is no turn ahead, no
        geometry for it, or the feature is switched off.

        This is what is left of the route once a constant GPS offset is
        taken seriously. Position-derived guidance - cross-track, the
        off-road distance, the bearing to an aim point - all inherit a
        bias measured at up to 3.17m holding its direction to within a
        degree across a whole run, so all of them can and did steer the
        robot off a road it was tracking correctly. The turn's DIRECTION
        does not: it comes from the polyline, and the only thing the fix
        has to get right is which junction is next and roughly how far,
        which a 3m error inside a 12m approach does not change.

        Shaped like the depth edge nudge rather than like a waypoint: it
        ramps in over route_turn_hint_lead_m so the camera is already
        coming round before the junction arrives, instead of arriving as
        a late demand for a large turn. It is capped well below the road
        mask's own authority, so it biases which way Matty drifts while
        the mask keeps deciding where the road actually is - the mask can
        always overrule it, which is the point.

        Scaled by how sharp the turn is, so a 30 deg bend gets a nudge and
        a 90 deg fork gets the full hint."""
        if self.route_turn_hint_max <= 0 or self.route_turn_dir is None:
            return 0.0
        if self.route_turn_dist_m is None or self.route_turn_dist_m > self.route_turn_hint_lead_m:
            return 0.0
        # 0 at the far edge of the lead distance, 1 at the junction itself
        ramp = 1.0 - self.route_turn_dist_m / max(1e-6, self.route_turn_hint_lead_m)
        # CLOSED on heading: push toward the bearing the route leaves the
        # turn on, in proportion to how far the robot still is from it, so
        # the hint stops by itself once the robot faces the exit.
        #
        # The open-loop version (turn angle x ramp, below) only ever knew
        # "the route turns left 4m ahead" - never "I have already turned".
        # Between the dormitory buildings the fix stopped advancing along
        # the route, so turn_dist sat at 3.4-5.1m, the hint stayed at
        # +6..7deg, junction_hint_gain tripled it, and Matty drove a
        # 301 deg circle in 21s (2026-09-11 run 174423 t=173-194). A
        # target BEARING cannot produce that: once it is reached the
        # command is zero, whatever the fix is doing. The exit bearing is
        # read off the polyline, so the GPS bias cannot touch it either.
        heading, _src = self._current_heading()
        if self.route_turn_closed_loop and self.route_exit_bearing is not None and heading is not None:
            err = normalize_angle(self.route_exit_bearing - heading)   # +ve: exit lies clockwise (right)
            remaining = min(1.0, abs(err) / max(1e-6, self.route_turn_hint_full_angle))
            return -math.copysign(self.route_turn_hint_max * ramp * remaining, err)
        sharp = min(1.0, abs(self.route_turn_dir) / max(1e-6, self.route_turn_hint_full_angle))
        return math.copysign(self.route_turn_hint_max * ramp * sharp, self.route_turn_dir)

    def _arrow_turn_hint(self):
        """Steering nudge toward a planned turn, satnav-arrow style - see
        route_guidance_style in __init__. Returns radians, + = left.

        Armed once per junction. While armed it steers toward the road on the
        turn side, in proportion to how much of the turn is still to do, and
        only as far as the network shows road on that side. It disarms on
        heading or on odometry, never on GPS progress.

        "How much is still to do" depends on route_turn_frame:
          'exit_bearing' - current heading (compass) against the router's
                           exit bearing, armed on the router's distance.
          'odometry'     - odometry heading against a target set once, at
                           arming, and armed on an odometry countdown - see
                           _arm_odometry_turn. No compass anywhere."""
        if self.route_turn_hint_max <= 0 or not self.route_mode:
            self._arrow = None
            return 0.0
        odometry = self.route_turn_frame == 'odometry'
        if odometry:
            heading = self._odom_heading
        else:
            heading, _src = self._current_heading()
        td, dist = self.route_turn_dir, self.route_turn_dist_m
        if dist is not None and dist > self.route_turn_hint_lead_m and self._arrow is None:
            self._arrow_done_key = None           # clear of any junction - nothing to remember
        if odometry:
            if self._arrow is None and heading is not None and self._last_xy is not None:
                self._arm_odometry_turn()
        elif (self._arrow is None and td is not None and dist is not None
                and dist <= self.route_turn_hint_lead_m and heading is not None
                and self._last_xy is not None):
            key = (self.route_plan_seq, int(round(math.degrees(td))))
            if key != self._arrow_done_key:
                if abs(td) > self.route_turn_max_turn:
                    self._arrow_done_key = key
                    print(self.time, 'route turn of %+.0f deg %.1fm ahead is sharper than %.0f deg - '
                                      'not steering it, following the road' % (
                                          math.degrees(td), dist, math.degrees(self.route_turn_max_turn)))
                else:
                    exit_bearing = self.route_exit_bearing
                    if exit_bearing is None and self.route_road_bearing is not None:
                        exit_bearing = normalize_angle(self.route_road_bearing - td)
                    if exit_bearing is not None:
                        self._arrow = dict(key=key, exit=exit_bearing, start=self._last_xy,
                                           turn=abs(td), settled=0, arm_dist=dist, est=dist)
                        print(self.time, 'route turn %+.0f deg in %.1fm armed - exit bearing %.0f deg, '
                                          'will steer when a branch is visible' % (
                                              math.degrees(td), dist, math.degrees(exit_bearing) % 360))
        if self._arrow is None or heading is None:
            return 0.0
        travelled = math.hypot(self._last_xy[0] - self._arrow['start'][0],
                               self._last_xy[1] - self._arrow['start'][1])
        if self._arrow.get('odometry'):
            # rotation still to do, + = left, from the unwrapped odometry
            # heading - see _odo_unwrapped; err is its clockwise counterpart
            remaining = self._arrow['todo'] - (self._odo_unwrapped - self._arrow['unwrap0'])
            self._arrow['remaining'] = remaining
            err = -remaining
            self._arrow['est'] = self._arrow['anchor'] - self._odo_fwd
        else:
            err = normalize_angle(self._arrow['exit'] - heading)   # +: exit lies clockwise (right)
            self._arrow['est'] = self._arrow['arm_dist'] - travelled
        # Done means facing the exit - relative to the size of the turn (a
        # flat 20 deg released a 29 deg fork before it was steered), held for
        # several cycles (the compass jumped 20 deg within a second near
        # 170448 t=96-103), and only after actually driving some of it.
        need = min(self.route_turn_done, max(self.route_turn_done_frac * self._arrow['turn'],
                                             self.route_turn_done_min))
        if abs(err) < need and travelled >= 1.0:
            self._arrow['settled'] += 1
        else:
            self._arrow['settled'] = 0
        wrong_way = (self._arrow.get('odometry')
                     and self._arrow['remaining'] * math.copysign(1.0, self._arrow['todo'])
                     > abs(self._arrow['todo']) + self.route_turn_abandon_extra)
        if self._arrow['settled'] >= 5 or travelled > self.route_turn_timeout_m or wrong_way:
            done = self._arrow['settled'] >= 5
            print(self.time, 'route turn %s (%.0f deg off exit, %.1fm driven since armed)' % (
                'done' if done else
                ('abandoned - heading swung away from it' if wrong_way else 'abandoned'),
                math.degrees(abs(err)), travelled))
            todo = abs(self._arrow.get('todo') or 0.0)
            if (not done and self.replan_on_missed_turn and todo > 0
                    and abs(self._arrow.get('remaining', 0.0)) > self.route_turn_missed_frac * todo):
                # the turn was armed, counted down and never driven - the
                # router can plan another way round from here (no U-turn:
                # its uturn_penalty_m still applies)
                print(self.time, 'that turn was missed (%.0f deg of %.0f never driven) - asking for a new route'
                      % (math.degrees(abs(self._arrow['remaining'])), math.degrees(todo)))
                self.publish('missed_turn', {'remaining_deg': round(math.degrees(self._arrow['remaining']), 1),
                                             'turn_deg': round(math.degrees(self._arrow['todo']), 1)})
            self._arrow_done_key = self._arrow['key']
            self._turn_done_keys.add(self._arrow['key'])
            self._arrow = None
            return 0.0
        side = -1 if err > 0 else 1                             # steer sign that reduces err
        if self._arrow.get('odometry') and side * self._arrow['todo'] < 0:
            # past the planned rotation: straightening out is the road
            # follower's job, not a pull the other way
            self._arrow['pulling'] = False
            return 0.0
        branch = self.branch_road_right if side < 0 else self.branch_road_left
        target = self.branch_dir_right if side < 0 else self.branch_dir_left
        blind = self._blind_turn_hint(side, err)
        if target is None:
            if blind:
                self._announce_blind(blind, err)
            self._arrow['pulling'] = abs(blind) >= math.radians(3.0)
            return blind
        span = max(1e-6, self.route_turn_branch_full_frac - self.route_turn_branch_min_frac)
        visible = max(0.0, min(1.0, (branch - self.route_turn_branch_min_frac) / span))
        remaining = min(1.0, abs(err) / max(1e-6, self.route_turn_hint_full_angle))
        # pull the road follower toward the road on the turn side - never past
        # it, and never the other way
        ramp = 1.0
        if 0 < self.route_turn_near_m < self.route_turn_hint_lead_m:
            # distance still to go; once past the estimate it stays at full
            # until done/timeout, so a turn is never dropped because GPS
            # decided it was already passed
            est = max(0.0, self._arrow['est'])
            ramp = max(0.0, min(1.0, (self.route_turn_hint_lead_m - est)
                                / (self.route_turn_hint_lead_m - self.route_turn_near_m)))
        hint = visible * remaining * ramp * (target - self.last_dir)
        cap = self.route_turn_hint_max * self.junction_hint_gain
        if (self.route_turn_commit_gain > 0 and visible > 0 and self.route_turn_near_m > 0
                and self._arrow['turn'] >= self.route_turn_commit_min
                and -self.route_turn_commit_window_m <= self._arrow['est'] <= self.route_turn_near_m):
            # at the junction: steer from the turn still to do - see
            # route_turn_commit_gain in __init__
            commit = side * min(self.route_turn_commit_max, self.route_turn_commit_gain * abs(err))
            if abs(commit) > abs(target):
                hint = visible * (commit - self.last_dir)
                cap = max(cap, self.route_turn_commit_max)
        if hint * side < 0:
            hint = 0.0
        if abs(blind) > abs(hint):
            self._announce_blind(blind, err)
            hint, cap = blind, max(cap, self.route_turn_blind)
        hint = max(-cap, min(cap, hint))
        self._arrow['pulling'] = abs(hint) >= math.radians(3.0)
        return hint

    def _blind_turn_hint(self, side, err):
        """Steering toward an armed turn the camera cannot see - see
        route_turn_blind_deg in __init__. Radians, + = left, 0 unless all of:
        the odometry says we are at the junction, the camera still sees road
        where the robot is (so it has not already left it), and the depth on
        that side is clear."""
        arrow = self._arrow
        if self.route_turn_blind <= 0 or arrow is None or self.route_turn_near_m <= 0:
            return 0.0
        if arrow['turn'] < self.route_turn_blind_min_turn:
            return 0.0                      # mild turn - the camera can see that branch
        if not (-self.route_turn_blind_window_m <= arrow['est'] <= self.route_turn_near_m):
            return 0.0
        if self.road_ahead_frac < self.route_turn_blind_min_ahead:
            return 0.0                      # the camera no longer backs this up
        side_dist = self.right_dist if side < 0 else self.left_dist
        if side_dist is not None and side_dist <= self.turning_dist:
            return 0.0                      # something is there - obstacle avoidance owns this
        return side * min(self.route_turn_blind, 0.5 * abs(err))

    def _announce_blind(self, blind, err):
        """Say it once per turn, and only when the map is actually steering -
        not when the visible branch was already pulling harder."""
        if self._arrow is not None and not self._arrow.get('blind'):
            self._arrow['blind'] = True
            print(self.time, 'no branch visible at the mapped turn %+.0f deg away - steering %+.0f deg by the map '
                              'while the camera still sees road (%.2f ahead)'
                   % (math.degrees(err), math.degrees(blind), self.road_ahead_frac))

    def _track_turn(self):
        """The turn the router reports next, followed by odometry from the
        first report within route_turn_track_m - see route_turn_track_m.
        Kept while the router moves on to a later turn, until this one is
        armed, done, passed by route_turn_track_pass_m, or the plan changes."""
        if self._turn_done_seq != self.route_plan_seq:
            self._turn_done_seq, self._turn_done_keys = self.route_plan_seq, set()
        track = self._turn_track
        if track is not None and (track['seq'] != self.route_plan_seq
                                  or track['anchor'] - self._odo_fwd < -self.route_turn_track_pass_m):
            track = self._turn_track = None
        td, dist = self.route_turn_dir, self.route_turn_dist_m
        if td is None or dist is None or dist > self.route_turn_track_m:
            return track
        key = (self.route_plan_seq, int(round(math.degrees(td))))
        if key in self._turn_done_keys:
            return track
        if track is None:
            track = self._turn_track = dict(key=key, seq=self.route_plan_seq, td=td, rel=None, exit=None,
                                            samples=[], first_fwd=self._odo_fwd, anchor=None)
        if track['key'] == key:
            track['samples'].append(dist + self._odo_fwd)
            del track['samples'][:-40]
            ordered = sorted(track['samples'])
            track['anchor'] = ordered[len(ordered) // 2]
            if self.route_turn_rel is not None:
                track['rel'] = self.route_turn_rel
            if self.route_exit_bearing is not None:
                track['exit'] = self.route_exit_bearing
            if self.route_road_bearing is not None:
                track['road'] = self.route_road_bearing
        return track

    def _arm_odometry_turn(self):
        track = self._track_turn()
        if track is None or track['anchor'] is None:
            return
        est = track['anchor'] - self._odo_fwd
        if est > self.route_turn_hint_lead_m or est < -self.route_turn_near_m:
            return          # not there yet, or already past - too late to start a turn
        rel = track['rel'] if track['rel'] is not None else track['td']
        if abs(rel) < self.route_turn_min_rel:
            self._turn_done_keys.add(track['key'])
            self._turn_track = None
            return          # a jog or a kink, not a turn - see route_turn_min_rel_deg
        sharpest = rel
        if self.route_turn_arm_max_mismatch > 0 and abs(track['td']) > abs(rel):
            sharpest = track['td']          # the route turns back here - see route_turn_arm_max_mismatch_deg
        if abs(sharpest) > self.route_turn_max_turn:
            self._turn_done_keys.add(track['key'])
            self._arrow_done_key = track['key']
            self._turn_track = None
            print(self.time, 'route turn of %+.0f deg %.1fm ahead is sharper than %.0f deg - '
                              'not steering it, following the road' % (
                                  math.degrees(sharpest), est, math.degrees(self.route_turn_max_turn)))
            return
        if self.route_turn_align_max > 0 and track.get('road') is not None:
            heading, source = self._current_heading()
            if (heading is not None and source in self.route_turn_align_sources
                    and abs(normalize_angle(heading - track['road'])) > self.route_turn_align_max):
                if not track.get('misaligned_note'):
                    track['misaligned_note'] = True
                    print(self.time, 'route turn %.1fm ahead not armed - heading %.0f deg, the route runs %.0f deg here'
                          % (est, math.degrees(heading) % 360, math.degrees(track['road']) % 360))
                return
        driven = self._odo_fwd - track['first_fwd']
        todo = how = None                  # signed rotation still to do from here, + = left
        if driven >= self.route_turn_approach_m:
            todo = rel - normalize_angle(self._odom_heading - self._odom_heading_mean())
            how = 'approach heading over %.1fm driven' % driven
        else:
            heading, source = self._current_heading()
            if heading is not None and source == 'odo+gps' and track['exit'] is not None:
                todo = -normalize_angle(track['exit'] - heading)
                how = 'exit bearing %.0f deg against odo+gps heading %.0f deg' % (
                    math.degrees(track['exit']) % 360, math.degrees(heading) % 360)
                if self.route_turn_arm_max_mismatch > 0 and (
                        abs(todo) > self.route_turn_max_turn
                        or abs(normalize_angle(todo - rel)) > self.route_turn_arm_max_mismatch):
                    self._turn_done_keys.add(track['key'])
                    self._arrow_done_key = track['key']
                    self._turn_track = None
                    print(self.time, 'route turn %+.0f deg %.1fm ahead not armed - the %s asks for %+.0f deg, '
                                      'a U-turn or the wrong way - following the road' % (
                                          math.degrees(rel), est, how, math.degrees(todo)))
                    return
        if todo is None:
            return          # nothing driven and no trustworthy heading - wait
        target = normalize_angle(self._odom_heading + todo)
        self._arrow = dict(key=track['key'], exit=target, start=self._last_xy, turn=abs(rel), settled=0,
                           arm_dist=est, anchor=track['anchor'], est=est, odometry=True, pulling=False,
                           todo=todo, unwrap0=self._odo_unwrapped, remaining=todo)
        self._turn_track = None
        print(self.time, 'route turn %+.0f deg in %.1fm armed (%s), will steer when a branch is visible' % (
            math.degrees(rel), est, how))

    def _odom_heading_mean(self):
        """Circular mean of the odometry heading over the last ~3 m driven."""
        if not self._odo_hist:
            return self._odom_heading
        return math.atan2(sum(math.sin(h) for _, h in self._odo_hist),
                          sum(math.cos(h) for _, h in self._odo_hist))

    def _route_speed_limit_applies(self):
        """Whether the router's speed_limit caps this cycle - see
        route_speed_gate in __init__."""
        if self.route_speed_gate != 'camera':
            return True
        if self._camera_offroad_frac() > 0:
            return True
        arrow = self._arrow
        return arrow is not None and (arrow.get('est', float('inf')) <= self.route_turn_slow_m
                                      or arrow.get('pulling', False))

    def _scout_command(self, xy):
        """(speed, steering) while scouting for the road after the retreats
        are spent - see road_scout_m in __init__. None when the road is back."""
        travelled = math.hypot(xy[0] - self._road_scout[0], xy[1] - self._road_scout[1])
        if self.road_found_streak >= self.road_found_frames:
            print(self.time, 'road found while scouting after %.1fm - resuming' % travelled)
            self._road_scout = None
            return None
        ahead = self.last_obstacle if self.last_obstacle is not None else float('inf')
        corridor = self._corridor_hold_dist()
        if corridor is not None:
            ahead = min(ahead, corridor)
        if travelled >= self.road_scout_m or ahead <= self.road_scout_clear_m:
            print(self.time, 'scouted %.1fm without finding the road (%.1fm clear ahead) - holding'
                  % (travelled, ahead))
            self._road_scout = None
            self._road_hold = True
            return 0.0, 0.0
        return self.road_scout_speed, max(-self.road_scout_max, min(self.road_scout_max, self._road_last_dir))

    def _scout_or_hold(self, xy, why):
        """Start a bounded scout toward the last road the camera saw, or hold
        where the old code held - see road_scout_m in __init__."""
        ahead = self.last_obstacle if self.last_obstacle is not None else float('inf')
        on_road = (self.route_off_road_m is None or self.route_off_road_m <= self.road_scout_on_road_m)
        if (self.road_scout_m > 0 and self._road_scouts < self.road_scout_tries
                and on_road and ahead > self.road_scout_clear_m):
            self._road_scouts += 1
            self._road_scout = xy
            print(self.time, '%s - scouting up to %.1fm toward the last road seen (%+.0f deg), %.1fm clear ahead'
                  % (why, self.road_scout_m, math.degrees(self._road_last_dir), ahead))
            return self._scout_command(xy)
        print(self.time, '%s - holding until the road is visible' % why)
        self._road_hold = True
        return 0.0, 0.0

    def _road_lost_command(self, xy, dt):
        """(speed, steering) while retreating from, or holding after, a lost
        road - see road_lost_retreat_sec in __init__. None when not active."""
        if self.road_lost_retreat_sec <= 0:
            return None
        if self._road_scout is not None:
            command = self._scout_command(xy)
            if command is not None:
                return command
        if self._road_hold:
            if self.road_found_streak >= self.road_found_frames:
                print(self.time, 'road visible again - releasing the road-lost hold')
                self._road_hold = False
                self._road_hold_since = None
                self._road_scouts = 0
                return None
            if self._road_hold_since is None:
                self._road_hold_since = self.time
            held = (self.time - self._road_hold_since).total_seconds()
            if (self.road_hold_retry_sec <= 0 or held < self.road_hold_retry_sec
                    or self._road_hold_retries >= self.road_hold_max_retries):
                return 0.0, 0.0
            # see road_hold_retry_sec in __init__: a new view for the network.
            # Falls through to the retreat below, which starts at once
            # while the road is still lost.
            self._road_hold_retries += 1
            self._road_hold = False
            self._road_hold_since = None
            self._road_retreats = 0
            self._road_scouts = 0
            print(self.time, 'no road for %.0fs while holding - retry %d/%d: retreat and scout again'
                  % (held, self._road_hold_retries, self.road_hold_max_retries))
        if self._road_retreat is not None:
            travelled = math.hypot(xy[0] - self._road_retreat[0], xy[1] - self._road_retreat[1])
            if self.road_found_streak >= self.road_found_frames:
                print(self.time, 'road re-acquired after retreating %.1fm - resuming' % travelled)
                self._road_retreat = None
                return None
            if travelled >= self.road_lost_retreat_max_m or not self.retrace_queue:
                self._road_retreat = None
                return self._scout_or_hold(xy, 'road-lost retreat spent (%.1fm)' % travelled)
            return -self.backup_speed, self._next_retrace_steering(dt)
        if (self.road_lost and self.state == State.DRIVE and self.road_lost_since is not None
                and (self.time - self.road_lost_since).total_seconds() >= self.road_lost_retreat_sec):
            if self._road_retreats >= self.road_lost_max_retreats or not self.path_history:
                return self._scout_or_hold(xy, 'road lost and no retreat left')
            if self._arrow is not None and self._arrow.get('odometry') and self._arrow['est'] > 0:
                # The road ends before the map's junction: the GPS along-track
                # error is larger than the distance left (165145: ~10 m under
                # trees, Matty at a dead end the router put 10 m before the
                # fork). The end of the road is where the turn is.
                print(self.time, 'road ends %.1fm before the mapped junction - taking the armed turn from here'
                      % self._arrow['est'])
                self._arrow['anchor'] = self._odo_fwd
                self._arrow['est'] = 0.0
            self._road_retreats += 1
            self._road_retreat = xy
            self.retrace_queue = deque(reversed(self.path_history))
            self.current_backup_steering = 0.0
            print(self.time, 'road lost for %.1fs - retreating along the path just driven (%d/%d)' % (
                (self.time - self.road_lost_since).total_seconds(), self._road_retreats,
                self.road_lost_max_retreats))
            return -self.backup_speed, self._next_retrace_steering(dt)
        return None

    def _route_corridor_bias(self, dt=None):
        """Slow additive steering bias pulling back onto the planned
        corridor - the LOW-FREQUENCY half of the control split described
        at route_cross_track_gain in __init__. Positive = steer left.

        Sign convention, checked end to end: the router reports
        cross_track_m > 0 when the robot is to the RIGHT of the route's
        direction of travel, and this platform steers left for positive
        angles (matty.py integrates heading += dist/radius with radius
        positive for a positive joint angle), so a positive gain on a
        positive cross-track correctly steers left, back toward the route.
        The heading term follows the same convention.

        Smoothed with an EMA on top of the router's own per-fix smoothing:
        this term exists to move the equilibrium, not to react, and GPS
        noise must not reach the steering as jitter."""
        if not self.route_mode or self.route_cross_track_m is None:
            return 0.0
        # --- the driving tube: cross-track as a fraction of the road ---
        # The restoring push should say "how far toward the EDGE am I",
        # not "how many metres from a line". Measured over the 2026-09-06
        # runs, Matty sits at the same fraction of the way to the edge on
        # both road types it drove - 0.32 on 1.2m-half footways, 0.29 on
        # 2.5m-half service roads - so it is centring equally well on
        # both. But in bare metres that same quality of centring produced
        # 3.1deg of bias on the footway and 5.7deg on the wide road: the
        # hardest push where there is the most room, and where a push
        # fights the robot's legitimate use of that width.
        #
        # Normalising by the way's own half-width fixes the ratio, then
        # scaling back up by tube_reference_halfwidth_m keeps the gain's
        # units and its field-tuned meaning on a path of that width - a
        # footway is unchanged and everything wider is relaxed in
        # proportion. 0 disables and restores bare metres.
        # The normalisation only ever RELAXES, never amplifies - capped at
        # 1.0. On a wide road GPS can resolve where the robot sits across
        # it, so easing off there is free. On a NARROW one it cannot: a 1m
        # fix error is 1.4x the entire image width of a 0.70m path, so
        # amplifying the cross-track push as the road narrows would be
        # turning up the gain on the noise, on exactly the paths with no
        # room to absorb it. Narrow roads therefore get the same push as
        # the reference width and no more, and the direction half of this
        # term below - which a position error barely moves - does the rest.
        cross = self.route_cross_track_m
        if self.tube_reference_halfwidth_m > 0 and self.route_road_halfwidth_m:
            cross *= min(1.0, self.tube_reference_halfwidth_m / self.route_road_halfwidth_m)
        target = self.route_cross_track_gain * cross
        # The DIRECTION half, and the reason it is not scaled by anything
        # above: which way the mapped path runs here is a property of the
        # map, not of where the robot is on it, so a 1-3m fix error barely
        # moves it as long as the snap picked the right way. That makes it
        # the only part of the route signal that stays meaningful on a
        # path narrower than the fix error - which is the case this whole
        # module is heading toward at Stromovka.
        error = self._route_heading_error()
        if error is not None:
            target += self.route_heading_gain * error
        target = max(-self.route_bias_max, min(self.route_bias_max, target))
        if dt is None:
            return target  # diagnostic call (see _gps_status_line) - no state change
        self._route_bias += self.route_bias_alpha * (target - self._route_bias)
        return self._route_bias

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
        # The mask says less when the camera is pointing well off the
        # mapped road axis, because that is measurably when it starts
        # reporting a lawn as drivable - see _mask_trust. Outside route
        # mode this is exactly 1.0 and the line is a no-op.
        local_dir = self.last_dir * self._mask_trust()
        # scale the depth component's blend weight by how much of
        # depth_profile is actually valid right now - see
        # _depth_confidence. 1.0 (no reduction) in the ordinary case;
        # only pulls this down during a widespread sensor dropout, in
        # which case last_dir (RGB road extraction, unaffected by an IR
        # emitter being overwhelmed by sunlight) picks up the slack
        # instead of steering off of a component with nothing real left
        # to say.
        effective_free_space_weight = self.free_space_weight * self._depth_confidence()
        # ...and by how far off the mapped way the robot currently is.
        #
        # Discounting the MASK alone (see _mask_trust) does not fix the
        # drift, and measuring said so: it removes half of local_dir and
        # hands the freed weight straight to the depth terms, which were
        # measured pushing off-corridor harder than the mask ever did
        # (4.8deg and 1.6deg per cycle away, against the mask's 0.85deg).
        # The result was a NET restoring force of 0.03deg - worse than
        # doing nothing.
        #
        # The depth nudge is documented as a CRUISING preference - "lean
        # toward whichever side is more open" - explicitly not a safety
        # net; the safety net is stop_dist/turning_dist and the avoidance
        # state machine, and none of that is touched here. On a lawn
        # beside a path, "more open" is the lawn, so out here that
        # preference is not neutral, it is the thing keeping the robot
        # off the road. So its NON-URGENT part fades as the robot goes
        # off-corridor, while the part driven by something genuinely
        # close keeps full strength: at urgency 1 (a flank down at
        # stop_dist) this multiplier is exactly 1.0 whatever the
        # cross-track says.
        # The camera's own "I am leaving the road" signal, combined with the
        # route's by taking the STRONGER of the two: they are independent
        # witnesses (one map+GPS, one appearance) and either alone is a
        # sufficient reason to stop letting "which side is more open" steer.
        # Taking the max rather than averaging means the camera one still
        # works with no route loaded, which is the case that actually
        # happened - see _camera_offroad_frac.
        offroad = max(self._route_offroad_frac(), self._camera_offroad_frac())
        if offroad > 0 and self.edge_correction_max_rad > 0:
            urgency = min(1.0, abs(self._edge_bin_correction()) / self.edge_correction_max_rad)
            effective_free_space_weight *= 1.0 - offroad * (1.0 - urgency)
        if effective_free_space_weight > 0:
            # whole-profile average clipped to the normal cruising ceiling
            # first, THEN the (potentially much larger, see
            # _edge_bin_correction) edge term added on top and the TOTAL
            # clipped to avoid_steering - so an urgent edge call can get
            # as assertive as a real avoidance turn, but never past it
            whole_profile = max(-self.turn_angle, min(self.turn_angle, self._free_space_steering()))
            depth_component = whole_profile + self._edge_bin_correction()
            depth_component = max(-self.avoid_steering, min(self.avoid_steering, depth_component))
            depth_component = self._damp_outward(depth_component)
            local_dir = (1 - effective_free_space_weight) * local_dir + effective_free_space_weight * depth_component
        # Added AFTER the blend, not inside it: this is collision avoidance
        # for something in the robot's own path, and free_space_weight
        # exists to share cruising preference with the road mask - it has
        # no business diluting "that pole is in the way" to 60%.
        local_dir = max(-self.avoid_steering, min(self.avoid_steering,
                                                  local_dir + self._corridor_repulsion()))
        # Diagnostics only - nothing reads this back. Every term that can
        # move the wheels, recorded where it is decided, so a replay (or the
        # viewer HUD) can say WHICH signal steered instead of inferring it.
        self.steer_debug = dict(
            branch='local', last_dir=self.last_dir, mask_trust=self._mask_trust(),
            w_fs=effective_free_space_weight, offroad=offroad,
            fs=self._free_space_steering(), edge=self._edge_bin_correction(),
            repel=self._corridor_repulsion(), local_dir=local_dir,
            gps_weight=0.0, gps_steer=0.0, hint=0.0, bias=0.0,
            profile=list(self.depth_profile or []), bin_road=list(self.profile_bin_road))
        if self.route_guidance_style == 'arrow':
            # see route_guidance_style in __init__: the road network drives,
            # the map only suggests a visible branch at a junction
            hint = self._arrow_turn_hint()
            steering = max(-self.avoid_steering, min(self.avoid_steering, local_dir + hint))
            weight = 0.0
            current_heading, _source = self._current_heading()
            if (self.route_bearing_authority_max > 0 and self.route_mode and self.route_authority
                    and self.bearing_to_target is not None and current_heading is not None
                    and self.road_ahead_frac < self.camera_offroad_off_frac):
                # only with no road in front of the camera at all
                error = normalize_angle(current_heading - self.bearing_to_target)
                limit = self.route_steer_limit or self.turn_angle
                weight = min(self.route_authority, self.route_bearing_authority_max)
                steering = (1 - weight) * steering + weight * max(-limit, min(limit, error))
            self.steer_debug.update(branch='arrow', hint=hint, gps_weight=weight, target=steering,
                                    branch_l=self.branch_road_left, branch_r=self.branch_road_right)
            return self._rate_limit_steering(steering, dt)

        current_heading, _source = self._current_heading()
        if not self.follow_gps_target or self.bearing_to_target is None or current_heading is None:
            # No usable heading for the aim point - but in route mode the
            # cross-track bias only needs a POSITION, which is a separate
            # (and much better conditioned) question than a bearing, so it
            # still applies. This is the case where GPS has a fix but no
            # heading source has settled yet.
            bias = self._route_corridor_bias(dt)
            self.steer_debug.update(branch='no_gps', bias=bias, target=local_dir + bias)
            return self._rate_limit_steering(local_dir + bias, dt)

        error = normalize_angle(current_heading - self.bearing_to_target)

        if self.route_mode and self.route_guidance_mode == 'corridor':
            # --- corridor mode: the route is a BIAS, not a competitor ---
            # Blending here would let the mask cancel the only restoring
            # signal there is (see route_cross_track_gain in __init__ for
            # the measurement). Adding cannot be cancelled - it moves the
            # equilibrium the mask settles into, while leaving the mask in
            # full charge of the fast corrections it is genuinely good at.
            # The bias must never out-argue the depth. _edge_bin_correction
            # is the "something is close on that flank" signal, and it is
            # geometric - it does not hallucinate and it does not depend on
            # GPS. Measured over the 2026-09-05 runs, the corridor bias
            # pulled against the more-open side in 5.9% of cycles with both
            # signals strong (bins differing >0.5m while cross-track >1.5m),
            # which at +-15deg of bias is easily enough to cancel the
            # avoidance nudge - the reported "GPS overwrites the bins".
            #
            # So the bias yields in proportion to how hard the edge term is
            # already pushing. At a full-strength edge correction the route
            # says nothing at all; on an open path it is unaffected. This
            # keeps the documented priority order (local geometry beats the
            # map) in the one place the additive form could break it.
            bias = self._route_corridor_bias(dt)
            edge = self._edge_bin_correction()
            urgency = 0.0
            if self.edge_correction_max_rad > 0:
                urgency = min(1.0, abs(edge) / self.edge_correction_max_rad)
            # The same reasoning as the depth fade above, applied to the
            # yield itself: off the corridor, a non-urgent edge call is
            # not a reason to stop steering back onto the road.
            urgency = max(0.0, min(1.0, urgency - offroad))
            if edge * bias < 0:
                # ...and ONLY when the two actually disagree. The yield
                # exists so local geometry can override the map, which is
                # a question that does not arise when both push the same
                # way - and cutting the restoring force there was pure
                # loss. Measured over the 2026-09-05 runs, the edge term
                # and the bias were both above 3deg on 3142 cycles and
                # disagreed on 77% of them, near-cancelling on 40%; while
                # more than 1.5m off the corridor the edge term pointed
                # further off it three times as often as back toward it,
                # because once the robot is on the verge the thing on its
                # flank IS the road edge it should be crossing back over.
                steering = local_dir + (1.0 - urgency) * bias
            else:
                steering = local_dir + bias
            limit = self.route_steer_limit or self.turn_angle
            # The router publishes a cruise authority for corridor mode
            # and this branch used to throw it away, steering on the
            # additive bias alone. That was survivable only because a
            # separate bug was compensating for it: half the route was
            # being classified as "junction" (see Route.junction_turn),
            # and junction mode DOES blend the aim point, at 0.80. Fixing
            # that classification removed the compensation, and measured
            # paired against the old behaviour the on-road restoring
            # command fell from +0.82 to +0.24 deg at 0.3-0.8m off-centre
            # and from +0.21 to -0.39 deg at 0.8-1.5m. At Robotour that
            # trade is the wrong way round: a faster robot that holds the
            # road less well is a worse robot, because leaving the road
            # ends the run.
            #
            # So the authority is now used where it was always meant to
            # be, deliberately and everywhere, instead of arriving by
            # accident at forks. corridor_aim_frac keeps it well under the
            # junction figure - the mask still centres better than a 1-3m
            # fix on a straight path, which is the whole reason
            # cruise_authority is lower than junction_authority - while
            # the additive bias continues to do what blending alone
            # cannot: move the equilibrium the mask settles into.
            if self.route_authority is not None and self.corridor_aim_frac > 0:
                w = min(1.0, self.route_authority * self.corridor_aim_frac)
                steering = (1 - w) * steering + w * max(-limit, min(limit, error))
            steering = max(-limit, min(limit, steering))
            # --- off the mapped way: the map takes over, not the mask ---
            # A bias capped at route_bias_max is the right size for
            # holding a lane. It is not the right size for getting back
            # onto one, and the 2026-09-05 runs show why the difference
            # matters: while more than 1.5m off the corridor the finished
            # command steered back at a net 2.2deg per cycle, which at
            # 0.4m/s takes about 160s to recover 2.5m - the reported
            # "drove parallel to the road on the grass for a very long
            # time". Some of that restoring force was not even the bias:
            # it came from the router classifying half the route as
            # "junction" and handing the aim point 0.85 authority, which
            # is a side effect of a different bug (see
            # Route.junction_turn) and disappears once that is fixed.
            #
            # So say it directly instead. Past mask_trust_cross_full_m the
            # aim point - a real point on a mapped way, a few metres ahead
            # - is blended in with weight proportional to how far off the
            # robot is, reaching route_offroad_authority at
            # mask_trust_cross_none_m. This is the competition
            # requirement in one line: a path that looks drivable but is
            # not on the map must not be able to out-vote the map.
            #
            # It cannot run away with the robot. It is still clipped to
            # route_steer_limit (the cruising ceiling), still rate
            # limited, still entirely subordinate to the avoidance state
            # machine, which runs above _drive_steering and does not call
            # it at all while a maneuver is active. 0 disables.
            if offroad > 0 and self.route_offroad_authority > 0:
                w = offroad * self.route_offroad_authority
                steering = (1 - w) * steering + w * max(-limit, min(limit, error))
            self.steer_debug.update(branch='corridor', bias=bias, target=steering)
            return self._rate_limit_steering(steering, dt)

        # --- junction / recovery: the route takes over ---
        # A fork needs a decisive turn, not a nudge: at the cruising 20deg
        # ceiling Matty's turning radius is 0.91m and a 90deg turn swings
        # wide across the corner; the router raises the ceiling toward
        # 40deg (radius 0.44m) approaching a planned junction, and relaxes
        # the rate limit so it can actually be reached inside the junction.
        steer_limit = self.route_steer_limit if self.route_mode and self.route_steer_limit \
            else self.turn_angle
        # The aim-point bearing does NOT survive the GPS bias either: at
        # the cruising 8m lookahead a 3m lateral offset is 21deg of false
        # bearing, which is most of a junction turn pointed at nothing.
        # Where the route's own turn geometry is available it is used
        # instead, because it is a property of the polyline; the bearing
        # remains only as the fallback for a route that cannot supply one.
        if self.route_turn_hint_max > 0 and self.route_turn_dir is not None:
            # ADDITIVE, like corridor mode - not blended by route_authority.
            # Blending made sense for an aim-point bearing that carried
            # position; the hint carries only direction, and at 0.85
            # authority a hint that has correctly gone to zero would still
            # silence 85% of the road mask and depth steering. Close
            # geometry still takes it back, and a mask that sees no road at
            # all that way still vetoes most of it.
            hint = self._route_turn_hint() * self.junction_hint_gain
            if self.edge_correction_max_rad > 0 and self.junction_edge_yield > 0:
                urgency = min(1.0, abs(self._edge_bin_correction()) / self.edge_correction_max_rad)
                hint *= 1.0 - self.junction_edge_yield * urgency
            road_frac_that_way = self.left_road_frac if hint > 0 else self.right_road_frac
            if road_frac_that_way < self.route_min_road_frac:
                hint *= self.route_veto_authority
            self.steer_debug.update(branch='hint', hint=hint,
                                    target=max(-steer_limit, min(steer_limit, local_dir + hint)))
            return self._rate_limit_steering(max(-steer_limit, min(steer_limit, local_dir + hint)), dt)
        bearing_steering = max(-steer_limit, min(steer_limit, error))

        if self.route_mode and self.route_authority is not None:
            # The router decides how much the bearing is worth right now -
            # low on a straight path (the mask centres better than a 1-3m
            # GPS fix), high at a junction (the mask has no opinion about
            # which branch leads to the target) and while recovering from
            # a confirmed excursion. See osm_router.py's "division of
            # labour" section.
            #
            # This REPLACES the road-fraction gate rather than combining
            # with it, and that is the deliberate part. That gate exists
            # to stop the robot chasing a bearing off the road - a real
            # risk when the bearing points at a destination 200m away
            # across a lawn. In route mode the bearing points a few meters
            # ahead along a mapped path, so the thing the gate protects
            # against is already gone - while its side effect, refusing to
            # turn toward a branch that is momentarily at the edge of the
            # mask, would break exactly the junction case this is for.
            weight = self.route_authority
            # ...with one veto kept. If the mask sees essentially NO
            # drivable surface the way the route wants to turn, do not
            # commit a hard junction turn into it: that is what a bad fix,
            # a mis-mapped branch or the wrong junction entirely looks
            # like. A threshold this low (route_min_road_frac, 0.02
            # against a typical 0.20-0.31 healthy fraction) only fires on
            # "there is nothing there at all", not on "the branch is at
            # the edge of the frame", which is why it can coexist with
            # dropping the proportional gate above.
            # positive steering is LEFT, same convention as the legacy
            # branch below - keep the two reading identically
            road_frac_that_way = self.left_road_frac if bearing_steering > 0 else self.right_road_frac
            if road_frac_that_way < self.route_min_road_frac:
                weight = min(weight, self.route_veto_authority)
            # ...and the same yield to close geometry that corridor mode
            # has had all along. This branch had only the mask-based veto
            # above, which is worth nothing exactly when it is needed: at
            # a junction the robot is usually well off the road axis, so
            # mask_trust is at its 0.25 floor and the mask is the signal
            # already known to be unreliable there.
            #
            # Field case, 2026-09-06 run 115237 t=425-427: the route
            # commanded a 35deg left turn at authority 0.85 with
            # mask_trust 0.25; the left zone collapsed 4.16 -> 2.94 ->
            # 0.76m and _edge_bin_correction went to its full -35deg, but
            # at 0.85 authority the depth kept only 15% of the vote -
            # about -2.6deg against +34deg of route - and the front
            # bumper hit 0.8s later. That is the reported "one of these
            # turns resulted in a crash".
            #
            # The priority order this file documents is that local
            # geometry beats the map. It was being enforced in one branch
            # and not the other.
            if self.edge_correction_max_rad > 0 and self.junction_edge_yield > 0:
                urgency = min(1.0, abs(self._edge_bin_correction()) / self.edge_correction_max_rad)
                weight *= 1.0 - self.junction_edge_yield * urgency
        elif self.bearing_blend_road_frac > 0:
            road_frac_that_way = self.left_road_frac if bearing_steering > 0 else self.right_road_frac
            weight = min(1.0, road_frac_that_way / self.bearing_blend_road_frac)
        else:
            weight = 1.0
        weight *= self._bearing_distance_scale()  # fade out approaching the target - see item 16
        steering = (1 - weight) * local_dir + weight * bearing_steering
        # The corridor bias is not a corridor-MODE feature. It used to be
        # computed only in the branch above, so for the 52% of forward
        # driving the router classified as "junction" the one signal that
        # knows the robot has drifted off the mapped way contributed
        # nothing at all - and _route_bias sat frozen at whatever value it
        # held when the mode last changed. Added here at (1 - weight) so
        # it cannot argue with the junction aim point, which is the
        # stronger and better-conditioned signal exactly when authority
        # is high.
        bias = self._route_corridor_bias(dt)
        self.steer_debug.update(branch='bearing', gps_weight=weight, gps_steer=bearing_steering,
                                bias=bias, target=steering + (1 - weight) * bias)
        return self._rate_limit_steering(steering + (1 - weight) * bias, dt)

    def _following_status(self):
        """Human-readable reason why GPS bearing-following is or isn't
        currently steering the robot - logging only, mirrors the guard
        clauses in _drive_steering()."""
        if self.route_mode:
            # in route mode the target is the router's rolling aim point,
            # so "no target" here means the router has nothing to follow -
            # which is normal before the start QR - and its own state is
            # the informative thing to report, not this file's
            parts = ['osm-route:%s' % self.route_state]
            if self.route_guidance_mode:
                parts.append(self.route_guidance_mode)
            if self.route_authority is not None:
                parts.append('authority=%.2f' % self.route_authority)
            herr = self._route_heading_error()
            if herr is not None:
                parts.append('road_axis_err=%+.0fdeg mask_trust=%.2f'
                              % (math.degrees(herr), self._mask_trust()))
            if self.route_guidance_mode == 'corridor':
                parts.append('bias=%+.1fdeg' % math.degrees(self._route_corridor_bias()))
            if self.route_remaining_m is not None:
                parts.append('remaining=%.0fm' % self.route_remaining_m)
            if self.route_cross_track_m is not None:
                parts.append('cross=%+.1fm' % self.route_cross_track_m)
            if self.route_hold:
                parts.append('HOLD=%s' % self.route_hold)
            return ' '.join(parts)
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
        if self.compass_cal_skipped:
            # legs rejected as too curved to learn from (item 23) - if this
            # climbs while compass_offset never settles, the robot is simply
            # never driving straight long enough to calibrate
            parts.append('cal_skipped=%d' % self.compass_cal_skipped)
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
            self.scan_confident_streak = 0
        # The sweep ORIGIN, unlike the samples, is per-ATTEMPT and is reset
        # on every entry - the single most expensive bug in the 2026-09-05
        # runs. scan_start_heading used to live in the block above, so
        # after the first abort within an episode `swept` was still being
        # measured from the heading the episode STARTED at. Any re-entry
        # therefore satisfied `swept >= current_scan_min_sweep` on its
        # first cycle and left TURNING again after min_turn_time (0.1s in
        # the shipped config), having rotated the robot by about 2 deg.
        # Measured over those runs: TURNING lasted a median of 0.16s, 66%
        # of its 276 episodes were under 0.35s, and 108 of 278 exits were
        # this exact case - the sweep requirement already satisfied at the
        # moment of entry. The robot spent 7.9% of the run reversing and
        # 4.3% "turning" in slices too short to change where it pointed,
        # which is the reported "avoidance loops repeating the same
        # actions and taking too long". Keeping scan_samples cumulative
        # (the fix this replaced, and still correct - a good heading seen
        # early must survive an abort) while making the sweep origin
        # per-attempt gives both: the evidence accumulates, the commitment
        # is measured from where this attempt actually began.
        self.scan_start_heading = self.last_heading

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

    def _escape_backup_steering(self):
        """Steering to hold while reversing during an escape - the other
        way round from the forward turn, which is what actually makes a
        K-turn a K-turn.

        This is not a preference, it is the chassis model. matty.py
        integrates heading += dist/radius with radius set by the joint
        angle and `dist` SIGNED by the direction of travel, so reversing
        with the steering held where it was during the forward leg
        retraces the same arc and gives back exactly the heading just
        gained. That is precisely what you want for the ordinary first
        backup (retrace - back out along your own tracks) and precisely
        what you do not want when the point of the maneuver is to end up
        pointing somewhere else.

        Countersteering instead makes both legs turn the same way: at
        backup_speed 0.2 m/s for the 1.0s minimum with 45deg of
        articulation the reverse leg is worth about 30deg, and the
        forward leg at avoid_speed another ~37deg - roughly 67deg per
        two-second cycle, against the ~2-6deg the same two seconds used
        to produce. It also needs no more room than the old behaviour:
        a three-point turn is the standard way to turn a long vehicle in
        a space too tight to turn in one sweep, which is exactly the
        situation escape mode exists for.

        Deliberately escape-mode only. Outside it the first backup is a
        retrace, and retracing is the highest-confidence direction
        available (ground the robot has just proven it can cross) - this
        must not touch that."""
        return -self.turn_sign * self.current_avoid_steering

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
                # ...and forget WHICH side was last picked, or the counter
                # just reset is immediately bumped again by comparing against
                # a pick made metres ago. With max_same_direction_repeats=1
                # that meant any two consecutive episodes choosing the same
                # side - however far apart - forced escape mode and FLIPPED
                # the side: at the 2026-09-11 ramp (172231 t=41.5) a correct
                # left turn, 10 m after an earlier left, became a right turn
                # into the gabion wall.
                self.last_turn_sign_choice = None
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
        self._odom_heading = math.radians(heading_cdeg / 100.0)
        if self._odo_unwrap_prev is not None:
            self._odo_unwrapped += normalize_angle(self._odom_heading - self._odo_unwrap_prev)
        self._odo_unwrap_prev = self._odom_heading
        if self._odo_hist_xy is not None:
            odx, ody = xy[0] - self._odo_hist_xy[0], xy[1] - self._odo_hist_xy[1]
            self._odo_travel += math.hypot(odx, ody)
            self._odo_fwd += odx * math.cos(self._odom_heading) + ody * math.sin(self._odom_heading)
        self._odo_hist_xy = xy
        self._odo_hist.append((self._odo_travel, self._odom_heading))
        while len(self._odo_hist) > 2 and self._odo_travel - self._odo_hist[0][0] > 3.0:
            self._odo_hist.popleft()
        if self.heading_est is not None and self.time is not None:
            self.heading_est.update_pose(self.time.total_seconds(), xy[0], xy[1], self._odom_heading)
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
        self._update_flank_memory(xy)
        # accumulate heading wander for the current GPS baseline - decides
        # whether that leg may teach the compass (item 23). Bookkeeping
        # only, and deliberately before the early returns below so a leg
        # spanning a stop is still judged on its full history.
        if self.emergency_stop_active:
            # Ahead of every other hold, including the blind hold: this one
            # is a human with a finger on a button, and nothing sensed can
            # outrank it. Only reachable with terminate_on_stop=False -
            # otherwise on_emergency_stop has already ended the run.
            self.send_speed_cmd(0, 0)
            self._last_commanded_speed = 0.0
            return

        if self._route_hold_is_absolute():
            # WAITING (no QR shown yet), ARRIVED, LOST or FAILED, with no
            # creep speed configured: the robot must sit still, FULL STOP,
            # and in particular must not reverse.
            #
            # It used to. The hold was applied as min(speed, 0) on the
            # finished command, which does nothing to a NEGATIVE speed -
            # so the avoidance state machine still ran underneath, and
            # anything close enough in front (a QR code held up to the
            # camera, for instance) sent it backing away while it was
            # supposed to be parked. Field-reported 2026-09-04.
            #
            # Returning here instead of capping is what actually stops it:
            # the state machine never runs, so nothing can command a
            # maneuver. The depth pipeline is untouched and every sensing
            # handler keeps running and logging - on_obstacle_zones still
            # updates the streaks, obstdet3d_zones still publishes - so
            # this suppresses the REACTION, not the sensing, and the
            # moment the hold lifts the streaks are already current.
            if self.state != State.DRIVE:
                # do not resume a half-finished maneuver when it lifts
                self._enter_drive()
            self.send_speed_cmd(0, 0)
            self._last_commanded_speed = 0.0
            self._last_commanded_steering = 0.0
            return

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
            # _update_blind_state. Do NOT hand this to the avoidance state
            # machine, which would read the centre's 0.0 fail-safe as an
            # obstacle and answer by reversing, i.e. by moving in the one
            # direction nothing watches at all. Placed ahead of every other
            # hold because it is the most fundamental of them: the others
            # act on what was sensed, this one applies when nothing was.
            # Whether this stands still or crawls forward is
            # blind_creep_speed - see _blind_command.
            speed, steering_angle = self._blind_command(xy, dt)
            self._last_commanded_speed = speed
            self._last_commanded_steering = steering_angle
            self.send_speed_cmd(speed, steering_angle)
            return

        if self.joint_escape_active:
            # ahead of the state machine: nothing it could decide is more
            # important than getting the rear off whatever it is on
            command = self._joint_escape_command()
            if command is not None:
                speed, steering_angle = command
                self._last_commanded_speed = speed
                self._last_commanded_steering = steering_angle
                self.send_speed_cmd(speed, steering_angle)
                return

        if self.joint_escape_active:
            # Ahead of the state machine and of every sensed hold: nothing
            # any of them could decide matters more than getting the rear
            # off whatever it has run onto - see _enter_joint_escape.
            command = self._joint_escape_command()
            if command is not None:
                speed, steering_angle = command
                self._last_commanded_speed = speed
                self._last_commanded_steering = steering_angle
                self.send_speed_cmd(speed, steering_angle)
                return

        command = self._road_lost_command(xy, dt)
        if command is not None:
            speed, steering_angle = command
            self._last_commanded_speed = speed
            self._last_commanded_steering = steering_angle
            self.send_speed_cmd(speed, steering_angle)
            return
        if self.ground_hazard_active and not (
                self.ground_hazard_retreat and self.state == State.BACKING_UP):
            # confirmed drop-off/staircase - stay stopped every cycle,
            # don't let the normal state machine drive through it.
            # The one motion allowed through is the retreat on_ground_hazard
            # started: holding 0 here as well would cancel it on the very
            # next cycle and re-create the freeze it exists to avoid, since
            # the hazard cannot clear until the robot has actually moved
            # back from the edge.
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
        if self.side_thresholds_lateral:
            side_lateral = math.sin(self.side_zone_inner_bearing)  # see on_obstacle_zones
            side_turning_dist = self.side_lateral_turn_m
            left_room = None if self.left_dist is None else self.left_dist * side_lateral
            right_room = None if self.right_dist is None else self.right_dist * side_lateral
        else:
            side_turning_dist = self.turning_dist * self.side_turning_dist_factor
            left_room, right_room = self.left_dist, self.right_dist
        left_clear = left_room is None or left_room > side_turning_dist
        right_clear = right_room is None or right_room > side_turning_dist
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
            if self.in_escape_mode:
                # a K-turn, not a retreat - see _escape_backup_steering
                steer = self._escape_backup_steering()
            else:
                steer = self.turn_sign * self.current_avoid_steering if self.state == State.TURNING else 0
            self._enter_backing_up(steer)

        # --- STATE MACHINE ---
        if self.avoid_obstacles and self.state == State.BACKING_UP:
            speed, steering_angle = -self.backup_speed, self._next_retrace_steering(dt)
            elapsed = self.time - self.state_start_time
            boost = self.escape_backup_time_boost if self.in_escape_mode else 1.0
            min_bt, max_bt = self.min_backup_time * boost, self.max_backup_time * boost

            # A standing emergency must clear before handing back to
            # TURNING. The two tests were not complements: BACKING_UP
            # released on any_side_clear (an OR over the sides, with an
            # unknown side counting as clear), while is_emergency fires
            # when EITHER side - or the centre, or now the corridor - is
            # under its stop threshold. So one clear flank was enough to
            # start turning while the other was still inside stop_dist,
            # and the very next cycle's centralized safety check aborted
            # straight back to BACKING_UP. Measured over the 2026-09-05
            # runs: 29% of the 279 entries into TURNING already had
            # stop_streak >= stop_confirm_frames at the moment of entry,
            # and 138 of 278 exits from TURNING were that abort - a
            # 1.1-second cycle of reverse-1.0s / turn-0.1s that gains
            # 0.2m of retreat and no heading at all.
            #
            # max_backup_time still forces the issue, so this cannot
            # deadlock: if the reading never clears the robot gives up
            # waiting and turns anyway, exactly as before.
            still_emergency = self.stop_streak >= self.stop_confirm_frames
            ready = any_side_clear and not still_emergency
            if (elapsed > min_bt and ready) or elapsed > max_bt:
                if elapsed > max_bt and not ready:
                    print(self.time, 'max backup time reached, turning despite %s'
                           % ('something still inside stop_dist' if still_emergency
                              else 'all zones blocked'))
                else:
                    print(self.time, 'done backing up, open zone detected, start turning')
                self._enter_turning()

        elif self.avoid_obstacles and self.state == State.TURNING:
            # current_avoid_steering/current_scan_min_sweep are severity-
            # scaled once, at entry (_enter_turning) - see item 13. Full
            # avoid_steering/scan_min_sweep still apply unchanged for a
            # close/severe encounter or in escape mode; a shallow/diagonal
            # graze gets a smaller, quicker correction instead.
            speed = self.avoid_speed
            steering_angle = self._limit_tail_swing(self.turn_sign * self.current_avoid_steering)
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
                steering_angle = self._limit_tail_swing(
                    max(-self.realign_max_steering,
                        min(self.realign_max_steering, error * self.realign_gain)))
                speed = self._adaptive_speed(dt) * self.realign_speed_factor

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
            grass_side, road_side = self._only_clear_side_is_grass()
            if grass_side and self._road_side_has_gap(road_side):
                print(self.time, 'the clear side is not road (%.2f vs %.2f) - going round on the road side instead'
                       % (self.left_road_frac if road_side < 0 else self.right_road_frac,
                          self.right_road_frac if road_side < 0 else self.left_road_frac))
                self._forced_turn_sign = road_side
                self._enter_turning()
                speed = self.avoid_speed
                steering_angle = self._limit_tail_swing(self.turn_sign * self.current_avoid_steering)
            elif not any_side_clear or grass_side:
                if grass_side:
                    print(self.time, 'the only clear side is not road (%.2f vs %.2f) and there is no room past '
                                      'the obstacle - backing up instead of leaving the road'
                           % (self.left_road_frac, self.right_road_frac))
                    self._enter_backing_up(0, use_retrace=True)
                elif self.in_escape_mode:
                    # Retracing means going back the way we came, which in
                    # escape mode is by definition the direction that has
                    # already failed - and it gives back the heading the
                    # last forward leg gained. Swing out of it instead.
                    print(self.time, 'all zones blocked and escaping - backing up on a countersteer '
                                      '(K-turn) instead of retracing', self.last_obstacle)
                    self._enter_backing_up(self._escape_backup_steering(), use_retrace=False)
                else:
                    print(self.time, 'all zones blocked, backing up straight to find room (retracing path)', self.last_obstacle)
                    self._enter_backing_up(0, use_retrace=True)
                speed, steering_angle = -self.backup_speed, self._next_retrace_steering(dt)
            else:
                print(self.time, 'obstacle nearby, start turning', self.last_obstacle)
                self._enter_turning()
                speed = self.avoid_speed
                steering_angle = self._limit_tail_swing(self.turn_sign * self.current_avoid_steering)

        elif self.waypoint_reached:
            speed, steering_angle = 0, 0

        else:
            speed = self._adaptive_speed(dt)
            # _limit_tail_swing was applied to TURNING and REALIGNING but
            # not to DRIVE, on the reasoning that ordinary cruising does
            # not steer hard enough to sweep anything. In route mode it
            # does: the router raises the ceiling to 40deg at a fork, and
            # in the 2026-09-06 run 115237 crash DRIVE was commanding
            # +35deg with the left zone at 0.76m. The chassis geometry
            # that makes a hard turn sweep the flank does not care which
            # state asked for it, so the limit belongs on every
            # forward-driving command, not on two of the three.
            steering_angle = self._limit_tail_swing(self._drive_steering(dt))

        # Route hold (item 28). Applied HERE, as a cap on the finished
        # command rather than as an early return near the top, on purpose:
        # every hold this file already has (blind, ground_hazard, bumper)
        # is a safety stop that must pre-empt the state machine, whereas
        # this one is a navigation decision - "there is nowhere to go yet"
        # or "we are there" - and obstacle avoidance must keep full
        # authority underneath it. Capping only the FORWARD speed leaves
        # an avoidance backup free to run while the robot waits.
        #
        #   'creep' - no route yet: before the start QR is read, or during
        #             a re-plan. route_hold_creep_speed decides whether
        #             that means standing still (0.0, the default) or
        #             inching forward.
        #   'stop'  - arrived, lost, or no route exists. A real stop.
        if self.route_hold == 'creep':
            # only the crawling form reaches here - an absolute hold has
            # already returned above, before the state machine ran
            speed = min(speed, self.route_hold_creep_speed)
        elif self.route_speed_limit is not None and self._route_speed_limit_applies():
            # Slowing into a planned fork, or while recovering back onto
            # the corridor. Both are situations where the steering has to
            # do something large and the cost of getting it wrong is
            # leaving the path, and both get easier at half the speed: the
            # turn fits in less ground, and every adaptive stopping
            # distance downstream shrinks with it. Never speeds anything
            # up - it is a cap, applied after the state machine, so an
            # avoidance maneuver already going slower stays slower.
            speed = min(speed, self.route_speed_limit)

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
