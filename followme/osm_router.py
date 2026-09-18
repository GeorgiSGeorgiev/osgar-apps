"""
  OSM route planning for Matty - Robotour 2026 (Stromovka).

  Turns "here is a target coordinate" into "follow THIS path, and at that
  fork go left", using an OpenStreetMap extract downloaded ahead of time
  (osm_fetch.py). It plans once when a QR code is read, then every cycle
  publishes a single aim point a few meters ahead along the planned route,
  plus how much the follower should trust it right now.

  --- What problem this actually solves ---

  tulak_obstacle.py can already drive to a GPS target: read a QR, get
  bearing_to_target, steer toward it, blended against the RedRoad mask.
  On open ground that works. In a park it is exactly what drives onto the
  grass, because a bearing points THROUGH whatever lies between the robot
  and the target - a lawn, a flowerbed, a pond. Measured on the 2026-08-29
  Stromovka runs against the real OSM data for that park: the median GPS
  fix sits 1.25m from a mapped path, but three separate multi-minute
  episodes have the robot 6-35m away from any path, sustained for 20-60s
  at a time. Those are not GPS noise - noise does not hold one direction
  for a minute - they are the robot genuinely out on the grass, and run
  125548 spent its whole length ~14m off-path (median), 24m from the
  nearest mapped footway.

  So the fix is not a better bearing, it is a better TARGET: aim at a
  point on a mapped path a few meters ahead instead of at a destination
  200m away. Everything downstream - the road mask, the depth blend, the
  whole avoidance state machine - is unchanged and still has priority.
  This module only answers "which way is the route", never "is it safe".

  --- Division of labour (this is the important part) ---

  Three imperfect inputs, and the question is not which to trust but what
  each is actually an instrument FOR. Two measurements off the 2026-08-29
  Stromovka logs settle it (see replay_osm_router.py to reproduce both):

  1. The road mask supplies NO restoring signal toward the route. While
     1-8m off the planned corridor it steered back toward it in only
     40-48% of frames - worse than a coin flip - with a mean contribution
     of -0.7 to -1.0 deg, i.e. very slightly AWAY. That is not a defect:
     it is a lane-KEEPING sensor with no notion of which lane, so once the
     robot is on the lawn and the lawn looks drivable, nothing in it ever
     says "the road is over there". This is the mechanism behind the
     field report that Matty turns toward the grass and keeps going.

  2. The mask degrades predictably with camera-to-road misalignment. With
     the camera within 15 deg of the road axis it fragments into multiple
     blobs in 27% of frames and its whole-mask centroid lands >10% of
     half-width away from the largest blob in 7%. At 45-60 deg off-axis
     those become 51% and 30% - a 4x rise in exactly the failure where the
     centroid sits between a road blob and a lawn blob and points at
     neither. 45-60 deg off-axis is precisely the state an avoidance turn
     leaves the robot in.

  So the split is by FREQUENCY BAND and by question, not by trust level:

    depth / avoidance - is the way ahead passable, right now. Geometric,
                     never hallucinates a surface. Overrides everything,
                     unchanged by any of this.
    RedRoad mask   - fast lateral centring within whatever surface is
                     ahead, at 10Hz. Better at this than any 1-3m GPS
                     fix. Faded out as the camera swings off the road
                     axis, per measurement 2.
    OSM route      - the LOW-frequency terms: which branch at this fork,
                     and how far off the corridor am I. GPS position
                     error is noisy but BOUNDED and roughly zero-mean;
                     the mask's error is unbounded and self-reinforcing
                     (measurement 1: a 60-second, 35-metre excursion).
                     For the slow terms, bounded-but-noisy beats
                     smooth-but-divergent, and heavy low-passing turns
                     the former into something usable.

  That asymmetry - bounded vs divergent - is the whole answer to "three
  unstable sensors". The map does not have to be accurate. It has to be
  non-divergent, and it is.

  --- Three control situations, not one authority knob ---

  _guidance() names the situation instead of interpolating one number,
  because these want genuinely different control:

    CORRIDOR (the default) - the route is a slow additive BIAS on top of
      the mask, never a blend. A blend at 45% authority lets the mask
      cancel the only restoring signal that exists; an additive bias
      cannot be outvoted, it moves the equilibrium the mask settles into.
      Gains are gentle (~3 deg per metre of cross-track, capped at 15 deg)
      precisely because GPS is noisy: this is the slow term.

    JUNCTION - a planned fork. The mask has no opinion about which branch
      leads to the target, so here the route does take over (authority
      0.85). Critically it also gets a much larger steering ceiling:
      Matty's turning radius is 0.16/tan(steer/2), so 20 deg gives 0.91m
      and a 90 deg turn swings wide across the corner, while 40 deg gives
      0.44m and fits inside a path. The rate limit is relaxed to match
      (winding on 40 deg at 30 deg/s would take most of the junction) and
      speed is capped so the turn has room.

    RECOVERY - confirmed off the corridor. Route dominant, short
      lookahead so the return angle is steep, speed capped.

  --- Following the route: pure pursuit on arclength, not "next node" ---

  The route is one polyline. The robot's position along it is a single
  scalar `s` (meters from the start), advanced two ways:

    - odometry, every pose2d (10Hz): the forward component of the pose2d
      displacement. This needs no frame alignment between the odometry
      frame and the map - only the SIGNED distance travelled, which is
      what the projection onto the robot's own heading gives. Reversing
      during an avoidance maneuver correctly moves `s` backwards.
    - GPS, every fix (1Hz): project the fix onto the route and pull `s`
      toward that projection (gps_snap_alpha).

  Searching for that projection only WITHIN A WINDOW around the current
  `s` is what makes this robust where plain nearest-point snapping is not.
  Stromovka's paths run parallel a few meters apart and the route crosses
  itself; a global nearest-point search jumps between them on GPS noise
  alone. A +-15m window cannot, because the robot cannot have moved that
  far since the last fix.

  The aim point is then the route point at `s + lookahead`, clamped so it
  never reaches past the next junction by more than junction_overshoot_m.
  Without that clamp a long lookahead cuts the corner at a fork - and
  cutting a corner in a park means driving across the grass inside it,
  which is the exact thing this module exists to prevent.

  --- Which ways Matty may use ---

  Filtering lives here (in config), not in the downloaded map file, so it
  can be changed and re-checked without re-downloading anything.

  EXCLUDED outright, and none of these are style preferences:
    steps        - unclimbable. Stromovka has 10 of them inside the area
                   driven on 2026-08-29 and they connect otherwise
                   sensible paths, so a router that ignores this WILL
                   route over a staircase.
    trunk/primary/secondary/motorway and their _link - live traffic.
    construction/proposed/raceway/platform/corridor/elevator - not roads.
    access=private/no, foot=no - not ours to drive on.
    anything with sac_scale - mountain-hiking path grading; if a way
                   needs one, it is not a park path.

  Everything else is allowed but COSTED: cost = length x highway penalty x
  surface penalty. A gravel or dirt path is passable and stays in the
  graph, it just has to be meaningfully shorter to win against asphalt.
  All penalties are clamped to >= 1.0 so the A* heuristic (straight-line
  distance) stays admissible.

  A path that exists on the ground but not in OSM is, by construction,
  never routed onto - the graph simply has no edge there. That is the
  "if some road seems legit but is not on OSM, do not take it" rule, and
  it needs no separate check: what enforces it in practice is the
  corridor monitor below noticing that the robot has left the planned
  polyline, whatever the reason.

  --- When things go wrong ---

  Cross-track distance from the planned route is watched continuously
  (smoothed, since a single fix means little):

    < corridor_soft_m         normal. Thresholds are sized from the
                              measured 2026-08-29 data: on-path fixes sit
                              at a 1.25m median, real excursions ran
                              6-35m, so there is a wide, empty band
                              between "noisy" and "actually off".
    > corridor_hard_m, held for corridor_hard_confirm_sec:
        ... and the robot is close to some OTHER mapped way
                              -> avoidance has put us on a different real
                              path. Re-plan from here; nothing is wrong.
        ... and it is not     -> RECOVERING: aim at a much shorter
                              lookahead so the return angle is steep, and
                              raise authority.
    farther than lost_dist_m from ANY allowed way, confirmed
                              -> LOST. Publish hold='stop'. There is no
                              honest automatic answer here and the robot
                              is somewhere it should not be; stopping is
                              what a human can recover from.

  Re-planning is bounded (max_replans). A stall detector re-plans once if
  `s` has not advanced for stall_sec while the robot is nominally driving,
  penalising the edge it is stuck on - but note it cannot distinguish
  "blocked" from "operator paused the robot", so it is deliberately slow.

  --- Run modes: the QR command protocol ---

  The QR code is the whole operator interface, so that changing what the
  robot is doing never means editing a config in the field:

    release the emergency stop  -> WAITING. Stopped, holding, waiting to
                                   be shown something.
    QR "start" / "go"           -> FREE. Road mask plus obstacle
                                   avoidance only: drive forward and stay
                                   on the road. No route, no GPS bearing,
                                   no corridor bias. This is the debug
                                   mode - it exercises everything except
                                   this module's own guidance.
    QR "<lat>, <lon>"           -> plan a route and FOLLOW it (the
                                   competition mode). Needs a GPS fix
                                   first; the module waits for one and
                                   says so rather than failing.
    QR "stop"/"abort"/"cancel"  -> forget the target, back to WAITING.
    press + release the e-stop  -> back to WAITING from anywhere.

  All matched case-insensitively on the whole decoded string. A code that
  is neither a command nor a coordinate pair is logged and ignored, so a
  stray sticker cannot change the mode. Every transition is idempotent
  against the camera re-reading a held-up code ~10x/second: the guard is
  on "does this change the state", not on "is this the same text as last
  time", which is what lets start -> stop -> start work while start ->
  start does nothing.

  Showing coordinates again after a "stop" DOES re-plan - the stop
  cleared the target, so it is a new one. That is the intended way to
  restart a leg. Showing the same coordinates while already following
  them does not.

  Multi-checkpoint Robotour runs need nothing extra: arrive, get shown
  the next code, continue.

  Two config gates, both defaulting to the full behaviour:

    enable_gps_routing=False  coordinate codes are ignored; start/stop
                              still work. "Just drive on the road" with
                              no rewiring.
    wait_for_start_qr=False   no start gate at all - boot straight into
                              FREE. The behaviour from before this
                              protocol existed.

  The e-stop reset needs terminate_on_stop=False in the app, otherwise
  the process dies on the press and there is no release to observe. The
  app then holds itself stopped for as long as the button is engaged
  (see on_emergency_stop there) rather than trusting the firmware to
  refuse motion - which this module has no way to verify.

  --- What is NOT handled ---

    - oneway is ignored. Every allowed way is bidirectional. In a park at
      0.5m/s this is the right call; on a route that uses service roads
      shared with cars it is a real omission.
    - barrier nodes (gates, bollards, lift gates) are not modelled. A
      bollard is fine for Matty and a closed gate is not, and OSM rarely
      distinguishes them reliably enough to act on.
    - a route that starts by requiring a U-turn is planned, not avoided,
      beyond uturn_penalty_m biasing the first step away from it. Matty
      is articulated (min radius ~0.39m per matty.py) so a U-turn on a 3m
      park path is geometrically possible but was never tested; place the
      robot facing roughly along the intended direction at the start.
    - elevation. Stromovka is flat enough; a park with a steep slope
      would want an incline penalty.

  Nothing here has been run against hardware. It HAS been replayed
  against the recorded 2026-08-29 Stromovka logs (see test_osm_router.py),
  which is what the threshold numbers above come from.
"""
import datetime
import heapq
import json
import math
import os

import numpy as np
import xml.etree.ElementTree as ET

from osgar.node import Node

from tulak_obstacle import haversine_distance, normalize_angle, parse_gps_qr


# Way types Matty may drive on at all. Anything not listed is excluded -
# a whitelist rather than a blacklist, deliberately: a new/unusual
# highway=* value appearing in the data should mean "do not route over
# this thing I have never heard of", not "assume it is fine".
DEFAULT_ALLOWED_HIGHWAY = [
    'footway', 'path', 'pedestrian', 'cycleway', 'track', 'bridleway',
    'service', 'living_street', 'residential', 'unclassified', 'tertiary',
]

# Multiplies the length of an edge. >1 means "usable, but prefer not to".
# All values are clamped to >=1.0 when the graph is built, so the A*
# straight-line heuristic stays admissible.
# Half-width of the drivable surface, in metres, by highway type - used
# ONLY to turn "distance to the nearest mapped centreline" into "how far
# past the edge of the road am I", which is the number that decides
# whether the robot is off the road. See RoadGraph.seg_halfwidth.
#
# OSM's own `width` tag is used wherever it exists, but it is present on
# only 2% of the ways in the Suchdol and Stromovka extracts, so a
# per-type default carries almost all of the work. These are deliberately
# GENEROUS (a bit wider than the typical real surface): being wrong in
# the tolerant direction costs a late reaction, being wrong in the tight
# direction means fighting the road mask while the robot is legitimately
# on the road, which is worse and harder to notice.
#
# Not surveyed - they are conventional widths for these types. The one
# that matters most at Stromovka is `footway`, which is 54% of that
# extract's ways.
DEFAULT_HIGHWAY_HALFWIDTH = {
    'footway': 1.2,
    # `path` is NOT a narrow footway - at Stromovka it is the narrow one.
    # Of the 51 width-tagged ways in that extract, the 2 tagged `path`
    # are 0.5 and 1.0m wide, against a footway median of 3.0m. Carrying
    # the footway figure here made off_road_m under-report on exactly
    # the paths where leaving the road is easiest.
    'path': 0.6,
    'steps': 1.0,
    'cycleway': 1.5,
    'bridleway': 1.5,
    'pedestrian': 2.5,
    'track': 2.0,
    'service': 2.5,
    'living_street': 3.0,
    'residential': 3.0,
    'unclassified': 3.0,
    'tertiary': 3.5,
    'secondary': 4.0,
}
DEFAULT_HALFWIDTH_M = 2.0

DEFAULT_HIGHWAY_PENALTY = {
    'footway': 1.0,
    'pedestrian': 1.0,
    'cycleway': 1.0,
    'path': 1.2,
    'track': 1.4,
    'bridleway': 1.6,
    'service': 1.2,
    'living_street': 1.3,
    'residential': 1.8,     # has cars on it
    'unclassified': 1.8,
    'tertiary': 2.5,        # has more cars on it
}

# Surface is what actually decides whether Matty's wheels and the depth
# camera have a good time. Unsurveyed (None) is treated as neutral rather
# than as bad - in Stromovka 27 of 164 ways in the driven area carry no
# surface tag at all and most of them are ordinary park paths.
DEFAULT_SURFACE_PENALTY = {
    'asphalt': 1.0,
    'paved': 1.0,
    'concrete': 1.0,
    'concrete:plates': 1.1,
    'paving_stones': 1.1,
    'compacted': 1.1,
    'fine_gravel': 1.2,
    'sett': 1.3,            # cobbles - drivable, rattles the IMU
    'cobblestone': 1.4,
    'wood': 1.3,            # boardwalk
    'gravel': 1.5,
    'ground': 1.7,
    'dirt': 1.7,
    'earth': 1.7,
    'unpaved': 1.7,
    'grass': 3.0,           # in the graph only because a mapped grass
                            # track still beats leaving the map entirely
    'sand': 3.0,
    'mud': 4.0,
}

EXCLUDED_ACCESS = ('private', 'no', 'customers', 'permit', 'delivery')
PERMISSIVE_FOOT = ('yes', 'designated', 'permissive', 'official')

METERS_PER_DEG_LAT = 111320.0


def _way_penalty(tags, allowed, highway_penalty, surface_penalty,
                  default_highway_penalty, default_surface_penalty):
    """Cost multiplier for a way, or None if Matty must not use it at all.

    Access checks come first and are absolute; the rest is preference.
    foot=* overrides access=* because that is how OSM expresses "private
    driveway, but the footpath across it is public", which is common in a
    park with service roads."""
    highway = tags.get('highway')
    if highway not in allowed:
        return None
    if tags.get('sac_scale'):
        return None
    foot = tags.get('foot')
    if foot in ('no', 'private', 'use_sidepath'):
        return None
    if tags.get('access') in EXCLUDED_ACCESS and foot not in PERMISSIVE_FOOT:
        return None
    penalty = highway_penalty.get(highway, default_highway_penalty)
    penalty *= surface_penalty.get(tags.get('surface'), default_surface_penalty)
    if tags.get('smoothness') in ('bad', 'very_bad', 'horrible', 'very_horrible', 'impassable'):
        penalty *= 2.0
    # keep the A* heuristic admissible - see module docstring
    return max(1.0, penalty)


class Route:
    """A planned path as a polyline, addressed by arclength.

    Everything the follower asks is a question about a scalar position
    `s` along this polyline: where do I aim, how far to the next junction,
    how far off am I, how much is left. Keeping it one-dimensional is what
    makes the tracking robust - see the module docstring."""

    def __init__(self, points_ll, points_xy, junction_flags, goal_ll, meta=None,
                 corner_angle_rad=math.radians(35)):
        self.points_ll = list(points_ll)
        self.xy = np.asarray(points_xy, dtype=float)
        self.junction = np.asarray(junction_flags, dtype=bool)
        self.goal_ll = goal_ll
        self.meta = meta or {}
        deltas = np.diff(self.xy, axis=0)
        seg_len = np.hypot(deltas[:, 0], deltas[:, 1]) if len(deltas) else np.zeros(0)
        self.cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.total = float(self.cum[-1]) if len(self.cum) else 0.0
        self._seg_len = seg_len

        # A sharp bend is a turn even when no other path meets it. Nothing
        # in the graph marks one - `junction` counts DEGREE, so a path that
        # doubles back on itself, a switchback or a corner around a
        # building all look like ordinary straight going, and the follower
        # cruises into them at full lookahead, full speed and the 20 deg
        # cruising steering ceiling. Measured along real Stromovka and
        # Suchdol routes, away from any topological junction, the aim point
        # sat within 6-8 deg of the road direction at p90 but reached
        # 19-70 deg at p99 and 116-164 deg at worst - all of it at these
        # unmarked corners. That is the route asking Matty to drive off the
        # path, which is exactly what destabilises the road mask.
        #
        # So corners join junctions in `turn_s`: same short lookahead
        # clamp, same raised steering ceiling, same reduced speed.
        self.corner = np.zeros(len(self.xy), dtype=bool)
        for i in range(1, len(self.xy) - 1):
            before = self.xy[i] - self.xy[i - 1]
            after = self.xy[i + 1] - self.xy[i]
            if min(np.hypot(*before), np.hypot(*after)) < 0.5:
                continue  # too short to have a meaningful direction
            turn = abs(normalize_angle(math.atan2(after[0], after[1])
                                        - math.atan2(before[0], before[1])))
            self.corner[i] = turn >= corner_angle_rad

        self.junction_s = [float(self.cum[i]) for i in range(len(self.xy))
                           if self.junction[i] or self.corner[i]]
        # how sharp each of those turns is (radians). A 35deg bend and a
        # 90deg T both need junction treatment, but not the same speed -
        # see _guidance's speed cap.
        # How sharp each of those turns is (radians). This number sets the
        # speed cap at the fork, so what it means matters: it is how far
        # THIS ROUTE bends here, not how many ways meet here.
        #
        # It used to be max(geometric, 90deg) at any topological junction -
        # i.e. every place two paths meet was treated as a potential T and
        # capped at junction_speed_limit, even where the plan goes dead
        # straight through. On campus and park paths, which fork every
        # 15-25m, that is most of the route. Measured over the 2026-09-05
        # runs: 52% of junction-mode episodes had a real road-bearing
        # change under 10 degrees, 49% of all junction-mode time (767s of
        # 1553s) was spent capped at 0.30 m/s for turns under 25 degrees,
        # and the cap was the binding constraint on 51% of every forward
        # cycle in the session. That is the reported "slows too much
        # during turns, even though they are not sharp" - and it was never
        # a fail-safe, it was this line.
        #
        # The geometry is known here and is the honest answer, so use it.
        # The 90deg assumption survives only where the geometry genuinely
        # cannot be computed (an endpoint, or segments too short to have a
        # direction), which is the case it was written for.
        self.junction_turn = {}
        for i in range(len(self.xy)):
            if not (self.junction[i] or self.corner[i]):
                continue
            turn = math.pi / 2                      # geometry unknown - assume it could be a T
            if 0 < i < len(self.xy) - 1:
                before = self.xy[i] - self.xy[i - 1]
                after = self.xy[i + 1] - self.xy[i]
                if min(np.hypot(*before), np.hypot(*after)) >= 0.5:
                    turn = abs(normalize_angle(math.atan2(after[0], after[1])
                                                - math.atan2(before[0], before[1])))
            self.junction_turn[float(self.cum[i])] = turn

    def __len__(self):
        return len(self.points_ll)

    def xy_at(self, s):
        """Point at arclength s, linearly interpolated, clamped to the ends."""
        if len(self._seg_len) == 0:
            return tuple(self.xy[0])
        s = max(0.0, min(self.total, s))
        i = int(np.searchsorted(self.cum, s, side='right') - 1)
        i = max(0, min(len(self._seg_len) - 1, i))
        if self._seg_len[i] <= 0:
            return tuple(self.xy[i])
        t = (s - self.cum[i]) / self._seg_len[i]
        return tuple(self.xy[i] + t * (self.xy[i + 1] - self.xy[i]))

    def tangent_at(self, s):
        if len(self._seg_len) == 0:
            return (1.0, 0.0)
        s = max(0.0, min(self.total, s))
        i = int(np.searchsorted(self.cum, s, side='right') - 1)
        i = max(0, min(len(self._seg_len) - 1, i))
        d = self.xy[i + 1] - self.xy[i]
        n = math.hypot(d[0], d[1])
        return (d[0] / n, d[1] / n) if n > 0 else (1.0, 0.0)

    def project(self, xy, s_lo, s_hi):
        """Closest point on the route to `xy`, searched ONLY between
        arclengths s_lo and s_hi - see the module docstring for why the
        window is the whole point. Returns (s, signed_cross_track,
        distance) or None if the window contains no segment.

        Cross-track sign: positive means the robot is to the RIGHT of the
        route's direction of travel (standard east/north frame, so a
        positive z cross product means left, and this negates it)."""
        if len(self._seg_len) == 0:
            return None
        first = int(np.searchsorted(self.cum, s_lo, side='right') - 1)
        last = int(np.searchsorted(self.cum, s_hi, side='left'))
        first = max(0, min(len(self._seg_len) - 1, first))
        last = max(first + 1, min(len(self._seg_len), last + 1))

        a = self.xy[first:last]
        b = self.xy[first + 1:last + 1]
        d = b - a
        length2 = d[:, 0] ** 2 + d[:, 1] ** 2
        p = np.asarray(xy, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(length2 > 0,
                          ((p[0] - a[:, 0]) * d[:, 0] + (p[1] - a[:, 1]) * d[:, 1]) / np.where(length2 > 0, length2, 1.0),
                          0.0)
        t = np.clip(t, 0.0, 1.0)
        closest = a + t[:, None] * d
        dist = np.hypot(p[0] - closest[:, 0], p[1] - closest[:, 1])
        k = int(np.argmin(dist))
        s = float(self.cum[first + k] + t[k] * math.sqrt(length2[k]))
        seg_len = math.sqrt(length2[k])
        if seg_len > 0:
            tx, ty = d[k, 0] / seg_len, d[k, 1] / seg_len
        else:
            tx, ty = 1.0, 0.0
        cross = tx * (p[1] - closest[k, 1]) - ty * (p[0] - closest[k, 0])
        return s, -float(cross), float(dist[k])

    def next_turn_angle(self, s, min_turn_rad=0.0):
        """How sharp the next turn ahead is (radians), or 0 if none.
        min_turn_rad skips straight-through forks, same as
        next_junction_dist - the two must agree on what counts as a turn
        or the speed cap ends up describing a different fork from the one
        the lookahead clamp is aiming at."""
        for js in self.junction_s:
            if js > s + 0.5:
                turn = self.junction_turn.get(js, math.pi / 2)
                if min_turn_rad > 0 and turn < min_turn_rad:
                    continue
                return turn
        return 0.0

    def next_turn_signed(self, s, min_turn_rad=0.0):
        """Signed angle of the next qualifying turn ahead: positive when
        the route bends LEFT (anticlockwise), negative right. None when
        there is no turn left or its geometry is unknown.

        Signed, unlike next_turn_angle, because the follower needs the
        DIRECTION and that is the one thing about a junction the GPS bias
        cannot corrupt - it is a property of the polyline."""
        for js in self.junction_s:
            turn = self.junction_turn.get(js, math.pi / 2)
            if js <= s + 0.5 or (min_turn_rad > 0 and turn < min_turn_rad):
                continue
            i = int(np.searchsorted(self.cum, js))
            if i <= 0 or i >= len(self.xy) - 1:
                return None
            before = self.xy[i] - self.xy[i - 1]
            after = self.xy[i + 1] - self.xy[i]
            if min(np.hypot(*before), np.hypot(*after)) < 0.5:
                return None
            # +ve = anticlockwise = left, matching the platform's steering
            return normalize_angle(math.atan2(after[0], after[1])
                                    - math.atan2(before[0], before[1])) * -1.0
        return None

    def next_turn_exit_bearing(self, s, min_turn_rad=0.0, probe_m=4.0):
        """Compass bearing (radians) the route LEAVES the next qualifying
        turn on, measured from the turn node to a point probe_m further
        along - a few metres rather than the immediate segment, which can
        be a node-spacing stub with a meaningless direction. None when
        there is no such turn. A property of the polyline alone, so the
        per-run GPS offset cannot touch it; the follower closes its
        junction hint on it (see tulak_obstacle._route_turn_hint)."""
        for js in self.junction_s:
            turn = self.junction_turn.get(js, math.pi / 2)
            if js <= s + 0.5 or (min_turn_rad > 0 and turn < min_turn_rad):
                continue
            p0 = self.xy_at(js)
            p1 = self.xy_at(min(self.total, js + probe_m))
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            if math.hypot(dx, dy) < 0.5:
                return None
            return math.atan2(dx, dy)            # x=east, y=north -> compass bearing
        return None

    def next_turn_relative(self, s, min_turn_rad=0.0, probe_m=8.0, exit_probe_m=None, cluster_m=0.0):
        """Signed angle (radians, + = left) of the same turn next_turn_signed
        reports, measured between the chord probe_m BEFORE the node and the
        chord probe_m AFTER it instead of the two segments touching it.

        The follower measures a turn by how far its odometry heading has
        rotated, so it needs the direction change between the road it came
        along and the road it leaves on - not a node-spacing stub. At the
        09-15 junction J the branch leaves at 41 deg for 7.9 m, then runs 59
        and 84 deg; the approaches differ the same way (264 -> 239 -> 221
        from the east). None when there is no such turn."""
        for js in self.junction_s:
            turn = self.junction_turn.get(js, math.pi / 2)
            if js <= s + 0.5 or (min_turn_rad > 0 and turn < min_turn_rad):
                continue
            first = last = js
            if cluster_m > 0:
                # Turns closer together than cluster_m are ONE manoeuvre as far
                # as the heading is concerned: measure from before the first
                # to after the last, even if the first is already behind. The
                # -90/+89 jog 6.3 m apart at the start of 171614 nets to ~0
                # and is not armed at all; each node alone read -36 and +78.
                qualifying = [q for q in self.junction_s
                              if not (min_turn_rad > 0 and self.junction_turn.get(q, math.pi / 2) < min_turn_rad)]
                k = qualifying.index(js)
                while k > 0 and first - qualifying[k - 1] <= cluster_m:
                    k -= 1
                    first = qualifying[k]
                k = qualifying.index(js)
                while k + 1 < len(qualifying) and qualifying[k + 1] - last <= cluster_m:
                    k += 1
                    last = qualifying[k]
            a = self.xy_at(max(0.0, first - probe_m))
            p0 = self.xy_at(first)
            p1 = self.xy_at(last)
            # the exit chord may be longer: 171614 starts 1 m from a -90 deg
            # turn followed 4 m later by a +89 deg one - a jog in the mapped
            # footway. 8 m chords read it as -74 deg and the replay pulled
            # right for 45 s; the direction the route actually leaves in is
            # what the follower has to face.
            b = self.xy_at(min(self.total, last + (exit_probe_m or probe_m)))
            if math.hypot(p0[0] - a[0], p0[1] - a[1]) < 0.5 or math.hypot(b[0] - p1[0], b[1] - p1[1]) < 0.5:
                return None
            bearing_in = math.atan2(p0[0] - a[0], p0[1] - a[1])
            bearing_out = math.atan2(b[0] - p1[0], b[1] - p1[1])
            return -normalize_angle(bearing_out - bearing_in)
        return None

    def next_junction_dist(self, s, min_turn_rad=0.0):
        """Distance from s forward to the next TURN - a junction or a sharp
        bend, see self.corner - on the route, or
        the distance to the route end if there is none left (arriving is,
        for the purposes of the lookahead clamp and the authority ramp,
        the same kind of event as a fork: a place not to overshoot).

        min_turn_rad skips forks the route runs straight through. A place
        where paths merely MEET is not an event for a robot that is not
        changing direction there: it needs no shortened lookahead, no
        raised steering ceiling, no reduced speed, and - most of all - no
        handover of authority from the road mask to a 1-3m GPS fix. The
        mask centres better than the fix does on a straight path, which
        is exactly what cruise_authority being lower than
        junction_authority already says. 0 keeps every fork (previous
        behaviour)."""
        for js in self.junction_s:
            if js > s + 0.5:
                if min_turn_rad > 0 and self.junction_turn.get(js, math.pi / 2) < min_turn_rad:
                    continue
                return js - s
        return max(0.0, self.total - s)

    def dist_since_junction(self, s, min_turn_rad=0.0):
        best = None
        for js in self.junction_s:
            if js <= s + 0.5:
                if min_turn_rad > 0 and self.junction_turn.get(js, math.pi / 2) < min_turn_rad:
                    continue
                best = js
        if best is None:
            return float('inf')
        return s - best


class RoadGraph:
    """The drivable subset of an OSM extract, as a routable graph.

    Every node of every allowed way becomes a graph vertex (not just the
    junctions) - the graph is small enough that this costs nothing, and it
    means the planned polyline comes out with the real path geometry
    already in it, no separate geometry lookup."""

    def __init__(self, map_data, allowed=None, highway_penalty=None, surface_penalty=None,
                 default_highway_penalty=1.5, default_surface_penalty=1.2,
                 corner_angle_deg=35.0, highway_halfwidth=None, default_halfwidth=DEFAULT_HALFWIDTH_M,
                 allowed_way_ids=None):
        # bend sharp enough to be treated as a turn - see Route.__init__
        self.corner_angle_rad = math.radians(corner_angle_deg)
        allowed = set(allowed or DEFAULT_ALLOWED_HIGHWAY)
        highway_penalty = highway_penalty or DEFAULT_HIGHWAY_PENALTY
        surface_penalty = surface_penalty or DEFAULT_SURFACE_PENALTY

        nodes_ll = {k: (v[0], v[1]) for k, v in map_data['nodes'].items()}
        bbox = map_data.get('bbox')
        if bbox:
            self.lat0 = (bbox[0] + bbox[2]) / 2.0
            self.lon0 = (bbox[1] + bbox[3]) / 2.0
        else:
            self.lat0 = sum(v[0] for v in nodes_ll.values()) / len(nodes_ll)
            self.lon0 = sum(v[1] for v in nodes_ll.values()) / len(nodes_ll)
        self.m_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(self.lat0))

        self.node_xy = {}
        self.node_ll = {}
        self.adj = {}
        seg_a, seg_b, seg_pen, seg_nodes, seg_half = [], [], [], [], []
        self.skipped_ways = 0
        self.kept_ways = 0
        self.outside_boundary_ways = 0
        halfwidth_by_highway = dict(DEFAULT_HIGHWAY_HALFWIDTH)
        halfwidth_by_highway.update(highway_halfwidth or {})

        for way in map_data['ways']:
            penalty = _way_penalty(way['tags'], allowed, highway_penalty, surface_penalty,
                                    default_highway_penalty, default_surface_penalty)
            if penalty is None:
                self.skipped_ways += 1
                continue
            if allowed_way_ids is not None and str(way.get('id')) not in allowed_way_ids:
                # outside the competition area - see load_boundary_ways
                self.skipped_ways += 1
                self.outside_boundary_ways += 1
                continue
            self.kept_ways += 1
            # how wide this way's surface is, for the off-road test - see
            # DEFAULT_HIGHWAY_HALFWIDTH. An explicit width tag wins where
            # it exists; anything unparseable falls through to the type.
            half = halfwidth_by_highway.get(way['tags'].get('highway'), default_halfwidth)
            try:
                half = max(half, float(way['tags']['width']) / 2.0)
            except (KeyError, TypeError, ValueError):
                pass
            ids = [n for n in way['nodes'] if n in nodes_ll]
            for nid in ids:
                if nid not in self.node_xy:
                    self.node_ll[nid] = nodes_ll[nid]
                    self.node_xy[nid] = self.to_xy(*nodes_ll[nid])
                    self.adj[nid] = []
            for u, v in zip(ids, ids[1:]):
                if u == v:
                    continue
                ux, uy = self.node_xy[u]
                vx, vy = self.node_xy[v]
                length = math.hypot(vx - ux, vy - uy)
                if length <= 0:
                    continue
                cost = length * penalty
                self.adj[u].append((v, cost))
                self.adj[v].append((u, cost))
                seg_a.append((ux, uy))
                seg_b.append((vx, vy))
                seg_pen.append(penalty)
                seg_nodes.append((u, v))
                seg_half.append(half)

        self.seg_a = np.asarray(seg_a, dtype=float).reshape(-1, 2)
        self.seg_b = np.asarray(seg_b, dtype=float).reshape(-1, 2)
        self.seg_penalty = np.asarray(seg_pen, dtype=float)
        self.seg_nodes = seg_nodes
        self.seg_halfwidth = np.asarray(seg_half, dtype=float)
        d = self.seg_b - self.seg_a
        self._seg_d = d
        self._seg_len2 = d[:, 0] ** 2 + d[:, 1] ** 2
        self._seg_len = np.sqrt(self._seg_len2)
        self.degree = {nid: len(nbrs) for nid, nbrs in self.adj.items()}
        # spatial index for snap() - see _build_grid. 100m cells: big
        # enough that a park fits in a handful, small enough that a city
        # cell holds only a few hundred segments.
        self._grid_cell = 100.0
        self._grid = self._build_grid()

    # --- geometry -------------------------------------------------
    def to_xy(self, lat, lon):
        return ((lon - self.lon0) * self.m_per_deg_lon,
                (lat - self.lat0) * METERS_PER_DEG_LAT)

    def to_ll(self, x, y):
        return (y / METERS_PER_DEG_LAT + self.lat0,
                x / self.m_per_deg_lon + self.lon0)

    def _build_grid(self):
        """Uniform grid over the segments, so snap() does not have to touch
        all of them.

        Brute force is fine for a park (8k segments, 0.2ms) and is not fine
        for a city: the whole-Prague extract has 732k segments and measured
        40ms per snap, which runs on every GPS fix and again in the stall
        check. This makes the cost depend on local density instead of map
        size, which is what lets one big map file be practical at all."""
        cell = self._grid_cell
        grid = {}
        for index, ((ax, ay), (bx, by)) in enumerate(zip(self.seg_a, self.seg_b)):
            ix0, ix1 = sorted((int(ax // cell), int(bx // cell)))
            iy0, iy1 = sorted((int(ay // cell), int(by // cell)))
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    grid.setdefault((ix, iy), []).append(index)
        return {key: np.asarray(value, dtype=np.int64) for key, value in grid.items()}

    def _candidates(self, px, py, rings):
        cell = self._grid_cell
        cx, cy = int(px // cell), int(py // cell)
        found = [self._grid[(ix, iy)]
                 for ix in range(cx - rings, cx + rings + 1)
                 for iy in range(cy - rings, cy + rings + 1)
                 if (ix, iy) in self._grid]
        return np.concatenate(found) if found else None

    def snap(self, lat, lon):
        """Nearest point on any allowed way. Returns
        (segment_index, t, distance_m, (x, y)) or None on an empty graph.

        Searches the grid outward ring by ring and only accepts a result
        once it is provably the global nearest: after covering every cell
        within `rings` of the query cell, any segment nearer than
        rings*cell must have been among the candidates, so a best distance
        under that bound cannot be beaten by anything outside. Otherwise it
        widens, and falls back to a full scan if the map turns out to be
        empty around here (which is itself worth knowing - see
        _off_map_reason)."""
        if len(self.seg_a) == 0:
            return None
        px, py = self.to_xy(lat, lon)
        for rings in (1, 2, 4, 8, 16):
            candidates = self._candidates(px, py, rings)
            if candidates is None:
                continue
            result = self._snap_among(px, py, candidates)
            if result[2] <= rings * self._grid_cell:
                return result
        return self._snap_among(px, py, None)

    def _snap_among(self, px, py, indices):
        """The actual point-to-segment minimum, vectorised over `indices`
        (or every segment when None)."""
        seg_a = self.seg_a if indices is None else self.seg_a[indices]
        seg_d = self._seg_d if indices is None else self._seg_d[indices]
        seg_len2 = self._seg_len2 if indices is None else self._seg_len2[indices]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(seg_len2 > 0,
                          ((px - seg_a[:, 0]) * seg_d[:, 0] + (py - seg_a[:, 1]) * seg_d[:, 1])
                          / np.where(seg_len2 > 0, seg_len2, 1.0),
                          0.0)
        t = np.clip(t, 0.0, 1.0)
        cx = seg_a[:, 0] + t * seg_d[:, 0]
        cy = seg_a[:, 1] + t * seg_d[:, 1]
        dist = np.hypot(px - cx, py - cy)
        k = int(np.argmin(dist))
        i = k if indices is None else int(indices[k])
        return i, float(t[k]), float(dist[k]), (float(cx[k]), float(cy[k]))

    def snap_consistent(self, latlon, heading, m_per_deg, radius_m):
        """snap(), but among the ways within radius_m of the fix (or the
        nearest, if that is further) prefer one that runs along `heading`
        (compass radians; ways are undirected, so 180 deg off is aligned).
        Cost = distance + m_per_deg * misalignment in degrees.

        Field case 165145 (2026-09-15): Matty drove south-west (GPS course
        217-220 deg) down a footway mapped at 244 deg while the fix wandered
        up to ~10 m under the trees; a replan from one fix 0.2 m from a
        footway mapped at 287/107 deg put it on that one, 10 m "before" a
        junction it had in fact reached. That way is 67-70 deg off the
        heading, the real one 24-27 deg: at 0.15 m/deg the real one wins
        whenever it is less than ~6 m further away."""
        if len(self.seg_a) == 0 or heading is None:
            return None
        px, py = self.to_xy(*latlon)
        rings = int(math.ceil(radius_m / self._grid_cell)) + 1
        idx = self._candidates(px, py, rings)
        if idx is None:
            return None
        seg_a, seg_d, seg_len2 = self.seg_a[idx], self._seg_d[idx], self._seg_len2[idx]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(seg_len2 > 0,
                          ((px - seg_a[:, 0]) * seg_d[:, 0] + (py - seg_a[:, 1]) * seg_d[:, 1])
                          / np.where(seg_len2 > 0, seg_len2, 1.0), 0.0)
        t = np.clip(t, 0.0, 1.0)
        cx = seg_a[:, 0] + t * seg_d[:, 0]
        cy = seg_a[:, 1] + t * seg_d[:, 1]
        dist = np.hypot(px - cx, py - cy)
        bearing = np.arctan2(seg_d[:, 0], seg_d[:, 1])
        misalign = np.abs((bearing - heading + np.pi / 2) % np.pi - np.pi / 2)
        cost = dist + m_per_deg * np.degrees(misalign)
        cost[dist > max(radius_m, float(dist.min()))] = np.inf
        cost[seg_len2 <= 0] = np.inf
        if not np.isfinite(cost).any():
            return None
        k = int(np.argmin(cost))
        return int(idx[k]), float(t[k]), float(dist[k]), (float(cx[k]), float(cy[k]))

    # --- routing --------------------------------------------------
    def _overlay(self, seg_index, t, virtual_id):
        """Edges connecting a virtual node (the snapped start or goal,
        which sits partway along a real segment) to that segment's two
        real endpoints. Returned as {node_id: [(neighbour, cost), ...]}
        covering BOTH directions, since the goal has to be reachable FROM
        its segment's endpoints while the start has to reach them."""
        u, v = self.seg_nodes[seg_index]
        length = float(self._seg_len[seg_index])
        penalty = float(self.seg_penalty[seg_index])
        to_u = length * t * penalty
        to_v = length * (1.0 - t) * penalty
        return {
            virtual_id: [(u, to_u), (v, to_v)],
            u: [(virtual_id, to_u)],
            v: [(virtual_id, to_v)],
        }

    def plan(self, start_ll, goal_ll, start_heading=None, uturn_penalty_m=0.0,
             blocked_edges=None, snap_heading_m_per_deg=0.0, snap_radius_m=15.0):
        """A* from the point on the map nearest `start_ll` to the point
        nearest `goal_ll`. `start_heading` is a compass bearing (radians,
        0=north, clockwise) used only to bias the very first step away
        from a U-turn - see uturn_penalty_m and the module docstring.

        `blocked_edges` is a set of frozenset({node_a, node_b}) pairs that
        must not be used, so a re-plan can route around an edge the robot
        has demonstrably failed to get through.

        Returns a Route, or None if either end cannot be snapped or no
        path exists."""
        blocked_edges = blocked_edges or set()
        start_snap = self.snap(*start_ll)
        if start_heading is not None and snap_heading_m_per_deg > 0:
            # see snap_consistent - the start is where a wrong branch costs most
            start_snap = self.snap_consistent(start_ll, start_heading, snap_heading_m_per_deg,
                                              snap_radius_m) or start_snap
        goal_snap = self.snap(*goal_ll)
        if start_snap is None or goal_snap is None:
            return None
        s_idx, s_t, s_dist, s_xy = start_snap
        g_idx, g_t, g_dist, g_xy = goal_snap

        START, GOAL = '__start__', '__goal__'
        overlay = {}
        for key, value in self._overlay(s_idx, s_t, START).items():
            overlay.setdefault(key, []).extend(value)
        for key, value in self._overlay(g_idx, g_t, GOAL).items():
            overlay.setdefault(key, []).extend(value)
        if s_idx == g_idx:
            # both ends on the same segment - the direct hop along it may
            # well be the whole route, and without this edge the search
            # would be forced out to an endpoint and back
            direct = abs(g_t - s_t) * float(self._seg_len[s_idx]) * float(self.seg_penalty[s_idx])
            overlay.setdefault(START, []).append((GOAL, direct))

        xy = dict(self.node_xy)
        xy[START], xy[GOAL] = s_xy, g_xy

        def neighbours(nid):
            for item in self.adj.get(nid, ()):
                yield item
            for item in overlay.get(nid, ()):
                yield item

        gx, gy = g_xy

        def heuristic(nid):
            x, y = xy[nid]
            return math.hypot(x - gx, y - gy)

        # First-step bias: a route whose first move is a U-turn is legal
        # but usually not what anyone wants from a robot that has to
        # execute it on a 3m path. Charge it as extra distance rather
        # than forbidding it, so a genuinely dead-end start still works.
        start_bias = {}
        if start_heading is not None and uturn_penalty_m > 0:
            for nbr, _cost in overlay.get(START, ()):
                if nbr not in xy:
                    continue
                nx, ny = xy[nbr]
                bearing = math.atan2(nx - s_xy[0], ny - s_xy[1]) % (2 * math.pi)
                delta = abs(normalize_angle(bearing - start_heading))
                start_bias[nbr] = uturn_penalty_m * (1.0 - math.cos(delta)) / 2.0

        open_heap = [(heuristic(START), 0.0, START)]
        best = {START: 0.0}
        came_from = {}
        closed = set()
        while open_heap:
            _f, g_cost, nid = heapq.heappop(open_heap)
            if nid in closed:
                continue
            if nid == GOAL:
                break
            closed.add(nid)
            for nbr, cost in neighbours(nid):
                if nbr in closed or nbr not in xy:
                    continue
                if frozenset((nid, nbr)) in blocked_edges:
                    continue
                new_cost = g_cost + cost
                if nid == START:
                    new_cost += start_bias.get(nbr, 0.0)
                if new_cost < best.get(nbr, float('inf')):
                    best[nbr] = new_cost
                    came_from[nbr] = nid
                    heapq.heappush(open_heap, (new_cost + heuristic(nbr), new_cost, nbr))
        if GOAL not in came_from:
            return None  # the goal's segment is in a component the start cannot reach

        chain = [GOAL]
        while chain[-1] != START:
            chain.append(came_from[chain[-1]])
        chain.reverse()

        points_xy, points_ll, junction = [], [], []
        for nid in chain:
            x, y = xy[nid]
            points_xy.append((x, y))
            points_ll.append(self.to_ll(x, y))
            # the virtual endpoints are not junctions, they are just where
            # this particular route happens to begin and end
            junction.append(nid not in (START, GOAL) and self.degree.get(nid, 0) >= 3)
        meta = {
            'start_snap_dist_m': s_dist,
            'goal_snap_dist_m': g_dist,
            'cost': best.get(GOAL, 0.0),
            'nodes': len(chain),
        }
        return Route(points_ll, points_xy, junction, goal_ll, meta,
                      corner_angle_rad=self.corner_angle_rad)


class RouteState:
    WAITING = 'waiting'          # stopped, waiting to be shown a QR code
    FREE = 'free'                # "start"/"go" - road following, no route
    NO_FIX = 'no_fix'            # target known, own position not
    PLANNING = 'planning'
    FOLLOWING = 'following'
    RECOVERING = 'recovering'    # confirmed off the planned corridor
    LOST = 'lost'                # not near any mapped way at all
    ARRIVED = 'arrived'
    FAILED = 'failed'            # no route exists to the target
    GPS_LOST = 'gps_lost'        # fixes stopped for too long to keep guessing


# QR command vocabulary, matched case-insensitively on the whole decoded
# string. Anything that is not one of these is tried as a coordinate pair
# (parse_gps_qr), and only if THAT also fails is the code reported as
# unrecognised - so a stray sticker in view cannot change the mode.
START_WORDS = ('start', 'go')
STOP_WORDS = ('stop', 'abort', 'cancel')


def load_boundary_ways(path):
    """OSM way ids inside the competition area, from a JOSM-edited .osm file.

    The organisers' file (documents/robotour2026-ver1.osm) is an extract with
    everything OUTSIDE the area marked action='delete': 2272 of 2488 nodes and
    142 of 195 ways. What survives is the area, so the allowed set is every
    surviving way that carries a highway tag - 38 of them, spanning
    50.1035..50.1067 N, 14.4193..14.4300 E (355 x 747 m), of which 37 are also
    in maps/stromovka.json.

    Ids, not a polygon: the ids are OSM's own and match the map extract
    exactly, so there is no edge case about a way that crosses the border or a
    node that sits on it."""
    root = ET.parse(path).getroot()
    nodes = {n.get('id'): (float(n.get('lat')), float(n.get('lon')))
             for n in root.findall('node') if n.get('lat')}
    ids, lats, lons = set(), [], []
    for way in root.findall('way'):
        if way.get('action') == 'delete':
            continue
        if not any(tag.get('k') == 'highway' for tag in way.findall('tag')):
            continue
        ids.add(str(way.get('id')))
        for nd in way.findall('nd'):
            point = nodes.get(nd.get('ref'))
            if point:
                lats.append(point[0])
                lons.append(point[1])
    bbox = (min(lats), max(lats), min(lons), max(lons)) if lats else (-90.0, 90.0, -180.0, 180.0)
    return ids, bbox


def _find_map_file(path):
    """Config paths are written relative to wherever osgar happens to be
    launched from, which is not always the app directory. Try the obvious
    places and say which ones were tried rather than failing with a bare
    FileNotFoundError halfway through a competition setup."""
    candidates = [path,
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), path)]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    candidates.append(os.path.join(repo_root, path))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError('OSM map file %r not found - tried:\n  %s'
                            % (path, '\n  '.join(candidates)))


class OSMRouter(Node):
    def __init__(self, config, bus):
        super().__init__(config, bus)
        bus.register('route_hint', 'route_plan')

        # map_file may be one path or several. With several, the one to use
        # is chosen from the first GPS fix (see _select_map) instead of by
        # editing the config per site - which is the failure this exists to
        # remove: on 2026-09-04 Matty ran at CZU with the Stromovka map
        # loaded, and every symptom pointed at the QR code or the GPS
        # rather than at the one config line that was wrong.
        # Competition area (item 9): only ways inside it may be planned on.
        # False keeps the whole map routable.
        self.enforce_boundary = config.get('enforce_boundary', False)
        self.boundary_min_ways = config.get('boundary_min_ways', 5)
        # how far outside the area's own bbox still counts as inside, so a
        # start a few metres beyond the edge does not fall back to the whole map
        self.boundary_margin_m = config.get('boundary_margin_m', 25.0)
        # A re-plan starts at the nearest way to the CURRENT fix, so a fix far
        # from any way produces a route from somewhere the robot has never
        # been. Near the building at 170734 the fix sat 10-13 m off and the
        # router re-planned seven times, each from a different wrong place,
        # while the camera was driving down a cobbled square. Above this
        # distance a re-plan waits for a better fix; the first plan of a run
        # is never blocked. 0 = previous behaviour.
        self.replan_max_snap_m = config.get('replan_max_snap_m', 0.0)
        self._snap_note_time = None
        self.boundary_ids = None
        self.boundary_bbox = None
        self._inside_boundary = None
        boundary = config.get('boundary_osm')
        if boundary and self.enforce_boundary:
            boundary_path = _find_map_file(boundary)
            self.boundary_ids, self.boundary_bbox = load_boundary_ways(boundary_path)
            print('competition area %s: %d allowed ways, lat %.4f..%.4f lon %.4f..%.4f'
                   % ((boundary_path, len(self.boundary_ids)) + self.boundary_bbox))

        map_files = config['map_file']
        if isinstance(map_files, str):
            map_files = [map_files]
        self.maps = []
        for entry in map_files:
            self.maps.append(self._load_map(entry, config))
        self._map_selected = len(self.maps) == 1
        self._use_map(self.maps[0])
        if not self._map_selected:
            print('%d maps loaded - the one to use will be chosen from the first GPS fix'
                   % len(self.maps))

        # --- following ---
        self.lookahead_m = config.get('lookahead_m', 8.0)
        self.lookahead_min_m = config.get('lookahead_min_m', 3.0)
        self.recovery_lookahead_m = config.get('recovery_lookahead_m', 4.0)
        # How far past the next junction the aim point may sit. Small on
        # purpose: the aim point crossing a fork before the robot reaches
        # it is what makes a robot cut the corner, and the inside of a
        # corner in a park is grass.
        self.junction_overshoot_m = config.get('junction_overshoot_m', 3.0)
        self.junction_zone_m = config.get('junction_zone_m', 12.0)
        self.junction_exit_m = config.get('junction_exit_m', 6.0)
        # How much the route has to bend at a fork before it is treated as
        # a turn at all - see Route.next_junction_dist. Everything junction
        # mode does (short lookahead, 40deg steering ceiling, 0.85
        # authority, reduced speed) is the right answer to "a turn is
        # coming" and the wrong answer to "two paths meet here and we are
        # going straight on". Below this the fork is ignored and the robot
        # cruises through it, which is what it is doing anyway.
        # 0 restores the previous behaviour (every fork is an event).
        # Deliberately well under gentle_turn_angle_deg: this decides
        # whether there is a turn, not how sharp it is, and a 20deg bend
        # on a wide path still deserves the corridor treatment rather than
        # nothing at all.
        self.junction_min_turn = math.radians(config.get('junction_min_turn_deg', 0.0))
        # metres past the edge of the nearest mapped road - see _gps_update.
        # None until the first fix; 0.0 means "on a road".
        self.off_road_m = None
        self.road_halfwidth_m = None
        # off_road_m at which the aim point is pulled all the way in to
        # lookahead_min_m. 0 disables (cross-track drives the ramp alone).
        self.offroad_lookahead_m = config.get('offroad_lookahead_m', 0.0)
        # Fraction of misaligned_turn_angle the heading must fall back to
        # before the "large heading error IS a turn" latch releases - see
        # _junction_approach. 1.0 restores the previous, un-hysteretic
        # behaviour.
        self.misaligned_turn_release_frac = config.get('misaligned_turn_release_frac', 1.0)
        # how far past a turn node the exit bearing is measured - see
        # Route.next_turn_exit_bearing
        self.turn_exit_probe_m = config.get('turn_exit_probe_m', 4.0)
        # chord length either side of a turn node for turn_rel_deg - see
        # Route.next_turn_relative. Published always (a new key; older
        # followers ignore it).
        self.turn_rel_probe_m = config.get('turn_rel_probe_m', 8.0)
        self.turn_rel_exit_probe_m = config.get('turn_rel_exit_probe_m', None)
        self.turn_rel_cluster_m = config.get('turn_rel_cluster_m', 0.0)
        # Keep publishing the next turn while RECOVERING. It was withheld, so
        # a turn inside a recovery episode was never announced: 170349
        # entered recovery 16.9 m before a 136 deg left hairpin (GPS 6-7 m
        # east of a road the camera was driving down) and the router counted
        # the junction as passed at t=375.7 without the follower ever arming
        # it. False = previous behaviour.
        self.recovery_turn_info = config.get('recovery_turn_info', False)
        # Heading source for the U-turn bias, the misalignment latch, the
        # heading-consistent snap and the replan check: 'compass' (previous)
        # or 'odometry_gps' - see heading_estimator.py.
        self.heading_source = config.get('heading_source', 'compass')
        self.heading_est = None
        if self.heading_source == 'odometry_gps':
            from heading_estimator import OdoGpsHeading
            self.heading_est = OdoGpsHeading(alpha=config.get('heading_offset_alpha', 0.5),
                                             min_baseline_m=config.get('heading_min_baseline_m', 4.0),
                                             valid_travel_m=config.get('heading_valid_travel_m', 60.0))
        # Plan start on the way that runs along the heading, not merely the
        # nearest - see RoadGraph.snap_consistent. 0 = nearest (previous).
        self.snap_heading_m_per_deg = config.get('snap_heading_m_per_deg', 0.0)
        self.snap_heading_radius_m = config.get('snap_heading_radius_m', 15.0)
        # "drifted onto another mapped path" re-plans only when the nearest
        # way runs within this many degrees of the heading AND the planned
        # route here does not. 0 = previous behaviour (any fix near another
        # way, for corridor_hard_confirm_sec). See _drift_replan_consistent.
        self.replan_heading_check_deg = config.get('replan_heading_check_deg', 0.0)
        self._drift_note_time = None
        # Re-plan when the follower says it missed a turn (app.missed_turn).
        # The alternative to a U-turn: the planner is asked for another way
        # round from where the robot actually is, and its uturn_penalty_m
        # still discourages turning back on the spot.
        self.replan_on_missed_turn = config.get('replan_on_missed_turn', False)
        self._misaligned_latched = False

        # --- how much the follower should trust the route bearing ---
        # See the "division of labour" section of the module docstring.
        # cruise is deliberately well below 1: on a straight path the
        # RedRoad mask centres better than a 1-3m GPS fix can.
        self.cruise_authority = config.get('cruise_authority', 0.45)
        self.junction_authority = config.get('junction_authority', 0.85)
        self.recovery_authority = config.get('recovery_authority', 0.9)

        # --- steering headroom, per situation (see _guidance) ---
        # Matty's turning radius is 0.16/tan(steer/2) - 0.91m at 20 deg,
        # 0.44m at 40 deg, 0.39m at the 45 deg platform limit. A fork needs
        # the small radius; ordinary path centring does not and is smoother
        # without it.
        self.cruise_steer_limit_deg = config.get('cruise_steer_limit_deg', 20.0)
        self.junction_steer_limit_deg = config.get('junction_steer_limit_deg', 40.0)
        self.recovery_steer_limit_deg = config.get('recovery_steer_limit_deg', 30.0)
        self.cruise_steer_rate_deg_s = config.get('cruise_steer_rate_deg_s', 30.0)
        self.junction_steer_rate_deg_s = config.get('junction_steer_rate_deg_s', 90.0)
        # How far the line to the aim point may diverge from the road's own
        # direction before the lookahead is shortened - see
        # _limit_aim_offset. Loose at a fork, because a fork IS a turn;
        # tight on a plain path, where anything large means the aim point
        # has cut across a bend.
        self.corridor_max_aim_offset = math.radians(config.get('corridor_max_aim_offset_deg', 50))
        self.junction_max_aim_offset = math.radians(config.get('junction_max_aim_offset_deg', 110))
        self.junction_speed_limit = config.get('junction_speed_limit', 0.3)
        # speed cap interpolates between these two by how sharp the turn is
        # heading error past which the follower is told it is TURNING, not
        # cruising - see _heading_misalignment
        self.misaligned_turn_angle = math.radians(config.get('misaligned_turn_angle_deg', 55))
        self.gentle_turn_speed = config.get('gentle_turn_speed', 0.45)
        self.gentle_turn_angle = math.radians(config.get('gentle_turn_angle_deg', 35))
        self.sharp_turn_angle = math.radians(config.get('sharp_turn_angle_deg', 80))
        self.recovery_speed_limit = config.get('recovery_speed_limit', 0.3)

        # --- corridor monitor ---
        # Sized from the 2026-08-29 Stromovka logs against this same OSM
        # data: on-path fixes 1.25m median / 3m at p90, real off-path
        # excursions 6-35m sustained for 20-60s. See module docstring.
        self.corridor_soft_m = config.get('corridor_soft_m', 4.0)
        self.corridor_hard_m = config.get('corridor_hard_m', 8.0)
        self.corridor_confirm = datetime.timedelta(
            seconds=config.get('corridor_hard_confirm_sec', 6.0))
        self.lost_dist_m = config.get('lost_dist_m', 25.0)
        self.lost_confirm = datetime.timedelta(seconds=config.get('lost_confirm_sec', 10.0))
        self.cross_track_alpha = config.get('cross_track_alpha', 0.4)

        # --- position tracking along the route ---
        self.gps_snap_alpha = config.get('gps_snap_alpha', 0.4)
        self.gps_search_back_m = config.get('gps_search_back_m', 12.0)
        self.gps_search_fwd_m = config.get('gps_search_fwd_m', 18.0)
        self.max_odometry_step_m = config.get('max_odometry_step_m', 0.5)

        # --- planning ---
        self.arrival_dist_m = config.get('arrival_dist_m', 3.0)
        self.uturn_penalty_m = config.get('uturn_penalty_m', 40.0)
        self.max_replans = config.get('max_replans', 6)
        self.replan_cooldown = datetime.timedelta(seconds=config.get('replan_cooldown_sec', 15.0))
        self.recovery_replan_after = datetime.timedelta(
            seconds=config.get('recovery_replan_after_sec', 30.0))
        self.stall_sec = datetime.timedelta(seconds=config.get('stall_sec', 60.0))
        self.stall_dist_m = config.get('stall_dist_m', 2.0)
        self.target_snap_warn_m = config.get('target_snap_warn_m', 15.0)
        # How far the robot, or the target, may be from the nearest mapped
        # way before planning is refused outright - see _off_map_reason.
        # 50m is well past any legitimate excursion onto a lawn (the worst
        # measured at Stromovka was 35m) and nowhere near the 2.5km that a
        # wrong map file produces.
        self.max_snap_dist_m = config.get('max_snap_dist_m', 50.0)
        # --- what to do when GPS stops (item 33) ---
        # Staged, because odometry is good briefly and useless eventually.
        # Measured over the 08-29 and 09-04 runs: 25 dropouts longer than
        # 2.5s, median 3.5s and NONE longer than 4.0s, while odometry path
        # length tracked GPS displacement to a median ratio of 1.04 (p10
        # 0.91, p90 1.20) over 10s windows. So an ordinary dropout is worth
        # riding out on odometry alone, and a long one is not.
        #
        #   0 .. dead_reckon_sec   follow the route on odometry. Cross-track
        #                          is withheld (it cannot be measured), so
        #                          the follower's corridor bias decays to
        #                          zero instead of holding a stale offset -
        #                          a frozen bias is a constant steering
        #                          error with no feedback, which is how you
        #                          spiral off a path rather than hold it.
        #   .. lost_sec            DEGRADED: route still followed, but
        #                          authority ramps down toward the road mask
        #                          and speed is capped. The map's idea of
        #                          where we are is decaying and the steering
        #                          should reflect that.
        #   beyond                 GPS_LOST: route guidance off entirely,
        #                          road following plus obstacle avoidance
        #                          only - the same thing "start"/"go" does,
        #                          which is the best-tested mode there is.
        #                          The target is KEPT, so a returning fix
        #                          re-plans and carries on.
        #
        # Deliberately NOT "dead-reckon onward to the target": the last
        # known point is where the robot already is, so it is not a target,
        # and steering at a distant one on wheel odometry that this project
        # documents as drifting through blind turns would be confidently
        # wrong rather than merely uninformed.
        self.gps_dead_reckon_sec = config.get('gps_dead_reckon_sec', 15.0)
        self.gps_lost_sec = config.get('gps_lost_sec', 45.0)
        self.gps_degraded_speed_limit = config.get('gps_degraded_speed_limit', 0.3)
        # Mirrors of tulak_obstacle's own compass settings - used ONLY for
        # the U-turn bias at plan time, so a mismatch costs a suboptimal
        # first step, never a wrong route. Keep them in sync anyway.
        # --- run-mode gates (QR command protocol) ---
        # Both default to the full competition behaviour. enable_gps_routing
        # False keeps the QR start/stop control but ignores coordinate codes
        # entirely, which is the "just drive on the road" debugging config
        # without having to rewire anything. wait_for_start_qr False removes
        # the start gate as well, so the robot drives as soon as the camera
        # and depth pipeline are up - the behaviour from before this
        # protocol existed.
        self.enable_gps_routing = config.get('enable_gps_routing', True)
        self.wait_for_start_qr = config.get('wait_for_start_qr', True)
        self.compass_sign = config.get('compass_sign', 1)
        offset_deg = config.get('compass_offset_deg')
        self.compass_offset = math.radians(offset_deg) if offset_deg is not None else None
        self.compass_hardiron = math.radians(config.get('compass_hardiron_deg', 0.0))
        self.compass_hardiron_phase = math.radians(config.get('compass_hardiron_phase_deg', 0.0))
        self.log_interval = datetime.timedelta(seconds=config.get('log_interval_sec', 3.0))

        # --- runtime state ---
        self.state = RouteState.WAITING if self.wait_for_start_qr else RouteState.FREE
        self.estop_engaged = False
        self.fail_reason = None
        self.route = None
        self.goal_ll = None
        # bumped on every successful plan - the follower keys off it to
        # drop anything it was accumulating against the previous route
        self.plan_seq = 0
        self.s = 0.0
        self.cross_track = 0.0
        self.last_fix = None
        self.last_fix_time = None
        self.last_heading = None       # OSGAR convention, from 'rotation'
        self.have_heading = False
        self._prev_pose = None
        self.replans = 0
        self.blocked_edges = set()
        self._corridor_bad_since = None
        self._lost_since = None
        self._recovering_since = None
        self._stall_anchor_s = None
        self._stall_anchor_time = None
        self._last_replan_time = None
        self._last_log_time = None
        self._pending_plan_reason = None

    def _load_map(self, entry, config):
        map_file = _find_map_file(entry)
        with open(map_file, encoding='utf-8') as f:
            map_data = json.load(f)
        graph = self._build_graph(map_data, config, None)
        area_graph = None
        if self.boundary_ids is not None:
            area_graph = self._build_graph(map_data, config, self.boundary_ids)
        if area_graph is not None and area_graph.kept_ways < self.boundary_min_ways:
            # this map and that boundary file describe different places -
            # routing on %d ways would strand the robot, so say so loudly and
            # use the whole map instead
            print('WARNING: only %d ways of %s are inside the competition area - that area is elsewhere, '
                   'planning on the whole map' % (area_graph.kept_ways, map_file))
            area_graph = None
        elif area_graph is not None:
            print('competition area available for %s: %d ways inside, %d outside - used while the robot is '
                   'inside the area' % (map_file, area_graph.kept_ways, area_graph.outside_boundary_ways))
        bbox = map_data.get('bbox')
        # The area is printed at boot on purpose: running the wrong map for
        # the site is silent everywhere else, and its symptom (see
        # _off_map_reason) looks like a QR or GPS problem rather than a
        # config one.
        print('OSM map %s: %d routable nodes, %d segments (%d ways excluded), area %s'
               % (map_file, len(graph.node_xy), len(graph.seg_nodes), graph.skipped_ways,
                  'unknown' if not bbox else
                  'lat %.4f..%.4f lon %.4f..%.4f' % (bbox[0], bbox[2], bbox[1], bbox[3])))
        return {'path': map_file, 'bbox': bbox, 'graph': graph, 'area_graph': area_graph}

    def _build_graph(self, map_data, config, allowed_way_ids):
        return RoadGraph(
            map_data,
            allowed=config.get('allowed_highway'),
            highway_penalty=config.get('highway_penalty'),
            surface_penalty=config.get('surface_penalty'),
            default_highway_penalty=config.get('default_highway_penalty', 1.5),
            default_surface_penalty=config.get('default_surface_penalty', 1.2),
            corner_angle_deg=config.get('corner_angle_deg', 35.0),
            highway_halfwidth=config.get('highway_halfwidth'),
            default_halfwidth=config.get('default_halfwidth_m', DEFAULT_HALFWIDTH_M),
            allowed_way_ids=allowed_way_ids)

    def _use_map(self, entry):
        self._map_entry = entry
        self.graph = entry['graph']
        self._inside_boundary = None
        self.map_bbox = entry['bbox']
        self.map_path = entry['path']

    @staticmethod
    def _bbox_contains(bbox, latlon):
        if not bbox:
            return False
        south, west, north, east = bbox
        return south <= latlon[0] <= north and west <= latlon[1] <= east

    @staticmethod
    def _bbox_margin_m(bbox, latlon):
        """How far inside its own bbox a point sits, in meters - the
        distance to the nearest edge. Negative outside."""
        if not bbox:
            return float('-inf')
        south, west, north, east = bbox
        lat, lon = latlon
        m_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(lat))
        return min((lat - south) * METERS_PER_DEG_LAT, (north - lat) * METERS_PER_DEG_LAT,
                   (lon - west) * m_per_deg_lon, (east - lon) * m_per_deg_lon)

    def _apply_boundary(self, lat, lon):
        """Route only inside the competition area while the robot is in it -
        see enforce_boundary. Outside it (a test drive somewhere else) the
        whole map stays routable, so the area file cannot strand the robot at
        another site."""
        entry = getattr(self, '_map_entry', None)
        if entry is None or entry.get('area_graph') is None or self.boundary_bbox is None:
            return
        south, north, west, east = self.boundary_bbox
        margin = self.boundary_margin_m / METERS_PER_DEG_LAT
        inside = (south - margin <= lat <= north + margin
                  and west - margin * 1.6 <= lon <= east + margin * 1.6)
        if inside == self._inside_boundary:
            return
        self._inside_boundary = inside
        self.graph = entry['area_graph'] if inside else entry['graph']
        print(self.time, 'ROUTE: %s the competition area - planning on %d segments'
               % ('inside' if inside else 'outside', len(self.graph.seg_nodes)))

    def _select_map(self, latlon):
        """Pick the loaded map that actually covers where the robot is,
        once - on the first fix. Preference: a map whose bbox contains the
        fix; among those (or if none do) the one whose network is nearest.
        The off-map guard still applies afterwards, so choosing badly from
        a set that covers nowhere near here still refuses to plan rather
        than planning nonsense."""
        if self._map_selected:
            return
        self._map_selected = True

        def snap_distance(entry):
            snap = entry['graph'].snap(*latlon)
            return snap[2] if snap else float('inf')

        # Containing the point is not enough - it matters HOW WELL. A big
        # regional extract can clip a corner of somewhere a small local map
        # covers properly, and a graph truncated at the bbox edge does not
        # fail loudly, it quietly stops offering the ways that continue past
        # the cut. Caught exactly this way: Stromovka (50.1058, 14.4273)
        # falls 89m inside the Suchdol extract's corner while sitting 690m
        # inside the Stromovka one. So among the maps that contain the
        # point, take the one it is furthest INSIDE.
        inside = [m for m in self.maps if self._bbox_contains(m['bbox'], latlon)]
        if inside:
            best = max(inside, key=lambda m: self._bbox_margin_m(m['bbox'], latlon))
        else:
            best = min(self.maps, key=snap_distance)
        self._use_map(best)
        print(self.time, 'ROUTE: using map %s for %.6f,%.6f (%s, nearest way %.0fm)'
               % (os.path.basename(best['path']), latlon[0], latlon[1],
                  ('%.0fm inside its area' % self._bbox_margin_m(best['bbox'], latlon))
                  if inside else 'NO map covers this position',
                  snap_distance(best)))

    # --- inputs ---------------------------------------------------
    def on_missed_turn(self, data):
        """The follower armed a turn, counted it down and never drove it -
        see replan_on_missed_turn. Plan another way round from here rather
        than leave the robot going the wrong way down the route."""
        if not self.replan_on_missed_turn or self.route is None:
            return
        if self.state not in (RouteState.FOLLOWING, RouteState.RECOVERING):
            return
        self._request_replan('the follower missed a turn (%s deg of %s)'
                              % (data.get('remaining_deg'), data.get('turn_deg')))

    def on_rotation(self, data):
        self.last_heading = math.radians(data[0] / 100.0)
        self.have_heading = True

    def _compass_bearing(self):
        """last_heading (OSGAR: 0=east, anticlockwise) as a compass bearing
        (0=north, clockwise), same conversion tulak_obstacle._compass_heading
        does. Returns None without a calibration offset, in which case the
        planner simply skips the U-turn bias rather than applying a
        possibly-mirrored one."""
        if self.heading_est is not None:
            return self.heading_est.heading()      # heading_source 'odometry_gps'
        if not self.have_heading or self.compass_offset is None:
            return None
        heading = normalize_angle(math.pi / 2 - self.compass_sign * self.last_heading
                                   + self.compass_offset)
        if self.compass_hardiron:
            # same hard-iron correction the app applies - see
            # _apply_compass_calibration in tulak_obstacle.py. Only feeds
            # the U-turn bias here, so a mismatch is cheap, but 10 deg is
            # enough to pick the wrong end of the start segment.
            heading = normalize_angle(
                heading + self.compass_hardiron * math.cos(heading - self.compass_hardiron_phase))
        return heading % (2 * math.pi)

    def _enter_waiting(self, reason):
        """Stopped, holding, waiting to be shown a QR - the state the robot
        sits in after the emergency stop is released and after every "stop"
        code. Any previous target is forgotten, so re-showing the same
        coordinates afterwards is a fresh run rather than a no-op."""
        had = self.goal_ll
        self._clear_route()
        if not self.wait_for_start_qr:
            # configured to skip the start gate entirely - go straight to
            # plain road following (the pre-QR-protocol behaviour)
            self.state = RouteState.FREE
            print(self.time, 'ROUTE: %s - wait_for_start_qr=False, road following' % reason)
            return
        print(self.time, 'ROUTE: %s - waiting for a QR (start/go, stop, or coordinates)%s'
               % (reason, '' if had is None else ', cleared target %.6f,%.6f' % had))

    def on_emergency_stop(self, data):
        """Mode reset on the physical button, which is what makes this
        practical to debug with: press, release, and Matty is back to
        waiting for a QR instead of resuming whatever it was doing.

        Only the RELEASE resets, and only after a press this node actually
        saw - matty.py publishes one unsolicited emergency_stop on its
        first status report, and a boot with the button already out must
        not read as a release. Note this needs terminate_on_stop=False in
        the app for the process to survive the press at all; the app holds
        itself stopped meanwhile."""
        if data:
            self.estop_engaged = True
            return
        if not self.estop_engaged:
            return
        self.estop_engaged = False
        self._enter_waiting('emergency stop released')

    def on_qr_code(self, data):
        text = data.strip()
        command = text.lower()

        if command in STOP_WORDS:
            if self.state == RouteState.WAITING:
                return  # already waiting - the camera re-reads a held-up code every frame
            self._enter_waiting('QR "%s"' % text)
            return

        if command in START_WORDS:
            if self.state == RouteState.FREE:
                return
            self._clear_route()
            self.state = RouteState.FREE
            print(self.time, 'ROUTE: QR "%s" - road following only, no route, no GPS guidance'
                   % text)
            return

        parsed = parse_gps_qr(text)
        if parsed is None:
            print(self.time, 'ROUTE: QR code is neither a command nor GPS coordinates:', data)
            return
        if not self.enable_gps_routing:
            print(self.time, 'ROUTE: coordinates %.6f,%.6f ignored - enable_gps_routing=False'
                   % parsed)
            return
        if parsed == self.goal_ll:
            # The oak republishes a decoded code on EVERY frame it stays in
            # view, so this fires ~10x a second for as long as someone
            # holds the card up. Unconditional, including after FAILED:
            # re-running A* over several thousand nodes ten times a second
            # to fail identically each time is the one way this module
            # could starve the rest of the robot of CPU. To deliberately
            # re-plan the same target, show "abort" first.
            return
        print(self.time, 'ROUTE: new target from QR: %.6f, %.6f' % parsed)
        self.goal_ll = parsed
        self.route = None
        self.replans = 0
        self.blocked_edges = set()
        self._pending_plan_reason = 'new-target'
        self.state = RouteState.PLANNING if self.last_fix else RouteState.NO_FIX
        self._try_plan()

    def on_nmea_data(self, data):
        lat, lon = data.get('lat'), data.get('lon')
        if lat is None or lon is None:
            return
        if data.get('lat_dir') == 'S':
            lat = -lat
        if data.get('lon_dir') == 'W':
            lon = -lon
        was_lost = self.state == RouteState.GPS_LOST
        self.last_fix = (lat, lon)
        self.last_fix_time = self.time
        self._apply_boundary(lat, lon)
        if self.heading_est is not None and self.time is not None:
            self.heading_est.update_fix(self.time.total_seconds(), lat, lon)
        self._select_map(self.last_fix)
        if was_lost:
            print(self.time, 'ROUTE: GPS back at %.6f,%.6f - re-planning to the kept target'
                   % (lat, lon))
            self.state = RouteState.PLANNING
            self._pending_plan_reason = 'GPS returned'
            self._try_plan()
            return

        if self._pending_plan_reason is not None:
            self._try_plan()
            return
        if self.route is None:
            return
        self._gps_update(lat, lon)

    def on_pose2d(self, data):
        x_mm, y_mm, heading_cdeg = data
        pose = (x_mm / 1000.0, y_mm / 1000.0, math.radians(heading_cdeg / 100.0))
        if self.heading_est is not None and self.time is not None:
            self.heading_est.update_pose(self.time.total_seconds(), *pose)
        if self.route is not None and self._prev_pose is not None:
            dx = pose[0] - self._prev_pose[0]
            dy = pose[1] - self._prev_pose[1]
            # Forward component in the robot's own frame: this is speed*dt
            # including sign, recovered without needing to know how the
            # odometry frame is rotated relative to the map. Reversing
            # therefore moves `s` backwards, which is what we want during
            # an avoidance backup.
            step = dx * math.cos(pose[2]) + dy * math.sin(pose[2])
            step = max(-self.max_odometry_step_m, min(self.max_odometry_step_m, step))
            self.s = max(0.0, min(self.route.total, self.s + step))
        self._prev_pose = pose
        self._publish_hint()

    # --- planning -------------------------------------------------
    def _clear_route(self):
        self.goal_ll = None
        self.route = None
        self.state = RouteState.WAITING
        self.s = 0.0
        self.cross_track = 0.0
        self._pending_plan_reason = None
        self._corridor_bad_since = None
        self._lost_since = None
        self._recovering_since = None

    def _bbox_note(self, latlon):
        """' - outside the map bbox ...' when that is the obvious cause,
        empty otherwise. Worth spelling out because the symptom of running
        the wrong map file looks nothing like the cause."""
        bbox = self.map_bbox
        if not bbox:
            return ''
        south, west, north, east = bbox
        lat, lon = latlon
        if south <= lat <= north and west <= lon <= east:
            return ''
        return (' - OUTSIDE the loaded map area (%.4f..%.4f, %.4f..%.4f). '
                'Wrong map_file for this location?' % (south, north, west, east))

    def _off_map_reason(self, label, latlon):
        """None if this point is close enough to the mapped network to plan
        from/to, else a human-readable reason why not.

        Field case (2026-09-04, CZU): the robot ran at CZU Suchdol with
        maps/stromovka.json loaded. Both the robot and the QR target
        snapped to the SAME far corner of the Stromovka graph, 2.5km away,
        so the route came out 0.0m long and the router immediately reported
        ARRIVED and held - from the operator's side, "it read the
        coordinates and then refused to move". Nothing downstream could
        detect that; snapping has no opinion about how far it reached."""
        snap = self.graph.snap(*latlon)
        if snap is None:
            return '%s: the map contains no routable ways at all' % label
        distance = snap[2]
        if distance <= self.max_snap_dist_m:
            return None
        return ('%s %.6f,%.6f is %.0fm from the nearest mapped way (limit %.0fm)%s'
                % (label, latlon[0], latlon[1], distance, self.max_snap_dist_m,
                   self._bbox_note(latlon)))

    def _fail(self, reason):
        print(self.time, 'ROUTE: CANNOT PLAN - %s' % reason)
        self.route = None
        self.fail_reason = reason
        self.state = RouteState.FAILED
        self._pending_plan_reason = None

    def _try_plan(self):
        if self.goal_ll is None:
            return
        if self.last_fix is None:
            self.state = RouteState.NO_FIX
            return

        # Refuse before planning rather than warn after. The old code only
        # printed a warning about a far-away target - to stdout, which the
        # log does not capture - and then planned anyway.
        problem = (self._off_map_reason('own position', self.last_fix)
                   or self._off_map_reason('target', self.goal_ll))
        if problem:
            self._fail(problem)
            return

        reason = self._pending_plan_reason or 'replan'
        self.state = RouteState.PLANNING
        route = self.graph.plan(self.last_fix, self.goal_ll,
                                 start_heading=self._compass_bearing(),
                                 uturn_penalty_m=self.uturn_penalty_m,
                                 blocked_edges=self.blocked_edges,
                                 snap_heading_m_per_deg=self.snap_heading_m_per_deg,
                                 snap_radius_m=self.snap_heading_radius_m)
        self._pending_plan_reason = None
        self._last_replan_time = self.time
        if route is None:
            self._fail('no route to %.6f,%.6f over the allowed way network '
                        '(check the map area and the highway filter)' % self.goal_ll)
            return
        # Independent cross-check on the graph itself: a route that says we
        # are already there while the globe says the target is far away is
        # incoherent, whatever produced it. This is what would have caught
        # the CZU case even without the snap-distance limit above.
        direct = haversine_distance(*self.last_fix, *self.goal_ll)
        if route.total <= self.arrival_dist_m < direct:
            self._fail('planned route is %.1fm long but the target is %.0fm away in a straight '
                        'line - the graph cannot connect them sensibly%s'
                        % (route.total, direct, self._bbox_note(self.goal_ll)))
            return
        if route.total <= 0.0:
            print(self.time, 'ROUTE: already at the target')
        self.fail_reason = None
        self.route = route
        self.plan_seq += 1
        self.s = 0.0
        self.cross_track = 0.0
        self._corridor_bad_since = None
        self._recovering_since = None
        self._stall_anchor_s = 0.0
        self._stall_anchor_time = self.time
        self.state = RouteState.FOLLOWING
        junctions = int(route.junction.sum())
        print(self.time, 'ROUTE: planned %s - %.0fm, %d points, %d junctions '
                          '(start %.1fm off-path, target %.1fm off-path)'
               % (reason, route.total, len(route), junctions,
                  route.meta['start_snap_dist_m'], route.meta['goal_snap_dist_m']))
        if route.meta['goal_snap_dist_m'] > self.target_snap_warn_m:
            print(self.time, 'ROUTE: WARNING - the QR target is %.0fm from the nearest mapped way. '
                              'The route ends at the mapped point, not at the coordinate itself.'
                   % route.meta['goal_snap_dist_m'])
        self.publish('route_plan', {
            'reason': reason,
            'goal': list(self.goal_ll),
            'total_m': round(route.total, 1),
            'junctions': junctions,
            # logged, not just printed - how far each end had to reach to
            # find the network is the first thing to check when a route
            # looks wrong
            'start_snap_m': round(route.meta['start_snap_dist_m'], 1),
            'goal_snap_m': round(route.meta['goal_snap_dist_m'], 1),
            'points': [[round(la, 7), round(lo, 7)] for la, lo in route.points_ll],
        })

    def _replan_position_ok(self):
        """Whether the current fix is close enough to the network for a
        re-plan to start somewhere real - see replan_max_snap_m."""
        if self.replan_max_snap_m <= 0 or self.route is None or self.last_fix is None:
            return True
        snap = self.graph.snap(*self.last_fix)
        if snap is None or snap[2] <= self.replan_max_snap_m:
            return True
        if (self._snap_note_time is None
                or (self.time - self._snap_note_time).total_seconds() > 15.0):
            self._snap_note_time = self.time
            print(self.time, 'ROUTE: not re-planning - the fix is %.0fm from any mapped way, a new route would '
                              'start somewhere the robot has never been' % snap[2])
        return False

    def _request_replan(self, reason):
        if self.replans >= self.max_replans:
            return False
        if self._last_replan_time is not None and self.time - self._last_replan_time < self.replan_cooldown:
            return False
        if not self._replan_position_ok():
            return False
        self.replans += 1
        print(self.time, 'ROUTE: re-planning (%s), attempt %d/%d' % (reason, self.replans, self.max_replans))
        self._pending_plan_reason = reason
        self._try_plan()
        return True

    # --- tracking -------------------------------------------------
    def _gps_update(self, lat, lon):
        """Pull `s` toward where this fix projects onto the route, and run
        the corridor/lost monitors. Runs at the GPS rate (1Hz); the aim
        point itself is republished at the pose2d rate in between."""
        xy = self.graph.to_xy(lat, lon)
        projection = self.route.project(xy,
                                         self.s - self.gps_search_back_m,
                                         self.s + self.gps_search_fwd_m)
        if projection is None:
            return
        s_gps, cross, dist = projection
        self.s = max(0.0, min(self.route.total, self.s + self.gps_snap_alpha * (s_gps - self.s)))
        self.cross_track += self.cross_track_alpha * (cross - self.cross_track)

        if self.state == RouteState.ARRIVED:
            return

        # "am I near ANY mapped way" - deliberately a separate question
        # from "am I near the PLANNED one". The first tells you whether
        # the robot is on grass; the second whether it is on the right
        # path. Only the combination distinguishes "avoidance put us on a
        # neighbouring path" (re-plan, nothing wrong) from "we are out on
        # the lawn" (recover).
        snap = self.graph.snap(lat, lon)
        dist_any_way = snap[2] if snap else float('inf')

        # --- how far PAST THE EDGE of the nearest road the robot is ---
        # Distance to a centreline is not the same question. A 6m service
        # road tolerates 2.5m of it; a 2.4m Stromovka footway does not,
        # and 54% of that extract is footway. Subtracting the way's own
        # half-width (see DEFAULT_HIGHWAY_HALFWIDTH) turns one number into
        # the number that actually decides whether the robot has left the
        # road - which, at Robotour, ends the run.
        #
        # Deliberately measured against the NEAREST way rather than the
        # planned one: drifting onto a legal parallel path is not going
        # off-road, and is already handled a few lines below by re-
        # planning. cross_track stays what it was - lane discipline on the
        # plan - and is what the follower's corridor bias keeps using.
        if snap is not None and len(self.graph.seg_halfwidth):
            half = float(self.graph.seg_halfwidth[snap[0]])
            self.off_road_m = max(0.0, dist_any_way - half)
            self.road_halfwidth_m = half
        else:
            self.off_road_m = 0.0 if snap is None else dist_any_way
            self.road_halfwidth_m = None

        if dist_any_way > self.lost_dist_m:
            if self._lost_since is None:
                self._lost_since = self.time
            elif self.time - self._lost_since > self.lost_confirm and self.state != RouteState.LOST:
                print(self.time, 'ROUTE: LOST - %.0fm from the nearest mapped way for %.0fs, holding'
                       % (dist_any_way, (self.time - self._lost_since).total_seconds()))
                self.state = RouteState.LOST
        else:
            self._lost_since = None
            if self.state == RouteState.LOST:
                print(self.time, 'ROUTE: back within %.0fm of a mapped way, resuming' % dist_any_way)
                self.state = RouteState.FOLLOWING

        if self.state == RouteState.LOST:
            return  # holding for a human; nothing below can improve on that

        # NOT an early return while the lost check is merely COUNTING DOWN:
        # confirming LOST takes lost_confirm_sec (10s), the corridor monitor
        # confirms in corridor_hard_confirm_sec (6s), and a robot 25m from
        # any mapped path is emphatically off its corridor too. Returning
        # early there would leave it cruising at cruise_authority for the
        # four seconds between the two - i.e. at its least corrected
        # exactly when it is most obviously lost. Recovery starts first,
        # and LOST escalates on top of it if recovery does not take.
        if dist > self.corridor_hard_m:
            if self._corridor_bad_since is None:
                self._corridor_bad_since = self.time
            elif self.time - self._corridor_bad_since > self.corridor_confirm:
                if dist_any_way < self.corridor_soft_m:
                    # on a different real path - the avoidance maneuvers
                    # moved us, the map is still fine, just re-route
                    if (self._drift_replan_consistent(snap)
                            and self._request_replan('drifted onto another mapped path')):
                        return
                if self.state != RouteState.RECOVERING:
                    print(self.time, 'ROUTE: off the planned corridor (%.1fm, nearest mapped way '
                                      '%.1fm) - steering back' % (dist, dist_any_way))
                    self.state = RouteState.RECOVERING
                    self._recovering_since = self.time
        elif dist < self.corridor_soft_m:
            self._corridor_bad_since = None
            if self.state == RouteState.RECOVERING:
                print(self.time, 'ROUTE: back on the planned corridor (%.1fm)' % dist)
                self.state = RouteState.FOLLOWING
                self._recovering_since = None

        if (self.state == RouteState.RECOVERING and self._recovering_since is not None
                and self.time - self._recovering_since > self.recovery_replan_after):
            if not self._recovery_replan_consistent():
                # heading follows the planned road: a GPS offset, not an excursion
                self._recovering_since = self.time
            elif not self._request_replan('recovery took too long'):
                self._recovering_since = self.time  # cooldown/limit - wait and keep steering back

        self._check_stall()

    def _check_stall(self):
        if self._stall_anchor_time is None:
            self._stall_anchor_s, self._stall_anchor_time = self.s, self.time
            return
        if abs(self.s - self._stall_anchor_s) >= self.stall_dist_m:
            self._stall_anchor_s, self._stall_anchor_time = self.s, self.time
            return
        if self.time - self._stall_anchor_time < self.stall_sec:
            return
        # No progress along the route for a long time. This CANNOT tell a
        # blocked path from an operator holding the robot, so it is slow
        # and bounded by max_replans on purpose. Penalise the edge the
        # robot is sitting on so the re-plan actually produces something
        # different rather than the identical route.
        snap = self.graph.snap(*self.last_fix) if self.last_fix else None
        if snap is not None:
            self.blocked_edges.add(frozenset(self.graph.seg_nodes[snap[0]]))
        if self._request_replan('no progress along the route for %.0fs'
                                 % (self.time - self._stall_anchor_time).total_seconds()):
            self._stall_anchor_s, self._stall_anchor_time = self.s, self.time

    # --- output ---------------------------------------------------
    def _heading_misalignment(self):
        """How far the robot is pointing off the route's own direction here,
        in radians, or None if the heading is unknown.

        This is a TURN that has to be executed, and nothing else in
        _guidance sees it: junction proximity is about distance to the next
        mapped fork, so a robot facing 90 or 180 deg away from its route on
        a perfectly straight stretch is treated as ordinary cruising - 20
        deg of steering ceiling, 0.45 authority - and creeps round in a
        long arc while the road mask, which has no idea a route exists,
        pulls it straight.

        Measured over the 2026-09-05 runs: one leg spent 75 seconds at
        119-175 deg off the route axis, and the first two minutes of the
        worst run turned 1645 deg over 26 m of travel (63 deg/m against a
        7-15 deg/m norm) - the reported circling at the start, and the
        "wrong way for several seconds then a sharp turn" on the wide road
        by the dormitory. Both are the same thing: a large heading error
        that no mode reacted to."""
        bearing = self._compass_bearing()
        if bearing is None or self.route is None:
            return None
        tangent = self.route.tangent_at(self.s)
        return abs(normalize_angle(bearing - math.atan2(tangent[0], tangent[1])))

    def _junction_approach(self):
        """0..1 - how much this cycle is "at a fork" rather than "cruising a
        path". Ramps up over junction_zone_m before a planned junction and
        stays at 1 for junction_exit_m after it, because completing the turn
        out of a fork needs the same commitment as entering it."""
        # Only forks the route actually TURNS at count - see
        # next_junction_dist. A place where paths merely meet is not an
        # event for a robot going straight through it.
        mt = self.junction_min_turn
        to_junction = self.route.next_junction_dist(self.s, mt)
        # A large heading error IS a turn, wherever it happens - see
        # _heading_misalignment. Treated as fully "at a fork" so it gets the
        # steering ceiling and authority a turn needs instead of cruising
        # round in an arc.
        misalign = self._heading_misalignment()
        # Hysteresis, because this gate flips a lot more than the geometry
        # it is meant to describe. It keys off the live compass heading,
        # which swings through tens of degrees during an avoidance
        # maneuver, and every flip changes authority (0.45 <-> 0.85), the
        # steering ceiling (20 <-> 40 deg) and - through road_axis_err -
        # the follower's mask_trust (1.00 <-> 0.25). Field case,
        # 2026-09-06 run 115237 t=405-425: junction and corridor alternated
        # every few seconds through a 20-second avoidance thrash, so the
        # follower's whole control law changed underneath it repeatedly
        # while it was trying to resolve a tight spot.
        #
        # Once latched it takes misaligned_turn_release_frac of the entry
        # angle to let go, so a heading wobbling around the threshold
        # stays in one mode instead of switching on every sample.
        if misalign is not None:
            release = self.misaligned_turn_angle * self.misaligned_turn_release_frac
            if misalign > self.misaligned_turn_angle:
                self._misaligned_latched = True
            elif misalign < release:
                self._misaligned_latched = False
            if self._misaligned_latched:
                return 1.0, to_junction
        if self.route.dist_since_junction(self.s, mt) < self.junction_exit_m:
            return 1.0, to_junction
        if to_junction < self.junction_zone_m and self.junction_zone_m > 0:
            return 1.0 - to_junction / self.junction_zone_m, to_junction
        return 0.0, to_junction

    def _aim_offset(self, lookahead):
        """Angle between the straight line to the aim point and the road's
        own direction here, in radians. This is the number that decides
        whether the route is asking Matty to drive along the path or across
        it."""
        here = self.route.xy_at(self.s)
        aim = self.route.xy_at(min(self.route.total, self.s + lookahead))
        dx, dy = aim[0] - here[0], aim[1] - here[1]
        if math.hypot(dx, dy) < 0.5:
            return 0.0
        tangent = self.route.tangent_at(self.s)
        return abs(normalize_angle(math.atan2(dx, dy) - math.atan2(tangent[0], tangent[1])))

    def _limit_aim_offset(self, lookahead, max_offset):
        """Shorten the lookahead until the aim point stops pointing across
        the road instead of along it.

        Pure pursuit aims at a point some distance ahead ON the path, but
        the STRAIGHT LINE to it can leave the path entirely where the path
        bends back - an S-bend, a switchback, a path hugging a building.
        Measured over real planned routes at Stromovka and Suchdol, away
        from any junction: median offset 0 deg and p90 6-8 deg, so almost
        always harmless - but p99 reaches 19-70 deg and the worst cases
        164 deg, i.e. the route briefly asking for a turn straight off the
        road. That is exactly the input that makes the road mask decide the
        grass is drivable, so the rare case matters more than the median.

        Shortening rather than clamping the bearing is deliberate: it keeps
        the aim point ON the mapped path (a clamped bearing points at open
        ground that no longer corresponds to anything), and it converges by
        construction, since the chord tends to the tangent as the lookahead
        goes to zero."""
        if max_offset <= 0 or self._aim_offset(lookahead) <= max_offset:
            return lookahead
        low, high = self.lookahead_min_m, lookahead
        for _ in range(6):                      # ~0.1m resolution over a 8m span
            mid = 0.5 * (low + high)
            if self._aim_offset(mid) <= max_offset:
                low = mid
            else:
                high = mid
        return low

    def _gps_age_sec(self):
        if self.last_fix_time is None or self.time is None:
            return None
        return (self.time - self.last_fix_time).total_seconds()

    def _gps_degradation(self):
        """0 while fixes are current, ramping to 1 as they go stale - see
        gps_dead_reckon_sec in __init__."""
        age = self._gps_age_sec()
        if age is None or age <= self.gps_dead_reckon_sec:
            return 0.0
        span = max(1e-6, self.gps_lost_sec - self.gps_dead_reckon_sec)
        return max(0.0, min(1.0, (age - self.gps_dead_reckon_sec) / span))

    def _turn_rel_deg(self):
        rel = self.route.next_turn_relative(self.s, self.junction_min_turn, self.turn_rel_probe_m,
                                            self.turn_rel_exit_probe_m, self.turn_rel_cluster_m)
        return None if rel is None else round(math.degrees(rel), 1)

    def _turn_info(self, to_junction):
        """The next-turn keys _guidance publishes, for states that do not
        run the rest of it - see recovery_turn_info."""
        turn_signed = self.route.next_turn_signed(self.s, self.junction_min_turn)
        exit_bearing = self.route.next_turn_exit_bearing(self.s, self.junction_min_turn,
                                                         self.turn_exit_probe_m)
        return {
            'turn_dir_deg': None if turn_signed is None else round(math.degrees(turn_signed), 1),
            'turn_dist_m': round(to_junction, 1),
            'exit_bearing_deg': None if exit_bearing is None else round(math.degrees(exit_bearing) % 360.0, 1),
            'turn_rel_deg': self._turn_rel_deg(),
        }

    def _recovery_replan_consistent(self):
        """Whether a "recovery took too long" re-plan is warranted. Not while
        the heading runs along the planned road here (within
        replan_heading_check_deg): the robot is on it and the fix is off. A
        re-plan starts from the nearest way to that fix - 165145 t=45 and all
        five re-plans in 170349 started 5-8 m off-path. No heading = previous
        behaviour (re-plan)."""
        if self.replan_heading_check_deg <= 0:
            return True
        heading = self._compass_bearing()
        if heading is None:
            return True
        tx, ty = self.route.tangent_at(self.s)
        off = abs((math.atan2(tx, ty) - heading + math.pi / 2) % math.pi - math.pi / 2)
        if off > math.radians(self.replan_heading_check_deg):
            return True
        if self._drift_note_time is None or (self.time - self._drift_note_time).total_seconds() > 10.0:
            self._drift_note_time = self.time
            print(self.time, 'ROUTE: off the corridor by GPS but heading follows the planned road '
                              '(%.0f deg) - not re-planning' % math.degrees(off))
        return False

    def _drift_replan_consistent(self, snap):
        """Whether a "drifted onto another mapped path" re-plan is backed by
        the heading: the nearest way runs along it and the planned route
        here does not. One fix near another way is not enough - under trees
        the fix wanders 5-10 m and the planned way is usually still right
        (165145). See replan_heading_check_deg."""
        if self.replan_heading_check_deg <= 0 or snap is None:
            return True
        heading = self._compass_bearing()
        limit = math.radians(self.replan_heading_check_deg)
        off = lambda bearing: abs((bearing - heading + math.pi / 2) % math.pi - math.pi / 2)  # noqa: E731
        if heading is None:
            ok, why = False, 'no trustworthy heading yet'
        else:
            d = self.graph._seg_d[snap[0]]
            tx, ty = self.route.tangent_at(self.s)
            way_off, route_off = off(math.atan2(d[0], d[1])), off(math.atan2(tx, ty))
            ok = way_off <= limit and route_off > limit
            why = 'nearest way %.0f deg off the heading, planned route %.0f deg' % (
                math.degrees(way_off), math.degrees(route_off))
        if not ok and (self._drift_note_time is None
                       or (self.time - self._drift_note_time).total_seconds() > 10.0):
            self._drift_note_time = self.time
            print(self.time, 'ROUTE: near another mapped way but not re-planning onto it (%s)' % why)
        return ok

    def _guidance(self):
        """Everything the follower needs to know about HOW to use this aim
        point, not just where it is. Three situations that want genuinely
        different control, so they are named rather than interpolated into
        one authority number:

        CORRIDOR - cruising a path. The route is a slow BIAS on top of the
            mask, not a competitor to it. Measured over the 2026-08-29
            Stromovka logs: while 1-8m off the planned corridor the road
            mask steered back toward it only 40-48% of the time, mean
            contribution -0.7 to -1.0 deg, i.e. very slightly AWAY. The mask
            is not broken - it is a lane-KEEPING sensor with no notion of
            which lane - but it means the only restoring signal available
            comes from the map, and it must not be something the mask can
            outvote. Hence a small additive bias rather than a blend.

        JUNCTION - a planned fork. The mask has no opinion about which
            branch leads to the target, and the turn has to be decisive: a
            T needs up to 90 deg within the width of the junction. Here the
            route does take over, with a much larger steering limit, a
            relaxed rate limit and a lower speed so the turn fits.

        RECOVERY - confirmed off the planned corridor. Route dominant,
            short lookahead so the return angle is steep, reduced speed.

        Returns a dict merged into the published hint."""
        if self.state == RouteState.RECOVERING:
            hint = {}
            if self.recovery_turn_info:
                hint = self._turn_info(self.route.next_junction_dist(self.s, self.junction_min_turn))
            hint.update({
                'mode': 'recovery',
                'authority': self.recovery_authority,
                'lookahead_m': self.recovery_lookahead_m,
                'steer_limit_deg': self.recovery_steer_limit_deg,
                'steer_rate_deg_s': self.junction_steer_rate_deg_s,
                'speed_limit': self.recovery_speed_limit,
            })
            return hint

        approach, to_junction = self._junction_approach()
        lerp = lambda a, b: a + approach * (b - a)  # noqa: E731 - three uses, all identical

        # The aim point must not reach past the fork by more than
        # junction_overshoot_m: a long lookahead across a corner is exactly
        # what makes a robot cut it, and the inside of a corner in a park
        # is grass.
        # Note what happens at to_junction ~= lookahead_m: the aim point
        # lands ON the junction node, which is the least informative place
        # it can be. The bearing to a corner carries nothing about which
        # way the route leaves it - only the robot's own lateral offset -
        # so the follower steers to the centreline for several seconds and
        # then has to take the whole turn inside the fork. Measured over
        # the 2026-09-05 runs, the aim offset sat pinned at 26-32 deg for
        # the nine seconds before a 121 deg turn while the commanded
        # steering stayed within +-5 deg, and only swung once the node had
        # been passed: the reported "went the wrong way for several
        # seconds, then made a sharp turn when it was almost too late".
        # junction_overshoot_m is the lever - it is how far PAST the
        # corner the aim point is allowed to reach, and therefore how
        # early the turn becomes visible at all. _limit_aim_offset below
        # is what stops that reach turning into corner-cutting.
        lookahead = max(self.lookahead_min_m,
                        min(self.lookahead_m, to_junction + self.junction_overshoot_m))
        # Off the corridor, aim CLOSER. The steepest return a pure-pursuit
        # aim point can ever ask for is atan(cross_track / lookahead), so
        # at the cruising 8m lookahead a 2.5m excursion is bounded at
        # 17deg however much authority the route is given - and measured
        # over the 2026-09-05 runs the robot was already within 4.1deg of
        # that aim while more than 1.5m off. It was not ignoring the
        # route; it was converging as fast as the geometry permitted,
        # which is slowly, which is the reported "drove parallel to the
        # road on the grass for a very long time".
        #
        # RECOVERING already shortens to recovery_lookahead_m for exactly
        # this reason. This reaches for the same number continuously,
        # ramping over the same corridor_soft_m..corridor_hard_m span the
        # corridor monitor already uses, instead of only after
        # corridor_hard_confirm_sec of confirmed excursion. At 2.5m off
        # with the shipped span the aim comes in to 6.4m (21deg); at 4m
        # it reaches recovery_lookahead_m (45deg).
        span = self.corridor_hard_m - self.corridor_soft_m
        if span > 0 and self.lookahead_min_m < lookahead:
            excursion = min(1.0, max(0.0, (abs(self.cross_track) - self.corridor_soft_m) / span))
            # ...driven by whichever of the two excursion measures is more
            # urgent. cross_track is lane discipline on the PLAN and can
            # be large while the robot is still legitimately on a wide
            # road; off_road_m says it is on grass, which at Robotour ends
            # the run. Neither subsumes the other, so take the max rather
            # than choosing.
            if self.off_road_m is not None and self.offroad_lookahead_m > 0:
                excursion = max(excursion, min(1.0, self.off_road_m / self.offroad_lookahead_m))
            # Aims all the way in to lookahead_min_m rather than stopping
            # at recovery_lookahead_m: the return angle is atan(offset /
            # lookahead), so the lookahead IS the return angle, and the
            # distance driven off the road is offset/sin(that angle) -
            # independent of speed. Shortening the aim is therefore the
            # only lever that actually reduces METRES of road-leaving;
            # slowing down reduces how deep the excursion gets, not how
            # far it runs.
            target = min(self.recovery_lookahead_m, self.lookahead_min_m)                 if self.off_road_m else self.recovery_lookahead_m
            lookahead += excursion * (max(self.lookahead_min_m, target) - lookahead)
            lookahead = max(self.lookahead_min_m, lookahead)
        # ...and it must not point across an S-bend either - see
        # _limit_aim_offset. A turn IS expected at a fork, so the bound
        # opens up as `approach` rises.
        # ...and while the robot is already OFF the road, the junction
        # allowance does not apply. junction_max_aim_offset exists so a
        # fork can ask for a real turn; out on the grass it instead lets
        # the aim point point across whatever the robot has to cross to
        # get back, which is how a junction turn becomes an excursion.
        # Measured over the 2026-09-06 runs, four of the eleven sustained
        # off-road stretches were in junction mode with the commanded
        # steering pointing AWAY from the route (-0.9 to -3.8 deg), even
        # though the follower's off-road takeover was already active - it
        # only applies in the follower's corridor branch, and a junction
        # aim it cannot argue with is upstream of that.
        max_offset = lerp(self.corridor_max_aim_offset, self.junction_max_aim_offset)
        if self.off_road_m is not None and self.off_road_m > 0:
            max_offset = min(max_offset, self.corridor_max_aim_offset)
        lookahead = self._limit_aim_offset(lookahead, max_offset)
        degraded = self._gps_degradation()
        # --- the next real turn, as GEOMETRY rather than as an aim point ---
        # Everything derived from where the robot is thought to BE is
        # contaminated. Measured over the 2026-09-06 test4 runs, the GPS
        # fix sits at a near-constant offset from the map for the whole of
        # a run - 3.17m in run 154422 with a scatter of 0.68m and a
        # direction that drifts 0 deg between the run's two halves, 2.46m
        # in 160437 drifting 1 deg. That is a bias, not noise: no EMA, no
        # gain choice and no amount of slow-loop design removes a constant.
        # With it, cross_track says the robot is metres off a road it is
        # driving down the middle of, and the correction pushes it off the
        # other side - which is what put it into the wall by the
        # dormitories.
        #
        # What survives the bias is the shape of the plan: WHICH WAY the
        # route turns next and ROUGHLY how far ahead. The turn angle comes
        # from the polyline, so it does not depend on the fix at all, and
        # a 3m along-track error inside a 12m approach zone is tolerable.
        # Note the aim-point BEARING does not survive: at an 8m lookahead
        # a 3m lateral bias is 21 deg of false bearing, which is most of a
        # junction turn.
        turn_signed = self.route.next_turn_signed(self.s, self.junction_min_turn)
        exit_bearing = self.route.next_turn_exit_bearing(self.s, self.junction_min_turn,
                                                         self.turn_exit_probe_m)
        hint = {
            'turn_dir_deg': None if turn_signed is None else round(math.degrees(turn_signed), 1),
            'turn_dist_m': round(to_junction, 1),
            'exit_bearing_deg': None if exit_bearing is None
                                else round(math.degrees(exit_bearing) % 360.0, 1),
        }

        hint['turn_rel_deg'] = self._turn_rel_deg()

        # Speed at a turn scaled by how sharp it is, not a flat cap. With
        # corners counted as turns (which they must be - see Route), dense
        # campus paths put the robot in junction mode about half the time,
        # and a flat 0.3 m/s there was measured holding 48% of all forward
        # cycles at that speed. A 35deg bend does not need what a 90deg
        # fork needs.
        speed_limit = None
        if approach > 0.5:
            turn = self.route.next_turn_angle(self.s, self.junction_min_turn) or math.pi / 2
            misalign = self._heading_misalignment()
            if misalign is not None and misalign > self.misaligned_turn_angle:
                turn = max(turn, misalign)      # turning the robot round IS the sharp turn
            sharp = max(0.0, min(1.0, (turn - self.gentle_turn_angle)
                                  / max(1e-6, self.sharp_turn_angle - self.gentle_turn_angle)))
            speed_limit = self.gentle_turn_speed + sharp * (self.junction_speed_limit
                                                            - self.gentle_turn_speed)
        if degraded > 0:
            speed_limit = min(speed_limit or 1e9, self.gps_degraded_speed_limit)
        hint.update({
            'mode': ('junction' if approach > 0.5 else 'corridor')
                     if degraded < 1 else 'corridor',
            'approach': round(approach, 3),
            'gps_degraded': round(degraded, 2),
            # authority fades with the fix going stale: the route still
            # knows the shape of the path, it is losing track of where the
            # robot is on it, and the mask does not depend on GPS at all
            'authority': (1.0 - degraded) * lerp(self.cruise_authority, self.junction_authority),
            'lookahead_m': lookahead,
            # Steering headroom. turn_angle (20 deg) is the follower's
            # CRUISING correction ceiling and is right for that; a fork is
            # not a correction. At 20 deg Matty's turning radius is 0.91m
            # (matty.py: 0.16/tan(steer/2)) so a 90 deg turn needs 1.4m of
            # arc and swings wide; at 40 deg it is 0.44m and 0.7m of arc,
            # which fits inside a park path.
            'steer_limit_deg': lerp(self.cruise_steer_limit_deg, self.junction_steer_limit_deg),
            # ...and the rate limit has to allow actually reaching it. At
            # the follower's default 30 deg/s, winding on 40 deg takes 1.3s
            # = 0.65m at cruising speed, most of the junction.
            'steer_rate_deg_s': lerp(self.cruise_steer_rate_deg_s, self.junction_steer_rate_deg_s),
            # Slowing into a fork buys time for the turn and shortens every
            # stopping distance downstream. None = no cap.
            'speed_limit': speed_limit,
        })
        return hint

    def _publish_hint(self):
        hint = {
            'state': self.state,
            'lat': None,
            'lon': None,
            'remaining_m': None,
            'authority': 0.0,
            'cross_track_m': round(self.cross_track, 2),
            'off_route': self.state in (RouteState.RECOVERING, RouteState.LOST),
            'arrived': self.state == RouteState.ARRIVED,
            'hold': None,
            'reason': self.fail_reason if self.state == RouteState.FAILED else None,
        }
        if self.state == RouteState.FREE:
            # "start"/"go": road following and obstacle avoidance only.
            # Everything route-derived is withheld rather than left stale -
            # a cross-track from a route that is no longer being followed
            # would otherwise keep biasing the steering.
            hint['cross_track_m'] = None
            hint['mode'] = 'free'
        elif self.state in (RouteState.WAITING, RouteState.NO_FIX, RouteState.PLANNING):
            # nothing to follow yet - the follower crawls or stands still,
            # its choice (route_hold_creep_speed in tulak_obstacle)
            hint['hold'] = 'creep'
        elif self.state in (RouteState.ARRIVED, RouteState.LOST, RouteState.FAILED):
            hint['hold'] = 'stop'

        # The off-road distance is published unconditionally, in every
        # state that has a fix - unlike cross_track, which is withheld
        # without a route. Leaving the road is a terminal event whether or
        # not there is currently a plan to follow.
        hint['off_road_m'] = round(self.off_road_m, 2) if self.off_road_m is not None else None
        # How wide the surface under the robot is - published so the
        # follower can express cross-track as a fraction of the way to the
        # road EDGE rather than in bare metres. See _route_corridor_bias.
        hint['road_halfwidth_m'] = self.road_halfwidth_m

        age = self._gps_age_sec()
        hint['gps_age_sec'] = None if age is None else round(age, 1)
        if (self.state in (RouteState.FOLLOWING, RouteState.RECOVERING)
                and age is not None and age > self.gps_lost_sec):
            print(self.time, 'ROUTE: GPS LOST - no fix for %.0fs. Route guidance off; road '
                              'following and obstacle avoidance continue. Target kept (%.6f,%.6f) '
                              '- a returning fix will re-plan.' % (age, self.goal_ll[0], self.goal_ll[1]))
            self.state = RouteState.GPS_LOST
            hint['state'] = self.state
        if self.state == RouteState.GPS_LOST:
            # exactly what "start"/"go" publishes: nothing to follow, no
            # hold, no stale cross-track. The follower degrades to the road
            # mask plus depth on its own.
            hint['cross_track_m'] = None
            hint['mode'] = 'free'

        if self.route is not None and self.state in (RouteState.FOLLOWING, RouteState.RECOVERING):
            remaining = self.route.total - self.s
            if remaining <= self.arrival_dist_m:
                self.state = RouteState.ARRIVED
                print(self.time, 'ROUTE: ARRIVED - %.0fm route complete, %.1fm remaining, stopping'
                       % (self.route.total, remaining))
                hint['state'] = self.state
                hint['arrived'] = True
                hint['hold'] = 'stop'
            else:
                guidance = self._guidance()
                aim_lat, aim_lon = self.graph.to_ll(
                    *self.route.xy_at(self.s + guidance['lookahead_m']))
                # Which way the mapped path itself runs here, as a compass
                # bearing. The follower compares this against its own
                # heading to decide how much the road mask is worth right
                # now: measured on the 2026-08-29 logs, pointing 45-60 deg
                # off the road axis takes the mask from 27% to 51%
                # fragmented and quadruples the rate at which its centroid
                # lands between two blobs instead of on either. Published
                # rather than computed there because only this module knows
                # where the road runs.
                tangent = self.route.tangent_at(self.s)
                hint.update(guidance)
                hint.update({
                    'lat': aim_lat,
                    'lon': aim_lon,
                    'remaining_m': round(remaining, 1),
                    'road_bearing_deg': round(math.degrees(
                        math.atan2(tangent[0], tangent[1])) % 360.0, 1),
                    'junction_m': round(self.route.next_junction_dist(self.s, self.junction_min_turn), 1),
                    'progress_m': round(self.s, 1),
                    'plan_seq': self.plan_seq,
                })
                if age is not None and age > self.gps_dead_reckon_sec:
                    # cannot be measured without a fix, and a frozen value
                    # is a constant steering offset with no feedback
                    hint['cross_track_m'] = None
                    hint['off_road_m'] = None
                hint['authority'] = round(hint['authority'], 3)
                hint['lookahead_m'] = round(hint['lookahead_m'], 1)
        self.publish('route_hint', hint)
        self._maybe_log(hint)

    def _maybe_log(self, hint):
        if self._last_log_time is not None and self.time - self._last_log_time < self.log_interval:
            return
        self._last_log_time = self.time
        if self.route is None:
            print(self.time, 'ROUTE: %s%s' % (self.state,
                   '' if self.goal_ll is None else ' target=%.6f,%.6f' % self.goal_ll))
            return
        print(self.time, 'ROUTE: %-10s %-8s progress=%.0f/%.0fm cross=%+.1fm next_junction=%.0fm '
                          'authority=%.2f steer_limit=%.0fdeg aim=%s'
               % (self.state, hint.get('mode', '-'), self.s, self.route.total, self.cross_track,
                  self.route.next_junction_dist(self.s, self.junction_min_turn), hint['authority'],
                  hint.get('steer_limit_deg', 0),
                  'none' if hint['lat'] is None else '%.6f,%.6f' % (hint['lat'], hint['lon'])))


# vim: expandtab sw=4 ts=4
