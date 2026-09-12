"""
  Local OSM basemap for the run-mapping tools.

  Why this exists: map_events.py and map_runs.py used to pull raster tiles
  straight from OpenStreetMap's own tile servers. That traffic is not what
  those volunteer-run servers are for, and they now answer 403 "App is not
  following the tile usage policy" - the whole background of the map comes
  back as warning tiles (osm.wiki/Blocked). The answer is not to dress the
  requests up so they get through; it is to stop making them.

  We already have the map locally. osgar-apps/followme/maps/*.json are the
  OSM extracts osm_router.py routes on, so the ways can simply be drawn as
  vectors. That is offline, needs no service, and is arguably the better
  backdrop for this job: what you see is exactly the road network the
  router believes exists, which is the thing that decides whether Matty was
  on a mapped road or on grass.

  Only ways carrying a `highway` tag are drawn, split into two weights so
  the hierarchy stays readable - see CLASSES.
"""
import io
import json
import os


# Roads Matty could plausibly be on, by how they should be drawn. Anything
# with a highway tag that is not listed lands in 'path' (thin) rather than
# being dropped, so an unusual tag never makes a way silently invisible.
ROAD_TYPES = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified',
    'residential', 'living_street', 'service', 'road',
    'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
}

CLASSES = {
    'road': {'c': '#c9ced4', 'w': 5},
    'path': {'c': '#d9cdb4', 'w': 2},
}

DEFAULT_MAPS = ('maps/suchdol.json', 'maps/stromovka.json')


def bounds_of(points, margin_deg=0.0015):
    """(min_lat, min_lon, max_lat, max_lon) around (lat, lon) pairs, padded.
    None when there is nothing to bound. The default margin is roughly
    150 m, enough to show what the robot was driving past."""
    pts = [p for p in points if p and p[0] is not None and p[1] is not None]
    if not pts:
        return None
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    return (min(lats) - margin_deg, min(lons) - margin_deg,
            max(lats) + margin_deg, max(lons) + margin_deg)


def load_ways(map_files, bounds):
    """Ways from the given OSM extracts that have a node inside `bounds`,
    as [{'p': [[lat, lon], ...], 'k': 'road'|'path'}, ...] ready for JSON.

    Clipped to the bounds rather than loaded wholesale: the Suchdol extract
    alone is 11k ways, and embedding all of it would make the page huge for
    no benefit."""
    if bounds is None:
        return []
    min_lat, min_lon, max_lat, max_lon = bounds
    out = []
    for path in map_files:
        if not os.path.exists(path):
            continue
        with io.open(path, encoding='utf-8') as f:
            data = json.load(f)
        nodes = data.get('nodes', {})
        for way in data.get('ways', []):
            highway = way.get('tags', {}).get('highway')
            if not highway:
                continue
            pts = []
            inside = False
            for nid in way.get('nodes', []):
                n = nodes.get(nid) or nodes.get(str(nid))
                if not n:
                    continue
                lat, lon = float(n[0]), float(n[1])
                pts.append([round(lat, 6), round(lon, 6)])
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    inside = True
            if inside and len(pts) > 1:
                out.append({'p': pts, 'k': 'road' if highway in ROAD_TYPES else 'path'})
    return out


def ways_json(points, map_files=None, margin_deg=0.0015):
    """Convenience: everything above in one call, returning a JSON string."""
    ways = load_ways(list(map_files or DEFAULT_MAPS), bounds_of(points, margin_deg))
    return json.dumps(ways), len(ways)


# Drawn before the tracks so the run always sits on top of the map.
BASEMAP_JS = """
const WAY_STYLE = %s;
(WAYS || []).forEach(w => {
  const s = WAY_STYLE[w.k] || WAY_STYLE.path;
  L.polyline(w.p, {color:s.c, weight:s.w, opacity:1}).addTo(map);
});
if (TILE_URL) L.tileLayer(TILE_URL, {maxZoom:19}).addTo(map);
map.attributionControl.addAttribution('map data &copy; OpenStreetMap contributors (local extract)');
""" % json.dumps(CLASSES)
