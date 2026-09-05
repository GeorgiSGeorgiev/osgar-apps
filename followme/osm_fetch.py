#!/usr/bin/python
"""
  Download an OpenStreetMap road/path extract for a bounding box and save
  it as the offline map file osgar.osm_router:OSMRouter reads at startup.

  WHY A FILE AND NOT A LIVE QUERY: the robot has no usable internet at a
  competition, and even if it did, a route plan is not something to make
  contingent on a public API answering within a few seconds while the
  organizers wait at the start line. The whole map is a few hundred kB -
  fetch it at home, commit it, done. The router never touches the network.

  WHAT IS SAVED: essentially the raw Overpass answer, trimmed to the node
  coordinates and the way node-lists/tags. Deliberately NOT pre-filtered
  into "drivable / not drivable" - which ways Matty may use is a decision
  with real consequences (steps are unclimbable, a trunk road is lethal,
  a grass shortcut is against the point of the exercise) and it belongs in
  the router's config where it can be changed and re-checked without
  re-downloading anything. See the ALLOWED/EXCLUDED notes in osm_router.py.

  Usage
  -----
    # Stromovka (Robotour 2026), with a comfortable margin around the park
    python osm_fetch.py --bbox 50.0980 14.4080 50.1130 14.4360 \
           -o maps/stromovka.json

    # around a point, with a radius in meters
    python osm_fetch.py --around 50.1055 14.4225 1200 -o maps/stromovka.json

    # inspect what a saved file contains, without downloading anything
    python osm_fetch.py --info maps/stromovka.json

  Pick the bbox GENEROUSLY. A route is allowed to leave the area you
  expect to drive in - and a graph truncated at the bbox edge does not
  fail loudly, it just quietly stops offering the ways that continue past
  the cut, so the router either takes a detour or declares the target
  unreachable. Map data for a whole park is small; err on the large side.
"""
import argparse
import datetime
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from collections import Counter


OVERPASS_URLS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
]

# Sent because Overpass rejects requests without a real User-Agent (it
# answers 406, which looks like "no internet" if you do not know to expect
# it). Identifying the project is also simply what their usage policy asks.
USER_AGENT = 'osgar-matty-osm-router/1.0 (robotika.cz; Robotour route planning)'

QUERY_TEMPLATE = """
[out:json][timeout:{timeout}];
(
  way["highway"]({s},{w},{n},{e});
);
(._;>;);
out body;
"""


def fetch_bbox(south, west, north, east, timeout=90):
    """Ask Overpass for every highway=* way intersecting the box, plus the
    nodes they reference (the '(._;>;)' recurse-down step - without it the
    ways come back as bare node-id lists with no coordinates anywhere)."""
    query = QUERY_TEMPLATE.format(s=south, w=west, n=north, e=east, timeout=timeout)
    last_error = None
    for url in OVERPASS_URLS:
        print('querying %s ...' % url)
        try:
            request = urllib.request.Request(
                url,
                data=urllib.parse.urlencode({'data': query}).encode(),
                headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout + 60) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:  # noqa: BLE001 - any failure just means try the next mirror
            print('  failed: %s: %s' % (type(e).__name__, e))
            last_error = e
    raise RuntimeError('all Overpass mirrors failed, last error: %r' % (last_error,))


def to_map_file(overpass_json, bbox):
    """Overpass answer -> the compact structure osm_router.py loads.

    Node ids are kept as STRINGS because this goes through json (and, in
    the log, msgpack), and object keys are strings there regardless - so
    converting once here keeps the router from having to guess which it
    is holding."""
    nodes = {}
    ways = []
    for element in overpass_json.get('elements', []):
        if element['type'] == 'node':
            nodes[str(element['id'])] = [element['lat'], element['lon']]
        elif element['type'] == 'way' and 'highway' in element.get('tags', {}):
            ways.append({
                'id': element['id'],
                'nodes': [str(n) for n in element['nodes']],
                'tags': element['tags'],
            })
    # drop nodes no surviving way references - typically most of the
    # answer, since the recurse-down step pulls in everything
    used = {n for w in ways for n in w['nodes']}
    nodes = {k: v for k, v in nodes.items() if k in used}
    return {
        'format': 'osgar-osm-router-1',
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'source': 'overpass-api',
        'bbox': list(bbox),  # south, west, north, east
        'nodes': nodes,
        'ways': ways,
    }


def describe(map_data):
    ways = map_data['ways']
    print('generated : %s' % map_data.get('generated'))
    print('bbox      : %s' % (map_data.get('bbox'),))
    print('nodes     : %d' % len(map_data['nodes']))
    print('ways      : %d' % len(ways))
    print()
    print('highway=  : %s' % dict(Counter(w['tags'].get('highway') for w in ways).most_common()))
    print('surface=  : %s' % dict(Counter(w['tags'].get('surface') for w in ways).most_common(12)))
    barriers = Counter()
    for w in ways:
        if w['tags'].get('access'):
            barriers['access=' + w['tags']['access']] += 1
        if w['tags'].get('foot'):
            barriers['foot=' + w['tags']['foot']] += 1
    print('restrict  : %s' % dict(barriers.most_common()))


def around_to_bbox(lat, lon, radius_m):
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--bbox', nargs=4, type=float, metavar=('SOUTH', 'WEST', 'NORTH', 'EAST'),
                        help='area to download, in decimal degrees')
    group.add_argument('--around', nargs=3, type=float, metavar=('LAT', 'LON', 'RADIUS_M'),
                        help='area to download, as a centre point and a radius in meters')
    group.add_argument('--info', metavar='MAPFILE',
                        help='print a summary of an already-downloaded map file and exit')
    parser.add_argument('-o', '--output', default='maps/osm_map.json', help='where to write the map file')
    parser.add_argument('--timeout', type=int, default=90, help='Overpass server-side timeout, seconds')
    args = parser.parse_args()

    if args.info:
        with open(args.info, encoding='utf-8') as f:
            describe(json.load(f))
        return 0

    if args.around:
        bbox = around_to_bbox(*args.around)
    else:
        bbox = tuple(args.bbox)
    south, west, north, east = bbox
    if not (south < north and west < east):
        parser.error('bbox must be given as SOUTH WEST NORTH EAST with south<north and west<east')

    print('bbox: south=%.5f west=%.5f north=%.5f east=%.5f' % bbox)
    map_data = to_map_file(fetch_bbox(south, west, north, east, timeout=args.timeout), bbox)

    directory = os.path.dirname(os.path.abspath(args.output))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(map_data, f, ensure_ascii=False, separators=(',', ':'))
    print()
    print('wrote %s (%.1f kB)' % (args.output, os.path.getsize(args.output) / 1024.0))
    print()
    describe(map_data)
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: expandtab sw=4 ts=4
