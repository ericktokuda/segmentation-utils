#!/usr/bin/env python3
"""Convert via format to pixel-wise label
"""

import os
import argparse
import logging
from logging import debug
import numpy as np
import json
from shapely.geometry import Point, Polygon
import itertools
from PIL import Image

def parse_via_entry(entry, w, h):
    """Parse via line 

    Args:
    entry(dict): entry corresponding to objects in one file in via format
    w(int): image width
    h(int): image height

    Returns:
    np.ndarray: labels array in UINT8
    """

    labels_ids = {'background': 0, 'tag': 1, 'frame': 2, 'sign': 3}

    i = 0
    polys = []
    for annot in entry['regions'].values():
        shp = annot['shape_attributes']
        if shp['name'] != 'polygon': continue
        _class = annot['region_attributes']['class']
        x = shp['all_points_x'] # horiz
        y = shp['all_points_y'] # vert
        coords = [ (i, j) for i, j in zip(x, y)]
        polys.append((_class, Polygon(coords)))

    pixels = list(itertools.product(range(w), range(h)))

    mask = np.zeros((h, w), dtype=int)

    for coord in pixels:
        p = Point(coord)
        for _class, poly in polys:
            if _class == 'sign' and p.x == 47 and p.y == 159:
                print(poly.contains(p))
                print(poly.contains(Point(159, 47)))
            if poly.contains(p):
                label = labels_ids[_class]
                mask[coord[1], coord[0]] = label
                break # We assume one label for pixel.

    return np.uint8(mask)

def parse_via_file(viafile, imdir, outdir):
    """Parse via file in json format

    Args:
    viafile(str): path to via file in json format
    """

    fh = open(viafile)
    content = json.load(fh)

    for v in content.values():
        imgpath = os.path.join(imdir, v['filename'])
        outfilename = v['filename'].replace('.jpg', '.png')
        outpath = os.path.join(outdir, outfilename)
        w, h = Image.open(imgpath).size
        
        mask = parse_via_entry(v, w, h)
        Image.fromarray(mask).save(outpath)
        Image.fromarray(mask*60).save(os.path.join('/tmp', outfilename))
    fh.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('viafile', help='Path to via file in JSON format.')
    parser.add_argument('imdir', help='Path to images referenced in the via file')
    args = parser.parse_args()

    outdir = '/tmp/'

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)
    parse_via_file(args.viafile, args.imdir, outdir)

if __name__ == "__main__":
    main()

