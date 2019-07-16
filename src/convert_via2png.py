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

    polys = []
    for annot in entry['regions'].values():
        shp = annot['shape_attributes']
        if shp['name'] != 'polygon': continue
        _class = annot['region_attributes']['class']
        x = shp['all_points_x']
        y = shp['all_points_y']
        coords = [ (i, j) for i, j in zip(x, y)]

        polys.append((_class, Polygon(coords)))

    coords = list(itertools.product(range(w), range(h)))

    mask = np.zeros((w, h), dtype=int)

    for coord in coords:
        p = Point(coord)
        for _class, poly in polys:
            if poly.contains(p):
                label = labels_ids[_class]
                mask[coord[1], coord[0]] = label
                break

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
        outfilename = os.path.join(outdir, imgpath.replace('.jpg', '.png'))
        w, h = Image.open(imgpath).size
        
        mask = parse_via_entry(v, w, h)
        Image.fromarray(mask).save(outfilename)
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

