#!/usr/bin/env python3
"""Convert via format to pixel-wise label
"""

import os
from os.path import join as pjoin
import argparse
import logging
from logging import debug
import numpy as np
import json
from shapely.geometry import Point, Polygon
import itertools
from PIL import Image
from multiprocessing import Pool
import inspect
from datetime import datetime
import sys
import imageio
import pandas as pd

#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

#############################################################
def parse_via_entry(_arg):
    """Parse via line 

    Args:
    entry(dict): entry corresponding to objects in one file in via format
    w(int): image width
    h(int): image height

    Returns:
    np.ndarray: labels array in UINT8
    """

    imdir, outdir, entry = _arg

    labels_ids = {'background': 0, 'tag': 1, 'frame': 2, 'sign': 3}

    imgpath = os.path.join(imdir, entry['filename'])
    outfilename = entry['filename'].replace('.jpg', '.png')
    #debug('{}:{}'.format(idx, outfilename))
    outpath = os.path.join(outdir, outfilename)

    if os.path.exists(outpath): return

    w, h = Image.open(imgpath).size

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
            if poly.contains(p):
                label = labels_ids[_class]
                mask[coord[1], coord[0]] = label
                break # We assume one label for pixel.

    mask = np.uint8(mask)
    Image.fromarray(mask).save(outpath)
    #return np.uint8(mask)

#############################################################
def parse_via_file_parallel(viafile, imdir, outdir, N):
    """Parse via file in json format

    Args:
    viafile(str): path to via file in json format
    """

    fh = open(viafile)
    content = json.load(fh)
    files = list(content.values())
    args = [ (imdir, outdir, x) for x in files ]

    with Pool(N) as p:
        print(p.map(parse_via_entry, args))

    fh.close()

#############################################################
def calculate_miou(gtruth, pred):
    inter = np.logical_and(gtruth, pred)
    union = np.logical_or(gtruth, pred)
    return np.sum(inter) / np.sum(union)

#############################################################
def calculate_miou_batch(preddir, gtruthdir, outpath):
    """Calculate miou for each file and save in out

    Args:
    preddir(str): pred dir
    gtruthdir(str): gtruth dir
    outpath(str): csv output path
    """
    info(inspect.stack()[0][3] + '()')

    files = []
    for f in sorted(os.listdir(preddir)):
        if f.endswith('.png'): files.append(f)

    n = len(files)
    ious = np.ndarray(n, dtype=float)
    for i, f in enumerate(files):
        pred = imageio.imread(pjoin(preddir, f))
        gtruth = imageio.imread(pjoin(gtruthdir, f))
        ious[i] = calculate_miou(pred, gtruth)

    dfdict = dict(
            id = files,
            iou = ious,
            )
    df = pd.DataFrame(dfdict)
    df.to_csv(outpath, index=False)

#############################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('viafile', help='Path to via file in JSON format.')
    parser.add_argument('imdir', help='Path to images referenced in the via file')
    parser.add_argument('outdir', help='Outputdir')
    N = 5

    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)
    parse_via_file_parallel(args.viafile, args.imdir, args.outdir, N)

if __name__ == "__main__":
    main()

