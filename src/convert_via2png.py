#!/usr/bin/env python3
"""Convert via format to pixel-wise label
"""

import argparse
import logging
from logging import debug
import json
import imageio
from shapely.geometry import Point, Polygon

def parse_via_file(viafile, imdir, outdir):
    """Parse via file in json format

    Args:
    viafile(str): path to via file in json format
    """

    fh = open(viafile)
    content = json.load(fh)
    for v in content.values():
        imgpath = os.path.join(imdir, v['filename'])
        z = imageio.imread(imgpath)
        break
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

