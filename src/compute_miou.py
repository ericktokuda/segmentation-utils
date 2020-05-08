#!/usr/bin/env python3
"""Calculate mIoU given prediction results and gtruth files
"""

import argparse
import logging
import time
import os
from os.path import join as pjoin
from logging import info
import inspect

import numpy as np

def calculate_mious(preddir, gtruthdir, outpath):
    """Calculate miou for each file and save 

    Args:
    preddir, gtruthdir, outdir

    Returns:
    ret
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
        inters = np.logical_and(gtruth, pred)
        union = np.logical_or(gtruth, pred)
        ious[i] = np.sum(inters) / np.sum(union)

    dfdict = dict(
            id = files,
            iou = ious,
            )
    df = pd.DataFrame(dfdict)
    df.to_csv(outpath)


def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preddir', required=True, help='Predictions dir (png)')
    parser.add_argument('--gtruthdir', required=True, help='Gtruth dir (png)')
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    outfile = pjoin(args.outdir, 'mious.csv')
    calculate_mious(args.preddir, args.gtruthdir, outfile)

    info('Elapsed time:{}'.format(time.time()-t0))
if __name__ == "__main__":
    main()

