from py.test import approx
import numpy as np
import imageio
import os
from os.path import join as pjoin

from src import utils

def test_calculate_miou():
    pred = np.array([ [0, 0, 0], [0, 0, 0], [0, 0, 0], ])
    gtruth = np.array([ [0, 0, 0], [0, 1, 1], [0, 0, 0], ])
    assert utils.calculate_miou(pred, gtruth) == 0

    pred = np.array([ [0, 0, 0], [0, 1, 0], [0, 0, 0], ])
    assert utils.calculate_miou(pred, gtruth) == .5

    pred = np.array([ [0, 0, 0], [0, 1, 1], [0, 1, 0], ])
    assert utils.calculate_miou(pred, gtruth) == 2/3

    gtruth = np.array([ [0, 0, 0], [0, 0, 0], [0, 0, 0], ])
    assert utils.calculate_miou(pred, gtruth) == 0

def test_calculate_miou_batch(tmpdir):
    preddir = pjoin(tmpdir, 'pred')
    gtrudir = pjoin(tmpdir, 'gtru')
    os.mkdir(preddir)
    os.mkdir(gtrudir)

    gtru = np.array([ [0, 0, 0], [0, 1, 1], [0, 0, 0], ], dtype=np.uint8)
    pred = np.array([ [0, 0, 0], [0, 0, 0], [0, 0, 0], ], dtype=np.uint8)
    imageio.imwrite(os.path.join(gtrudir, 'im1.png'), gtru)
    imageio.imwrite(os.path.join(preddir, 'im1.png'), pred)

    pred = np.array([ [0, 0, 0], [0, 1, 0], [0, 0, 0], ], dtype=np.uint8)
    imageio.imwrite(os.path.join(gtrudir, 'im2.png'), gtru)
    imageio.imwrite(os.path.join(preddir, 'im2.png'), pred)

    pred = np.array([ [0, 0, 0], [0, 1, 1], [0, 1, 0], ], dtype=np.uint8)
    imageio.imwrite(os.path.join(gtrudir, 'im3.png'), gtru)
    imageio.imwrite(os.path.join(preddir, 'im3.png'), pred)

    gtru = np.array([ [0, 0, 0], [0, 0, 0], [0, 0, 0], ], dtype=np.uint8)
    imageio.imwrite(os.path.join(gtrudir, 'im4.png'), gtru)
    imageio.imwrite(os.path.join(preddir, 'im4.png'), pred)

    outpath = pjoin(tmpdir, 'out.csv')
    utils.calculate_miou_batch(preddir, gtrudir, outpath)
    lines = open(outpath).read().split('\n')
    assert lines[0] == 'id,iou'

    assert lines[1].split(',')[0] == 'im1.png'
    assert float(lines[1].split(',')[1]) == 0

    assert lines[2].split(',')[0] == 'im2.png'
    assert float(lines[2].split(',')[1]) == .5

    assert lines[3].split(',')[0] == 'im3.png'
    assert approx(float(lines[3].split(',')[1])) == 2/3

    assert lines[4].split(',')[0] == 'im4.png'
    assert approx(float(lines[4].split(',')[1])) == 0
