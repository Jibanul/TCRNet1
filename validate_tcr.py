import glob
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import json
from scipy.io import loadmat
from skimage.transform import resize
import tcr_cnn
import os
import ds_build
from PIL import Image

def scale(image, factor):
    x, y = image.shape
    x = int(round(factor * x))
    y = int(round(factor * y))
    return resize(image,(x, y))

def pad(image,nrows,ncols):
    out = np.zeros((nrows,ncols))
    m, n = image.shape
    o1 = nrows/2 + 1
    o2 = ncols/2 + 1
    r1 = int(round(o1 - m/2))
    r2 = int(round(r1 + m - 1))
    c1 = int(round(o2 - n/2))
    c2 = int(round(c1 + n -1))
    out[r1:r2+1,c1:c2+1] = image
    return out

def get_detections(input_image,ndetects):
    image = input_image.copy()
    minval = image.min()
    nrows,ncols = image.shape
    confs=[]
    row_dets=[]
    col_dets=[]
    for i in range(ndetects):
        row,col = np.unravel_index(image.argmax(), image.shape)
        val = image[row,col]
        r1 = max(row - 10, 0)
        r2 = min(r1 + 19, nrows)
        r1 = r2 - 19
        c1 = max(col - 20, 1)
        c2 = min(c1 + 39, ncols)
        c1 = c2 - 39
        image[r1: r2+1, c1:c2+1]=np.ones((20, 40)) * minval
        confs.append(val)
        row_dets.append(row)
        col_dets.append(col)

    confs = np.array(confs)
    Aah = confs.std() * 6** .5 / 3.14158 #wtf
    cent = confs.mean() - Aah * 0.577216649 #wtf
    confs = (confs - cent) / Aah

    row_dets = np.array(row_dets)
    col_dets = np.array(col_dets)

    return confs, row_dets, col_dets



def show(im):
    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(im)
    plt.show()


def run(seqlist, weightpath, datapath='/media/ssd2/aitr/ATR_Database/processed3', skip=30, square=1):
    net = tcr_cnn.tcr_net(weightpath).cuda()
    net.load_state_dict(torch.load(os.path.join(weightpath, 'trainedweights.pth')))

    # images = os.listdir(imgdir)

    imlist, annlist = ds_build.get_ims_and_anns(seqlist, datapath, skip)

    index = 0
    dets = []
    fas = []
    nframes = 0
    ntgt = 0
    for imfn, anns in zip(imlist, annlist):
        # if imfn[-4:]!='.mat':
        #     continue
        # print(sample['name'] + "_" + sample['frame'])
        # imfile = os.path.join(imgdir, imfn)

        # imfile = imgdir + sample['name'] + '_' + sample['frame'] + '.mat'
        # im = loadmat(imfn)['image']
        im = Image.open(imfn)
        im = np.array(im)[:, :, 0] * 1.0
        im = (im - 131.95) / 54.29

        target_range = ds_build.get_range(imfn)

        # target_range = sample['range'] * 1000
        scale_factor = target_range / 2500

        im = scale(im, scale_factor)
        nrows, ncols = im.shape
        # show(im)
        im = torch.tensor(im).unsqueeze(0).unsqueeze(0).float().cuda()
        output = net(im)
        if square==1:
            output = output ** 2
        output = output.cpu().detach()[0, 0, :, :].numpy()
        # show(output)
        output = pad(output, nrows, ncols)
        # confs: detection score, & row_dets, col_dets: detection location
        confs, row_dets, col_dets = get_detections(output, 30)
        row_dets = row_dets / scale_factor
        col_dets = col_dets / scale_factor

        targets = anns
        # targets = sample['targets']
        nt = len(targets)
        ndets = confs.shape[0]
        ntgt += nt
        nframes += 1

        foundtgt = np.zeros(ndets)

        for target in targets:
            x, y = ds_build.box2center(target[0])
            c = x
            r = y

            # r = target['center'][1]
            # c = target['center'][0]
            tmpdets = []
            for i in range(ndets):
                dist = ((r - row_dets[i]) ** 2 + (c - col_dets[i]) ** 2) ** .5
                # print(dist)
                if dist < 20:
                    foundtgt[i] = 1
                    tmpdets.append(confs[i])
            if len(tmpdets) >= 1:
                dets.append(max(tmpdets))
            # print('dets',dets)
            I = np.where(foundtgt == 0)[0]
            for a in confs[I]:
                fas.append(a)
            # print('fas',fas)

    # correct detection scores
    dets = np.array(dets)
    # np.save('./output/dets', dets)
    # false alarm/detection scores
    fas = np.array(fas)
    # np.save('./output/fas', fas)
    # number of tagrgets in the test data
    ntgt = np.array([ntgt])
    # np.save('./output/ntgt', ntgt)
    # number of frames in the test data
    nframes = np.array([nframes])
    # np.save('./output/nframes', nframes)
    print(ntgt, nframes, len(dets), len(fas))
    make_roc(dets, fas, ntgt, nframes)

# import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
def make_roc(dets, fas, ntgt, nframes):

    # dets = np.load('./output/dets.npy')
    # fas = np.load('./output/fas.npy')
    # ntgt = np.load('./output/ntgt.npy')[0]
    # nframes = np.load('./output/nframes.npy')[0]
    # print(ntgt, nframes, len(dets), len(fas))

    all_scores = np.concatenate([dets, fas])
    all_cls = np.concatenate([np.ones([len(dets)]), np.zeros([len(fas)])])
    print('MAP:', average_precision_score(all_cls, all_scores))


    maxv = max(max(fas), max(dets))
    minv = min(min(fas), min(dets))
    step = (maxv - minv) / 1000
    print(maxv, minv)
    pds = []
    fars = []
    t = minv
    while t < maxv:
        x = np.where(dets > t)
        pd = x[0].shape / ntgt
        pds.append(pd)

        y = np.where(fas > t)
        far = y[0].shape / (nframes * 3.4 * 2.6)
        fars.append(far)
        t += step

    # print(len(pds), len(fars))
    # plt.plot(fars, pds)
    # plt.savefig("./output/" + "roc_curve_TCRNet1_example")
    # plt.show()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', type=int, default=30)
    parser.add_argument('--seqlist', type=str, default='data/trainlist.txt')
    parser.add_argument('--weightpath', type=str, default='./weights_filters/exp1')
    parser.add_argument('--datapath', type=str, default='data/atrdb')
    parser.add_argument('--square', type=int, default=1)

    args = parser.parse_args()

    run(args.seqlist, args.weightpath, datapath=args.datapath, skip=args.skip, square=args.square)
