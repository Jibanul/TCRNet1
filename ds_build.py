import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import torchvision
import torch
from PIL import Image
import csv

CEGR_RANGE_MAP_VEH = {
    '01923': 1000,    '01925': 1500,    '01927': 2000,    '01929': 2500,
    '01931': 3000,    '01933': 3500,    '01935': 4000,    '01937': 4500,
    '01939': 5000,    '02003': 1000,    '02005': 1500,    '02007': 2000,
    '02009': 2500,    '02011': 3000,    '02013': 3500,    '02015': 4000,
    '02017': 4500,    '02019': 5000
}
CEGR_RANGE_MAP_FULL = {
    '01923': 1000,    '01925': 1500,    '01927': 2000,    '01929': 2500,
    '01931': 3000,    '01933': 3500,    '01935': 4000,    '01937': 4500,
    '01939': 5000,    '02003': 1000,    '02005': 1500,    '02007': 2000,
    '02009': 2500,    '02011': 3000,    '02013': 3500,    '02015': 4000,
    '02017': 4500,    '02019': 5000,    '01926': 500,    '01928': 1000,
    '01932': 1500,    '01934': 2000,    '01936': 2500,    '01938': 3000,
    '02002': 500,    '02004': 1000,    '02006': 1500,    '02008': 2000,
    '02010': 2500,    '02012': 3000,
}


def get_range(imfn):
    head, tail = os.path.split(imfn)
    if tail[4:9] in CEGR_RANGE_MAP_VEH:
        range = CEGR_RANGE_MAP_VEH[tail[4:9]]
    else:
        range = -1
    # if tail[:4] == 'cegr':
    #     range = CEGR_RANGE_MAP_FULL[tail[4:9]]
    # else:
    #     assert False, "only ATR Database images supported, this doesn't appear to be one."
    return range


def get_anno(imfn, truthfn):
    head, tail = os.path.split(imfn)
    anno = []
    with open(truthfn, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if tail == row[0]:
                anno.append([np.array([int(row[i+1]) for i in range(4)]), row[5]])
    return anno


def box2center(box):
    left, upper, right, lower = box
    x = (left + right) / 2
    y = (upper + lower) / 2
    return np.array([x, y])


def center2box(center, sz):

    x, y = np.round(center)
    x = int(x)
    y = int(y)

    halfsz = int(sz[0]/2), int(sz[1]/2)

    left = x - halfsz[0]
    upper = y - halfsz[1]
    right = x + halfsz[0]
    lower = y + halfsz[1]
    return left, upper, right, lower


def grabchips(imfn, annos, chipsz=(80, 40)):
    # read image
    image = Image.open(imfn)
    imsz = np.array(image.size)

    # get image annotations
    # annos = get_anno(imfn, truthfn)
    centers = [box2center(a[0]) for a in annos]

    # resize image to 2500 km
    scale = get_range(imfn) / 2500
    assert scale > 0, "possibly missing seq number"
    scaled_image = image.resize((imsz*scale).astype(int))

    targets = []
    clutters = []
    for center in centers:
        # get target chips centered on annotations
        chip = scaled_image.crop(center2box(center*scale, sz=chipsz))
        if chip.size != (80,40):
            print(chip)
        # chip.show()
        targets.append(chip)

        # get clutter chips not overlapping annotations
        clutterbox = None
        while clutterbox is None:
            center = np.random.rand(2) * (imsz*scale - 2 * np.array(chipsz)) + chipsz
            box = center2box(center, sz=chipsz)

            if torchvision.ops.boxes.box_iou(torch.Tensor([a[0]*scale for a in annos]), torch.Tensor([box])).sum() == 0:
                clutterbox = box
                # print(box)
                # print(annos)

        clutter = scaled_image.crop(clutterbox)
        # clutter.show()
        clutters.append(clutter)

        return targets, clutters

def save_chip(chip, fn):
    scipy.io.savemat(fn, {'image': np.array(chip)[:,:,0]*1.0})


def save_chips(outpath, imlist, annlist, chipsz=(80, 40)):
    assert len(imlist) == len(annlist)
    os.makedirs(os.path.join(outpath, 'targets'), exist_ok=True)
    os.makedirs(os.path.join(outpath, 'clutter'), exist_ok=True)

    # ind = 0
    for imfn, ann in zip(imlist, annlist):
        targets, clutters = grabchips(imfn, ann, chipsz=chipsz)
        head, tail = os.path.split(imfn)
        root, ext = os.path.splitext(tail)

        for i in range(len(targets)):
            outfn = os.path.join(outpath, 'targets', root + str(i) + '.mat')
            save_chip(targets[i], outfn)

            outfn = os.path.join(outpath, 'clutter', root + str(i) + '.mat')
            save_chip(clutters[i], outfn)

def read_annos(truthfn):
    annos = {}
    with open(truthfn, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row[0] not in annos:
                annos[row[0]] = []
            annos[row[0]].append([np.array([int(row[i+1]) for i in range(4)]), row[5]])
    return annos

def get_ims_and_anns(seqlist, datapath, skip):
    with open(seqlist, 'r') as f:
        seqs = f.readlines()
    seqs = [seq.strip() for seq in seqs]

    imlist = []
    annlist = []
    # ims_and_anns = []
    for seq in seqs:
        range = get_range(seq)

        if range<0 or range>=4500 or seq=='cegr02015_0097':
            print('skipping sequence', seq)
            continue

        print('starting sequence', seq)
        annfn = os.path.join(datapath, 'GT', seq + '.txt')
        annos = read_annos(annfn)

        curimlist = list(annos.keys())
        curimlist = curimlist[::skip]

        curannlist = [annos[im] for im in curimlist]
        curimlist = [os.path.join(datapath, 'IMAGES', im) for im in curimlist]
        imlist += curimlist
        annlist += curannlist

    return imlist, annlist


def build_ds(seqlist, outpath, datapath='/media/ssd2/aitr/ATR_Database/processed3', skip=5):
    imlist, annlist = get_ims_and_anns(seqlist, datapath, skip)
    save_chips(outpath, imlist, annlist)

    with open(seqlist, 'r') as f:
        seqs = f.readlines()
    seqs = [seq.strip() for seq in seqs]
    #
    # for seq in seqs:
    #     range = get_range(seq)
    #
    #     if range<0 or range>=4500 or seq=='cegr02015_0097':
    #         print('skipping sequence', seq)
    #         continue
    #
    #     print('starting sequence', seq)
    #     annfn = os.path.join(datapath, 'GT', seq + '.txt')
    #     annos = read_annos(annfn)
    #
    #     imlist = list(annos.keys())
    #     imlist = imlist[::skip]
    #
    #     annlist = [annos[im] for im in imlist]
    #     imlist = [os.path.join(datapath, 'IMAGES', im) for im in imlist]
    #
    #     save_chips(outpath, imlist, annlist)

    return seqs

# seqs = build_ds('data/trainlist.txt','data/exp1')


# grabchips('/media/ssd2/aitr/ATR_Database/processed3/IMAGES/cegr01923_0001_0001.png', '/media/ssd2/aitr/ATR_Database/processed3/GT/cegr01923_0001.txt')
# grabchips('/media/ssd2/aitr/ATR_Database/processed3/IMAGES/cegr01925_0001_0001.png', '/media/ssd2/aitr/ATR_Database/processed3/GT/cegr01925_0001.txt')
# grabchips('/media/ssd2/aitr/ATR_Database/processed3/IMAGES/cegr01933_0001_0001.png', '/media/ssd2/aitr/ATR_Database/processed3/GT/cegr01933_0001.txt')
# grabchips('/media/ssd2/aitr/ATR_Database/processed3/IMAGES/cegr01937_0001_0001.png', '/media/ssd2/aitr/ATR_Database/processed3/GT/cegr01937_0001.txt')


# imlist = [
# '/media/ssd2/aitr/ATR_Database/processed3/IMAGES/cegr01923_0001_0001.png',
# '/media/ssd2/aitr/ATR_Database/processed3/IMAGES/cegr01925_0001_0001.png',
# '/media/ssd2/aitr/ATR_Database/processed3/IMAGES/cegr01933_0001_0001.png',
# '/media/ssd2/aitr/ATR_Database/processed3/IMAGES/cegr01937_0001_0001.png'
# ]
# annlist = [
# '/media/ssd2/aitr/ATR_Database/processed3/GT/cegr01923_0001.txt',
# '/media/ssd2/aitr/ATR_Database/processed3/GT/cegr01925_0001.txt',
# '/media/ssd2/aitr/ATR_Database/processed3/GT/cegr01933_0001.txt',
# '/media/ssd2/aitr/ATR_Database/processed3/GT/cegr01937_0001.txt'
# ]
#
# save_chips('/home/thuster/PycharmProjects/TCRNet-1/data/debug', imlist, annlist)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', type=int, default=5)
    parser.add_argument('--seqlist', type=str, default='data/trainlist.txt')
    parser.add_argument('--outpath', type=str, default='data/exp1')
    parser.add_argument('--datapath', type=str, default='data/atrdb')

    args = parser.parse_args()
    build_ds(args.seqlist, args.outpath, datapath=args.datapath, skip=args.skip)


