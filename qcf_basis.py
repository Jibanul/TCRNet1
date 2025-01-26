import numpy as np
import glob
from numpy.linalg import eig as npeig
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
from numpy.linalg import inv as npinv
import os

def check_and_crop(chiparray, sz):
    (d1, d2) = sz
    if chiparray.shape != (d1, d2):
        dd1, dd2 = chiparray.shape
        delta1 = (dd1 - d1) / 2
        delta2 = (dd2 - d2) / 2
        assert delta1 >= 0 and delta2 >= 0 and delta1 % 1 == 0 and delta2 % 1 == 0, "image sizes " + str(chiparray.shape) + " and " + str(sz) + " not compatible"
        delta1 = int(delta1)
        delta2 = int(delta2)
        chiparray = chiparray[delta1:delta1 + d1, delta2:delta2 + d2]
    return chiparray

def get_chiparray(fn):
    image = loadmat(fn)
    k = 'target_chip'
    if k not in image:
        k = 'clutter_chip'
    if k not in image:
        k = 'image'

    chiparray = image[k]
    # chiparray = (chiparray - 8008.) / 2273.
    chiparray = (chiparray - 131.95) / 54.29

    return chiparray

def correlate(datapath, weightpath):
    r2fn = os.path.join(weightpath, 'R2.npy')
    if os.path.isfile(r2fn):
        return

    target_chips = glob.glob(os.path.join(datapath, 'targets', '*.mat'))
    clutter_chips = glob.glob(os.path.join(datapath, 'clutter', '*.mat'))
    d1 = 20
    d2 = 40
    count = len(target_chips)
    print(count, 'target chips')
    alltargets = np.zeros((d1,d2,count))
    for idx,chip in enumerate(target_chips):
        chiparray = get_chiparray(chip)
        # chiparray = loadmat(chip)['target_chip']
        chiparray = check_and_crop(chiparray, (d1, d2))

        chiparray = chiparray - chiparray.mean()
        alltargets[:,:,idx] = chiparray

    R1 = np.zeros((d1*d2,d1*d2))
    for idx in range(count):
        chipvec = alltargets[:,:,idx].transpose().reshape(d1*d2,1)
        R1 = R1 + np.matmul(chipvec, chipvec.transpose())
    R1 = R1/count
    os.makedirs(weightpath, exist_ok=True)
    np.save(os.path.join(weightpath,'R1'),R1)

    count = len(clutter_chips)
    print(count, 'clutter chips')
    allclutter = np.zeros((20,40,count))
    for idx,chip in enumerate(clutter_chips):
        # chiparray = loadmat(chip)['clutter_chip']
        chiparray = get_chiparray(chip)
        chiparray = check_and_crop(chiparray, (d1, d2))
        chiparray = chiparray - chiparray.mean()
        allclutter[:,:,idx] = chiparray


    x = allclutter[:, :, 0]
    x2 = np.flipud(np.fliplr(x))
    acf = signal.convolve2d(x, x2)
    for idx in range(count-1):
        x = allclutter[:,:,idx + 1]
        x2 = np.flipud(np.fliplr(x))
        tmp = signal.convolve2d(x,x2)
        acf = (acf * idx + tmp) / (idx + 1)

    mask=np.ones((d1,d2))
    pmask=signal.convolve2d(mask,mask,'full')
    cov = acf/pmask

    m = cov.shape[0]
    n = cov.shape[1]

    ad1=int((m+1)/2)
    ad2=int((n+1)/2)
    dim=int(ad1*ad2)


    CM = np.zeros((dim,dim))
    row_index = np.kron(np.ones(ad2), np.arange(0, ad1, 1)).astype("int64")
    col_index = np.kron(np.arange(0, ad2, 1), np.ones(ad1))
    iv = np.column_stack((row_index, col_index))

    for i in range(dim):
        for j in range(dim):
            index = (iv[j, :] - iv[i, :]).astype("int64")
            row = d1 -1 + index[0]
            col = d2 -1 + index[1]
            CM[i, j] = cov[row, col]
            CM[j, i] = CM[i, j]

    R2 = CM
    np.save(r2fn, R2)


def make_basis(weightpath):
    tfn = os.path.join(weightpath, 'target_filters.npy')
    cfn = os.path.join(weightpath, 'clutter_filters.npy')
    if os.path.isfile(tfn) and os.path.isfile(cfn):
        return

    R1 = np.load(os.path.join(weightpath, 'R1.npy'))
    R2 = np.load(os.path.join(weightpath, 'R2.npy'))
    A = .18 * R1 #wtf
    B = R2
    S = A + B
    delta, phi = npeig(S)
    sdelta = delta[delta.argsort()]
    sphi = phi[:, delta.argsort()]
    tmp2= np.cumsum(sdelta)/sdelta.sum()
    skip = tmp2[tmp2 < .001].shape[0] - 1 #wtf
    sdelta = sdelta[skip:]
    sphi = sphi[:,skip:]
    # 130 CHANGES; SIZE OF SDELTA
    abc = np.matmul(sphi,npinv(sdelta *np.eye(sdelta.shape[0]))) #wtf#wtf
    Sinv = np.matmul(abc,sphi.transpose())
    T=np.matmul(Sinv,(A-B))
    delta, phi = npeig(T)
    delta = np.real(delta)
    phi = np.real(phi)
    sdelta = delta[delta.argsort()]
    sphi = phi[:, delta.argsort()]
    S1=sphi[:,sdelta > .01]#wtf
    S2=sphi[:,sdelta < -.01]#wtf
    n1 = S1.shape[1]
    n2 = S2.shape[1]
    print(n1,n2)
    # np.save('./weights_filters/target_filters',S1)
    # np.save('./weights_filters/clutter_filters',S2)
    np.save(tfn, S1)
    np.save(cfn, S2)


def view_filter(S,idx):
    img_array = S[:, idx].reshape(40, 20).transpose()
    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array)
    ax.set_title('ljk', fontsize=10)
    plt.show()


def run_qcf(datapath, weightpath):
    correlate(datapath, weightpath)
    make_basis(weightpath)

# if __name__ == '__main__':
#     correlate('./data/train/chips20x40', './weights_filters')
#     make_basis('./weights_filters')

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chippath', type=str, default='./data/train/chips20x40')
    parser.add_argument('--weightpath', type=str, default='./weights_filters/demo')

    args = parser.parse_args()
    run_qcf(args.chippath, args.weightpath)


