import glob
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import tcr_cnn
import os
from qcf_basis import get_chiparray

def show(im):
    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(im)
    plt.show()

class nvesd_cnn_train(Dataset):
    def __init__(self, chippath, weightpath):

        target_files  = glob.glob(os.path.join(chippath,'targets', '*.mat'))
        clutter_files = glob.glob(os.path.join(chippath, 'clutter', '*.mat'))
        gaussian = loadmat('h.mat')['h']

        d1 = 40
        d2 = 80
        count = len(target_files)
        nt = count
        print(count, 'target chips')
        count = len(clutter_files)
        nc = count
        print(count, 'clutter chips')
        alltargets = np.zeros((d1, d2, count))
        for idx, chip in enumerate(target_files):
            # chiparray = loadmat(chip)['target_chip']
            chiparray = get_chiparray(chip)
            # chiparray = chiparray - chiparray.mean()
            # chiparray = (chiparray - 8008.)/2273.

            alltargets[:, :, idx] = chiparray

        allclutter = np.zeros((d1, d2, count))
        for idx, chip in enumerate(clutter_files):
            # chiparray = loadmat(chip)['clutter_chip']
            chiparray = get_chiparray(chip)
            # chiparray = chiparray - chiparray.mean()
            # chiparray = (chiparray - 8008.) / 2273.
            allclutter[:, :, idx] = chiparray

        print('clutter',allclutter.shape)

        yt = np.tile(gaussian,(nt,1,1))
        print('yt',yt.shape)

        yc = np.tile(np.zeros((17,37)),(nc,1,1)) #wtf is this?
        print('yc',yc.shape)

        self.x = np.concatenate((alltargets,allclutter),axis=2)
        print('x',self.x.shape)

        self.y = np.concatenate((yt,yc),axis=0)
        print('y',self.y.shape)

    def __len__(self):
        return self.x.shape[2]

    def __getitem__(self, idx):
        x = self.x[:,:,idx]
        x = np.expand_dims(x, axis=0)
        y = self.y[idx,:,:]
        y = np.expand_dims(y, axis=0)
        return x,y

import argparse
import qcf_basis

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chippath', type=str, default='./data/exp4')
    parser.add_argument('--weightpath', type=str, default='./weights_filters/exp4xx')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--loss', type=str, default='tcr')
    parser.add_argument('--wd', type=float, default=.01)
    parser.add_argument('--learn_filters', type=int, default=0)

    args = parser.parse_args()

    if args.learn_filters == 0:
        qcf_basis.run_qcf(args.chippath, args.weightpath)

    net = tcr_cnn.tcr_net(args.weightpath, learn_filters=args.learn_filters).cuda()
    epochs = args.epochs
    trainset = nvesd_cnn_train(args.chippath,args.weightpath)
    trainloader = DataLoader(
        dataset=trainset,
        batch_size=100,
        num_workers=5,
        shuffle=True
    )
    results = []

    if args.loss =='tcr':
        criterion = tcr_cnn.tcr_loss()
    elif args.loss =='mse':
        criterion = nn.MSELoss()
    elif args.loss =='bce':
        # criterion = nn.BCELoss()
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        assert False, "bad loss function"
    # criterion = tcr_cnn.tcr_loss()
    optimizer = optim.RMSprop(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=.1)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # print(i,data[0].shape,data[1].shape)
            x = data[0].float().cuda()
            gt = data[1].float().cuda()
            optimizer.zero_grad()
            outputs = net(x)
            if args.loss =='bce':
                outputs = outputs.mean(axis=3).mean(axis=2)
                gt = gt.max(axis=3)[0].max(axis=2)[0]
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()

            # print(loss.item())
#
#             # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, running_loss/10))
                running_loss = 0.0
#         scheduler.step()
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())

    os.makedirs(args.weightpath, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.weightpath, 'trainedweights.pth'))
    print('Finished Training')





