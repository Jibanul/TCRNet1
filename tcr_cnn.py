import torch.nn as nn
import torch
import numpy as np
import os

class tcr_net(nn.Module):
    def __init__(self, weightpath, learn_filters=0):
        super(tcr_net, self).__init__()
        self.cnv1 = nn.Conv2d(in_channels=1, out_channels=70, kernel_size=(20,40), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(70)
        self.cnv2 = nn.Conv2d(in_channels=70, out_channels=50, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(50)
        self.cnv3 = nn.Conv2d(in_channels=50, out_channels=30, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(30)
        self.cnv4 = nn.Conv2d(in_channels=30, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.leakyrelu = nn.LeakyReLU(0.1)
        if learn_filters == 0 and os.path.isfile(os.path.join(weightpath, 'target_filters.npy')) and os.path.isfile(os.path.join(weightpath, 'clutter_filters.npy')):
            target_filters = np.load(os.path.join(weightpath, 'target_filters.npy'))
            clutter_filters = np.load(os.path.join(weightpath, 'clutter_filters.npy'))
            print('targets', target_filters.shape)
            print('clutter', clutter_filters.shape)
            qcf_filter = np.concatenate((clutter_filters[:, 0:20], target_filters[:, -50:]), axis=1)
            qcf_filter = np.swapaxes(qcf_filter, 0, 1)
            qcf_filter = qcf_filter.reshape((-1, 40, 20))
            qcf_filter = np.expand_dims(qcf_filter, axis=1)
            qcf_filter = np.swapaxes(qcf_filter, 2, 3)
            print('qcf', qcf_filter.shape)
            layer1 = torch.tensor(qcf_filter).float()
            self.cnv1.weight = nn.Parameter(layer1)
            self.cnv1.weight.requires_grad = False

    def forward(self, x):
        x = self.leakyrelu(self.bn1(self.cnv1(x)))
        x = self.leakyrelu(self.bn2(self.cnv2(x)))
        x = self.leakyrelu(self.bn3(self.cnv3(x)))
        x = self.cnv4(x)
        # x = x **2
        return x

class tcr_loss(nn.Module):
    def __init__(self):
        super(tcr_loss, self).__init__()

    def forward(self, predictions, gt):
        # print('predictions',predictions.shape)
        # print('gt',gt.shape)
        sum = torch.sum(gt,dim=3)
        sum = torch.sum(sum,dim=2)
        clutter_idx = torch.where(sum == 0)[0]
        target_idx = torch.where(sum != 0)[0]

        # print('clutter',clutter_idx,'targets',target_idx)

        target_response = predictions[target_idx,:,:,:].squeeze()
        clutter_response = predictions[clutter_idx,:,:,:].squeeze()

        target_response = target_response **2
        clutter_response = clutter_response **2

        target_peak = target_response[:,8,18]  #corresponds to gaussian peak in gt, detect instead of hard code later
        clutter_energy = torch.sum(clutter_response,dim=2)
        clutter_energy = torch.sum(clutter_energy,dim=1)

        # print('peak',target_peak.shape,'clutter',clutter_energy)
        n1 = target_peak.shape[0]
        n2 = clutter_energy.shape[0]
        if n1 != 0:
            loss1 = torch.log(target_peak.sum()/n1)
        else:
            loss1 = 0

        if n2 != 0:
            loss2 = torch.log(clutter_energy.sum()/n2)
        else:
            loss2 = 0

        loss = loss2 - loss1
        # print('losses',loss2,loss1)
        return loss

