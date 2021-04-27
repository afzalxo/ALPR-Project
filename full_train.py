from __future__ import print_function, division
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
from torch.autograd import Variable
import numpy as np
import os
import argparse
from time import time
#from load_data import *
from utils.roi_pooling import roi_pooling_ims
from torch.optim import lr_scheduler
from dataloader.loader import ChaLocDataLoader
from dataloader.loader import LabelFpsDataLoader
from dataloader.loader import LabelTestDataLoader
from models.fh02 import fh02
from trainer.full_trainer import train_model
from utils.dsetparser import parse_dset_config

USE_WANDB = True
if USE_WANDB:
    import wandb
    wandb.init(project='alpr', name='detection full training')

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--epochs", default=300,
                help="epochs for train")
ap.add_argument("-b", "--batchsize", default=5,
                help="batch size for train")
ap.add_argument("-f", "--folder", required=True,
                help="folder to store model")
ap.add_argument("-i", "--dsetconf", default= 'dset_config/config.ccpd',
                help="folder to store model")
args = vars(ap.parse_args())

wR2Path = './wR2_ep24.pth'
use_gpu = torch.cuda.is_available()
print (use_gpu)

numClasses = 7
numPoints = 4
classifyNum = 35
imgSize = (480, 480)
# lpSize = (128, 64)
provNum, alphaNum, adNum = 38, 25, 35
batchSize = int(args["batchsize"]) if use_gpu else 2
modelFolder = str(args["folder"]) if str(args["folder"])[-1] == '/' else str(args["folder"]) + '/'
storeName = modelFolder + 'fh02.pth'
if not os.path.isdir(modelFolder):
    os.mkdir(modelFolder)


epochs = int(args["epochs"])
model_conv = fh02(numClasses, wR2Path).cuda()

print(model_conv)

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

dset_conf = parse_dset_config(args['dsetconf'])
trainloc = dset_conf['train']
valloc = dset_conf['val']

dst = LabelFpsDataLoader(trainloc, imgSize)
trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=8)
dstval = LabelTestDataLoader(valloc, imgSize)
valloader = DataLoader(dstval, batch_size=1, shuffle=True, num_workers=8)
lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv, trainloader, valloader, lrScheduler, batchSize, storeName, USE_WANDB, num_epochs=epochs)
