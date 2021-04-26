import cv2
import torch
import torch.nn as nn
from torch.utils.data import *
from imutils import paths
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import os
import argparse
from time import time
from torch.optim import lr_scheduler
from utils.dsetparser import parse_dset_config
from models.wR2 import wR2
from dataloader.loader import ChaLocDataLoader
from trainer.trainer import train_model 

USE_WANDB = True
if USE_WANDB:
    import wandb
    wandb.init(project='alpr', entity='afzal', name='detection_reluo')

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dsetconf", default='dset_config/config.ccpd',
                help="path to the dataset config file")
ap.add_argument("-n", "--epochs", default=300,
                help="epochs for train")
ap.add_argument("-b", "--batchsize", default=5,
                help="batch size for train")
args = vars(ap.parse_args())

def main():
    numClasses = 4
    imgSize = (480, 480)
    origSize = (720, 1160)
    batchSize = args['batchsize']
    epochs = args['epochs']
    lr = 0.001
    momentum = 0.9

    if USE_WANDB:
        config = wandb.config
        config.imgSize = imgSize
        config.batchSize = batchSize
        config.epochs = epochs
        config.lr = lr
        config.momentum = momentum
        wandb.save('./*.py')

    model = wR2(numClasses)
    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())) # This piece of shit hangs 
            #the node pretty fucking badly, to the point that the script process is unkillable and have to restart 
            #the node to restore operation, which results in a stopped docker container removing all its contents. 
            #(https://github.com/pytorch/pytorch/issues/24081#issuecomment-557074611). Cant disable IOMMU in BIOS 
            #since working on a remote node. Got no choice but to work with a single GPU. 
            #In summary, as Linus Torvalds would say: FUCK YOU NVIDIA
    model = model.cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    lrScheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    dset_conf = parse_dset_config(args['dsetconf'])
    #Loading Train Split
    trainloc=dset_conf['train']
    dst = ChaLocDataLoader(trainloc, imgSize)
    trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=4)
    #Loading Validation Split
    valloc=dset_conf['val']
    valdst = ChaLocDataLoader(valloc, imgSize)
    evalloader = DataLoader(valdst, batch_size=batchSize, shuffle=False, num_workers=4)
    print('Starting Training...')
    model_conv = train_model(model, criterion, optimizer, lrScheduler, trainloader, evalloader, batchSize, num_epochs=epochs, USE_WANDB=USE_WANDB)

if __name__=='__main__':
    main()

#image_loc = ['/media/largeHDD/afzal/dl_project/placeholder_data/']
#loader = ChaLocDataLoader(image_loc, img_size)
#item, label = loader.__getitem__(0)
#draw_bbox(item, label)
