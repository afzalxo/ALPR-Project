import torch
import torch.nn as nn
from torch.autograd import Variable
import wandb
import numpy as np
from time import time
from utils.utils import IoU

def train_model(model, criterion, optimizer, lrScheduler, trainloader, evalloader, batchSize, num_epochs=300, USE_WANDB=False):
    for epoch in range(1, num_epochs):
        model.train()
        print(f'Starting Epoch {epoch}')
        lossAver = []
        start = time()
        dset_len = len(trainloader)
        correct_pred = 0
        for i, (XI, YI) in enumerate(trainloader):
            # print('%s/%s %s' % (i, times, time()-start))
            YI = np.array([el.numpy() for el in YI]).T
            x = Variable(XI.cuda(0))
            y = Variable(torch.FloatTensor(YI).cuda(0), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)

            # Compute and print loss
            loss = 0.0
            if len(y_pred) == batchSize:
                loss += 0.8 * nn.L1Loss().cuda()(y_pred[:,:2], y[:,:2]) #Penalizing more on box center coordinates
                loss += 0.2 * nn.L1Loss().cuda()(y_pred[:,2:], y[:,2:])
                lossAver.append(loss.item())

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_iou = IoU(YI*480, 480*y_pred.cpu().detach().numpy())
                bin_correct = (batch_iou >= 0.7)
                correct_pred += np.sum(bin_correct)

            if (i*batchSize)%4999 == 1:
                #old_img = retrieve_img(XI[0])
                #old_img = draw_bbox(old_img, YI[0],'g')
                #old_img = draw_bbox(old_img, y_pred[0],'r')
                #iou_val = IoU(YI[0:1]*480, 480*y_pred[0:1].cpu().detach().numpy())
                #write_image(old_img, str(iou_val[0])+'.jpg')
                torch.save(model.state_dict(), 'trained_models/checkpoints/save_ckpt.pth')
                if USE_WANDB:
                #    wandb.log({'Sample Detections': [wandb.Image(channels_last(old_img*255.0), caption='iou='+str(iou_val)+'.jpg')]})
                    wandb.save('trained_models/checkpoints/save_ckpt.pth')

            if i % 500 == 0:
                curloss = sum(lossAver)/len(lossAver)
                curacc = correct_pred/((i+1)*batchSize)
                if USE_WANDB:
                    wandb.log({'Current Training Loss':curloss}, step=int((epoch-1)*dset_len/500+i/500))
                    wandb.log({'Current Training Acc':curacc}, step=int((epoch-1)*dset_len/500+i/500))
                print('Epoch {}, Processed: {}, Time: {}, Loss: {}, Acc: {}'.format(epoch, i*batchSize, time()-start, curloss, curacc))
        print ('Epoch %s Trained, Loss: %s, Accuracy: %s, Time: %s\n' % (epoch, sum(lossAver) / len(lossAver), correct_pred/(dset_len*batchSize), time()-start))
        lrScheduler.step()
        if USE_WANDB:
            #wandb.log({'Epoch Train Loss': sum(lossAver)/len(lossAver)}, step = epoch)
            #wandb.log({'Epoch Train Accuracy': correct_pred/(dset_len*batchSize)}, step = epoch)
            print('Storing in wandb...')
            wandb.save('trained_models/epochs/save_epoch' + str(epoch) + '.pth')
        torch.save(model.state_dict(), 'trained_models/epochs/save_epoch' + str(epoch) + '.pth')
        #Begin Eval here
        #val_model(model, evalloader, batchSize, epoch, USE_WANDB) 
    return 

def val_model(model, evalloader, batchSize, epoch, USE_WANDB):
    model.eval()
    lossAver = []
    start = time()
    dset_len = len(evalloader)
    correct_pred = 0
    for i, (XI, YI) in enumerate(evalloader):
        YI = np.array([el.numpy() for el in YI]).T
        x = Variable(XI.cuda(0))
        y = Variable(torch.FloatTensor(YI).cuda(0), requires_grad=False)
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = 0.0
        loss += 0.8 * nn.L1Loss().cuda()(y_pred[:,:2], y[:,:2]) #Penalizing more on box center coordinates
        loss += 0.2 * nn.L1Loss().cuda()(y_pred[:,2:], y[:,2:])
        lossAver.append(loss.item())

        batch_iou = IoU(YI*480, 480*y_pred.cpu().detach().numpy())
        #print('batch_iou:')
        #print(batch_iou)
        bin_correct = (batch_iou >= 0.7)
        #print('Correct/Incorrect:')
        #print(bin_correct)
        correct_pred += np.sum(bin_correct)
        #print('Number correct so far: {}'.format(correct_pred))

        if (i*batchSize)%10000 == 0:
            #write_image(old_img, str(iou_val[0])+'.jpg')
            #print(iou_val)
            if USE_WANDB:
                old_img = retrieve_img(XI[0])
                old_img = draw_bbox(old_img, YI[0],'g')
                old_img = draw_bbox(old_img, y_pred[0],'r')
                iou_val = batch_iou[0] 
                wandb.log({'Sample Test Detections': [wandb.Image(channels_last(old_img*255.0), caption='iou='+str(iou_val)+'.jpg')]}, commit=False) #step=int((epoch-1)*dset_len*batchSize/10000 + i*batchSize/10000))

        if i % 500 == 0:
            curloss = sum(lossAver)/len(lossAver)
            print('Evaluating Epoch: {}, Processed: {}, Time: {}, Loss: {}, Accuracy: {}'.format(epoch, i*batchSize, time()-start, curloss, correct_pred/((i+1)*batchSize)))
    print ('Epoch %s Evaluated, Loss: %s, Accuracy: %s, Time Elapsed: %s\n' % (epoch, sum(lossAver) / len(lossAver), correct_pred/((i+1)*batchSize), time()-start))
    if USE_WANDB:
        wandb.log({'Eval Epoch Loss': sum(lossAver)/len(lossAver)}, step=epoch)
        wandb.log({'Eval Epoch Accuracy': correct_pred/((i+1)*batchSize)}, step=epoch)
    return


