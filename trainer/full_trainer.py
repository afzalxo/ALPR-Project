import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from time import time
from utils.utils import IoU
import wandb

def isEqual(labelGT, labelP):
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    # print(sum(compare))
    return sum(compare)


def eval(model, testloader):
    count, error, correct = 0, 0, 0
    #dst = labelTestDataLoader(test_dirs, imgSize)
    #testloader = DataLoader(dst, batch_size=1, shuffle=True, num_workers=8)
    start = time()
    for i, (XI, labels, ims) in enumerate(testloader):
        count += 1
        YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
        x = Variable(XI.cuda())
        # Forward pass: Compute predicted y by passing x to the model

        fps_pred, y_pred = model(x)

        outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
        labelPred = [t[0].index(max(t[0])) for t in outputY]

        #   compare YI, outputY
        try:
            if isEqual(labelPred, YI[0]) == 7:
                correct += 1
            else:
                pass
        except:
            error += 1
        print(f'Evaluated: {count}, Correct: {correct}, Error: {error}')
    return count, correct, error, float(correct) / count, (time() - start) / count


def train_model(model, criterion, optimizer, trainloader, valloader, lrScheduler, batchSize, storeName, USE_WANDB, num_epochs=25):
    # since = time.time()
    for epoch in range(1, num_epochs+1):
        lossAver = []
        detlossA = []
        reclossA = []
        model.train(True)
        start = time()
        correct_pred = 0
        dset_len = len(trainloader)

        for i, (XI, Y, labels, ims) in enumerate(trainloader):
            if not len(XI) == batchSize:
                continue

            YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
            Y = np.array([el.numpy() for el in Y]).T
            x = Variable(XI.cuda())
            y = Variable(torch.FloatTensor(Y).cuda(), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model

            fps_pred, y_pred = model(x)

            # Compute and print loss
            loss = 0.0
            loss_det = 0.0
            loss_rec = 0.0
            loss_det += 0.8 * nn.L1Loss().cuda()(fps_pred[:,:2], y[:,:2])
            loss_det += 0.2 * nn.L1Loss().cuda()(fps_pred[:,2:], y[:,2:])
            for j in range(7):
                l = Variable(torch.LongTensor([el[j] for el in YI]).cuda())
                loss_rec += criterion(y_pred[j], l)
            loss = loss_det + loss_rec
            #Compute training accuracy
            batch_iou = IoU(Y*480., 480.*fps_pred.cpu().detach().numpy())
            bin_correct = (batch_iou >= 0.7)
            correct_pred += np.sum(bin_correct)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossAver.append(loss.item())
            detlossA.append(loss_det.item())
            reclossA.append(loss_rec.item())

            if i % 500 == 1:
                curloss = sum(lossAver) / len(lossAver)
                detL = sum(detlossA) / len(detlossA)
                recL = sum(reclossA) / len(reclossA)
                curacc = correct_pred / ((i+1)*batchSize)
                print('train %s images, use %s seconds, loss %s, det loss %s, rec loss %s, det acc %s\n' % (i*batchSize, time() - start, curloss if len(lossAver)>0 else 'NoLoss', detL, recL, curacc))
                torch.save(model.state_dict(), storeName)
                if USE_WANDB:
                    wandb.log({'Net Training Loss':curloss, 'Detection Loss':detL, 'Recognition Loss':recL, 'Current Training Acc':curacc}, step=int((epoch-1)*dset_len/500+i/500))
        print ('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        lrScheduler.step()
        model.eval()
        count, correct, error, precision, avgTime = eval(model, valloader)
        #with open(args['writeFile'], 'a') as outF:
        print('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
        print('*** total %s error %s precision %s avgTime %s\n' % (count, error, precision, avgTime))
        torch.save(model.state_dict(), storeName + str(epoch))
        if USE_WANDB:
            wandb.save(storeName+str(epoch))
    return model

