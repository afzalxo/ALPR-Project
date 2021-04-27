import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from time import time

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
        if use_gpu:
            x = Variable(XI.cuda(0))
        else:
            x = Variable(XI)
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
    return count, correct, error, float(correct) / count, (time() - start) / count


def train_model(model, criterion, optimizer, trainloader, valloader, lrScheduler, batchSize, num_epochs=25):
    # since = time.time()
    for epoch in range(0, num_epochs):
        lossAver = []
        model.train(True)
        start = time()

        for i, (XI, Y, labels, ims) in enumerate(trainloader):
            if not len(XI) == batchSize:
                continue

            YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
            Y = np.array([el.numpy() for el in Y]).T
            x = Variable(XI.cuda(0))
            y = Variable(torch.FloatTensor(Y).cuda(0), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model

            try:
                fps_pred, y_pred = model(x)
            except:
                continue

            # Compute and print loss
            loss = 0.0
            loss += 0.8 * nn.L1Loss().cuda()(fps_pred[:][:2], y[:][:2])
            loss += 0.2 * nn.L1Loss().cuda()(fps_pred[:][2:], y[:][2:])
            for j in range(7):
                l = Variable(torch.LongTensor([el[j] for el in YI]).cuda(0))
                loss += criterion(y_pred[j], l)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            try:
                lossAver.append(loss.data[0])
            except:
                pass

            if i % 10 == 1:
                #with open(args['writeFile'], 'a') as outF:
                print('train %s images, use %s seconds, loss %s\n' % (i*batchSize, time() - start, sum(lossAver) / len(lossAver) if len(lossAver)>0 else 'NoLoss'))
                torch.save(model.state_dict(), storeName)
        print ('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        lrScheduler.step()
        model.eval()
        count, correct, error, precision, avgTime = eval(model, valloader)
        #with open(args['writeFile'], 'a') as outF:
        print('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
        print('*** total %s error %s precision %s avgTime %s\n' % (count, error, precision, avgTime))
        torch.save(model.state_dict(), storeName + str(epoch))
    return model

