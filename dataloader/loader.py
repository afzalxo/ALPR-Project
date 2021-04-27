import cv2
import numpy as np
from torch.utils.data import *
import os

#Courtesy of https://github.com/detectRecog/CCPD/
class ChaLocDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        with open(img_dir, 'r') as file:
            self.img_paths = file.readlines()

        #self.img_dir = img_dir
        #self.img_paths = []
        #for i in range(len(img_dir)):
        #    self.img_paths += [el for el in paths.list_images(img_dir[i])]
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img_name = os.path.join('CCPD2019', img_name)
        img = cv2.imread(img_name.strip())
        if img is None:
            print('Could\'nt read image at path {}'.format(img_name))
            print('Later, Loser...')
            exit(0)
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))

        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        #---Uncomment to generate image with bounding box
        #cv2.rectangle(img, (leftUp[0], leftUp[1]), (rightDown[0], rightDown[1]), (255, 0, 0), 1) 
        #cv2.imwrite('res1.jpg', img)
        #---

        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        assert img.shape[0] == 1160
        new_labels = [(leftUp[0] + rightDown[0])/(2*ori_w), (leftUp[1] + rightDown[1])/(2*ori_h), (rightDown[0]-leftUp[0])/ori_w, (rightDown[1]-leftUp[1])/ori_h]

        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0

        return resizedImage, new_labels

class LabelFpsDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        with open(img_dir, 'r') as file:
            self.img_paths = file.readlines()


        #self.img_dir = img_dir
        #self.img_paths = []
        #for i in range(len(img_dir)):
        #    self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img_name = os.path.join('CCPD2019', img_name)
        img = cv2.imread(img_name.strip())
        if img is None:
            print('Could\'nt read image at path {}'.format(img_name))
            print('Later, Loser...')
            exit(0)
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2,0,1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]

        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        # fps = [[int(eel) for eel in el.split('&')] for el in iname[3].split('_')]
        # leftUp, rightDown = [min([fps[el][0] for el in range(4)]), min([fps[el][1] for el in range(4)])], [
        #     max([fps[el][0] for el in range(4)]), max([fps[el][1] for el in range(4)])]
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
        new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                      (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]

        return resizedImage, new_labels, lbl, img_name

class LabelTestDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        with open(img_dir, 'r') as file:
            self.img_paths = file.readlines()
        #self.img_dir = img_dir
        #self.img_paths = []
        #for i in range(len(img_dir)):
        #    self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img_name = os.path.join('CCPD2019', img_name)
        img = cv2.imread(img_name.strip())
        if img is None:
            print('Could\'nt read image at path {}'.format(img_name))
            print('Later, Loser...')
            exit(0)
        # img = img.astype('float32')
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2,0,1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        lbl = img_name.split('/')[-1].split('.')[0].split('-')[-3]
        return resizedImage, lbl, img_name
