'''
File: estimator.py
Project: MobilePose
File Created: Thursday, 8th March 2018 3:02:01 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Thursday, 8th March 2018 3:02:06 pm
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''

import itertools
import logging
import math
from collections import namedtuple

import cv2
import numpy as np
import torch

from scipy.ndimage import maximum_filter, gaussian_filter
from skimage import io, transform

from torch.autograd import Variable

class ResEstimator:
    def __init__(self, graph_path, target_size=(224, 224)):  #224 224
        self.target_size = target_size
        self.graph_path = graph_path
        self.net = torch.load(graph_path, map_location=lambda storage, loc: storage)
        self.net.eval()

    def addlayer(self, image):
        h, w = image.shape[:2]
        x = np.arange(0, w)
        y = np.arange(0, h)
        x, y = np.meshgrid(x, y)
        x = x[:,:, np.newaxis]
        y = y[:,:, np.newaxis]
        image = np.concatenate((image, x, y), axis=2)
        
        return image

    def wrap(self, image, output_size):
        image_ = image/256.0
        h, w = image_.shape[:2]
        if isinstance(output_size, int):
            if h > w:
                new_h, new_w = output_size * h / w, output_size
            else:
                new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image_, (new_w, new_h))
        pose_fun = lambda x: (x.reshape([-1,2]) * 1.0 /np.array([new_w, new_h])*np.array([w,h]))
        return {'image': image, 'pose_fun': pose_fun}
        
    def rescale(self, image, output_size):
        image_ = image/256.0
        h, w = image_.shape[:2]
        im_scale = min(float(output_size[0]) / float(h), float(output_size[1]) / float(w))
        new_h = int(image_.shape[0] * im_scale)
        new_w = int(image_.shape[1] * im_scale)
        image = cv2.resize(image_, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        left_pad =int( (output_size[1] - new_w) / 2.0)
        top_pad = int((output_size[0] - new_h) / 2.0)
        mean=np.array([0.485, 0.456, 0.406])
        pad = ((top_pad, top_pad), (left_pad, left_pad))
        image = np.stack([np.pad(image[:,:,c], pad, mode='constant', constant_values=mean[c])for c in range(3)], axis=2)
        pose_fun = lambda x: (((x.reshape([-1,2])-[left_pad, top_pad]) * 1.0 /np.array([new_w, new_h])*np.array([w,h])))
        return {'image': image, 'pose_fun': pose_fun}

    def to_tensor(self, image):
        x_mean = np.mean(image[:,:,3])
        x_std = np.std(image[:,:,3])
        y_mean = np.mean(image[:,:,4])
        y_std = np.std(image[:,:,4])
        mean=np.array([0.485, 0.456, 0.406, x_mean, y_mean])
        std=np.array([0.229, 0.224, 0.225, x_std, y_std])        
        image = torch.from_numpy(((image-mean)/std).transpose((2, 0, 1))).float()
        return image

    def inference(self, in_npimg, model):
        canvas = np.zeros_like(in_npimg)
        height = canvas.shape[0]
        width = canvas.shape[1]

        if 'resnet' in model:
            rescale_out = self.rescale(in_npimg, (227,227))
        elif 'mobilenet' in model:
            rescale_out = self.wrap(in_npimg, (224,224))
        
        image = rescale_out['image']
        image = self.addlayer(image)
        image = self.to_tensor(image)
        image = image.unsqueeze(0)
        pose_fun = rescale_out['pose_fun']

        keypoints = self.net(Variable(image))
        keypoints = keypoints.data.cpu().numpy()
        keypoints = pose_fun(keypoints).astype(int)

        return keypoints

    @staticmethod
    def draw_humans(npimg, pose, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255]]

        pairs = [[8,9],[11,12],[11,10],[2,1],[1,0],[13,14],[14,15],[3,4],[4,5],[8,7],[7,6],[6,2],[6,3],[8,12],[8,13]]
        colors_skeleton = ['r', 'y', 'y', 'g', 'g', 'y', 'y', 'g', 'g', 'm', 'm', 'g', 'g', 'y','y']
        colors_skeleton = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255]]

        for idx in range(len(colors)):
            cv2.circle(npimg, (pose[idx,0], pose[idx,1]), 3, colors[idx], thickness=3, lineType=8, shift=0)
        for idx in range(len(colors_skeleton)):
            npimg = cv2.line(npimg, (pose[pairs[idx][0],0], pose[pairs[idx][0],1]), (pose[pairs[idx][1],0], pose[pairs[idx][1],1]), colors_skeleton[idx], 3)

        return npimg

