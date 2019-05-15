# coding: utf-8

'''
File: training.py
Project: MobilePose
File Created: Thursday, 8th March 2018 2:50:11 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Thursday, 8th March 2018 2:50:51 pm
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''

# remove warning
import warnings
warnings.filterwarnings('ignore')


from networks import *
from dataloader import *
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_factory import DatasetFactory
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MobilePose Demo')
    parser.add_argument('--model', type=str, default="resnet")
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--retrain', type=bool, default=True)
    args = parser.parse_args()
    modeltype = args.model

    # user defined parameters
    num_threads = 10 #how many cores will you use to load data

    if modeltype =='resnet':
        modelname = "final-aug.t7"
        batchsize = 256
        minloss = 316.52189376 #changed expand ratio
        # minloss = 272.49565467 #fixed expand ratio
        learning_rate = 1e-05
        net = Net().cuda()
        inputsize = 224
    elif modeltype == "mobilenet":
        modelname = "final-aug.t7"
        batchsize = 64 #original is 128
        minloss = 396.84708708 # change expand ratio
        # minloss = 332.48316225 # fixed expand ratio
        learning_rate = 1e-06
        net = MobileNetV2(image_channel=5).cuda()
        inputsize = 224
    elif modeltype == "shufflenet":
        modelname = "final-aug.t7"
        batchsize = 256 #original is 128
        minloss = 396.84708708 # change expand ratio
        learning_rate = 1e-05
        net = ShuffleNetV2().cuda()
        inputsize = 224

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.backends.cudnn.enabled = True
    gpus = [0,1]
    print("GPU NUM: %d"%(torch.cuda.device_count()))


    logname = modeltype+'-log.txt'

    if not args.retrain:
        #load pretrain model
        net = torch.load('./models/%s/%s'%(modeltype,modelname)).cuda()
        #net = torch.load('./models/%s/%s'%(modeltype,modelname)).cuda(device_id=gpus[0])

    net = net.train()

    ROOT_DIR = "./pose_dataset/mpii" # root dir to the dataset
    PATH_PREFIX = './models/{}/'.format(modeltype) # path to save the model

    train_dataset = DatasetFactory.get_train_dataset(modeltype, inputsize)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize,
                            shuffle=False, num_workers = num_threads)


    test_dataset = DatasetFactory.get_test_dataset(modeltype, inputsize)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize,
                            shuffle=False, num_workers = num_threads)


    criterion = nn.MSELoss().cuda()
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


    def mse_loss(input, target):
        return torch.sum(torch.pow(input - target,2)) / input.nelement()

    train_loss_all = []
    valid_loss_all = []
    k=0

    for epoch in range(500):  # loop over the dataset multiple times
        
        train_loss_epoch = []
        for i, data in enumerate(train_dataloader):
            # training
            images, poses = data['image'], data['pose']
            images, poses = Variable(images.cuda()), Variable(poses.cuda())
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, poses)
            loss.backward()
            optimizer.step()

            train_loss_epoch.append(loss.data[0])

        if epoch%2==0:
            valid_loss_epoch = []
            for i_batch, sample_batched in enumerate(test_dataloader):
                # calculate the valid loss
                net_forward = net
                images = sample_batched['image'].cuda()
                poses = sample_batched['pose'].cuda()
                outputs = net_forward(Variable(images, volatile=True))
                valid_loss_epoch.append(mse_loss(outputs.data,poses))

            if k==0:
                # save the model
                minloss = np.mean(np.array(valid_loss_epoch))
                checkpoint_file = PATH_PREFIX + modelname
                torch.save(net, checkpoint_file)
                print('==> checkpoint model saving to %s'%checkpoint_file)

            k=1

            if np.mean(np.array(valid_loss_epoch)) < minloss:
                # save the model
                minloss = np.mean(np.array(valid_loss_epoch))
                checkpoint_file = PATH_PREFIX + modelname
                torch.save(net, checkpoint_file)
                print('==> checkpoint model saving to %s'%checkpoint_file)

            print('[epoch %d] train loss: %.8f, valid loss: %.8f' %
            (epoch + 1, np.mean(np.array(train_loss_epoch)), np.mean(np.array(valid_loss_epoch))))
            # write the log of the training process
            if not os.path.exists(PATH_PREFIX):
                os.makedirs(PATH_PREFIX)
            with open(PATH_PREFIX+logname, 'a+') as file_output:
                file_output.write('[epoch %d] train loss: %.8f, valid loss: %.8f\n' %
                (epoch + 1, np.mean(np.array(train_loss_epoch)), np.mean(np.array(valid_loss_epoch))))
                file_output.flush()

    checkpoint_file = PATH_PREFIX + 'final/' + modelname
    torch.save(net, checkpoint_file)
    print('==> checkpoint model saving to %s' % checkpoint_file)
                
    print('Finished Training')
