# coding: utf-8
'''
File: eval.py
Project: MobilePose
File Created: Thursday, 8th March 2018 1:54:07 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Thursday, 8th March 2018 3:01:51 pm
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from math import ceil

import argparse
import time

#import os
from dataloader import *
from coco_utils import *
from networks import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from dataset_factory import DatasetFactory

gpus = [0,1]
os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.backends.cudnn.enabled = True
print(torch.cuda.device_count())

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='MobilePose Demo')
    parser.add_argument('--model', type=str, default="resnet")
    args = parser.parse_args()
    modeltype = args.model

    # user defined parameters
    filename = "final-aug.t7"
    num_threads = 10

    PATH_PREFIX = "./results/{}".format(modeltype)
    full_name="./models/{}/{}".format(modeltype, filename)

    ROOT_DIR = "../deeppose_tf/datasets/mpii"
    
    if modeltype == 'resnet':
        full_name = "./models/demo/resnet18_227x227.t7" # Rescale Expansion ToTensor
        input_size = 227
      
        test_dataset = DatasetFactory.get_test_dataset(modeltype, input_size)

    elif modeltype == 'mobilenet':
        full_name = "./models/demo/mobilenetv2_224x224-robust.t7" # Wrap Expansion ToTensor
        input_size = 224
 
        test_dataset = DatasetFactory.get_test_dataset(modeltype, input_size)

    elif modeltype == 'shufflenet':
        full_name = "./models/demo/shufflenetv2_224x224.t7" # Wrap Expansion ToTensor
        input_size = 224
 
        test_dataset = DatasetFactory.get_test_dataset(modeltype, input_size)
        
    print("Loading testing dataset, wait...")

    test_dataset_size = len(test_dataset)

    test_dataloader = DataLoader(test_dataset, batch_size=test_dataset_size,
                            shuffle=False, num_workers = num_threads)

    # get all test data
    all_test_data = {}
    for i_batch, sample_batched in enumerate(tqdm(test_dataloader)):
        all_test_data = sample_batched
        
    def eval_coco(net_path, result_gt_json_path, result_pred_json_path):
        """
        Example:
        eval_coco('/home/yuliang/code/PoseFlow/checkpoint140.t7', 
        'result-gt-json.txt', 'result-pred-json.txt')
        """
        # gpu mode
        #net = Net().cuda()
        #net = torch.load(net_path).cuda()
        #net.eval()

        # cpu mode
        net = Net()
        net = torch.load(net_path, map_location=lambda storage, loc: storage)

        ## generate groundtruth json
        total_size = len(all_test_data['image'])
        all_coco_images_arr = [] 
        all_coco_annotations_arr = []
        transform_to_coco_gt(all_test_data['pose'], all_coco_images_arr, all_coco_annotations_arr)
        coco = CocoData(all_coco_images_arr, all_coco_annotations_arr)
        coco_str =  coco.dumps()
        result_gt_json = float2int(coco_str)

        # save ground truth json to file
        dirname = os.path.dirname(result_gt_json_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        f = open(result_gt_json_path, "w")
        print("==> write" + result_gt_json_path)
        f.write(result_gt_json)
        f.close()

        # generate predictioin json
        total_size = len(all_test_data['image'])
        all_coco_pred_annotations_arr = [] 
        
        bs = 100 # batchsize

        for i in tqdm(range(1, int(ceil(total_size / float(bs) + 1)))):
            sample_data = {}

            # gpu mode
            #sample_data['image'] = all_test_data['image'][bs * (i - 1) : min(bs * i, total_size)].cuda()
            # cpu mode
            sample_data['image'] = all_test_data['image'][100 * (i - 1) : min(100 * i, total_size)]
            t0 = time.time()
            output = net(Variable(sample_data['image'],volatile=True))  #FPS is calculated from this function
            print('FPS is %f'%(1.0/((time.time()-t0)/len(sample_data['image']))))

            transform_to_coco_pred(output, all_coco_pred_annotations_arr, bs * (i - 1))

        all_coco_pred_annotations_arr = [item._asdict() for item in all_coco_pred_annotations_arr]
        result_pred_json = json.dumps(all_coco_pred_annotations_arr, cls=MyEncoder)
        result_pred_json = float2int(result_pred_json)

        # save result predict json to file
        dirname = os.path.dirname(result_pred_json_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        f = open(result_pred_json_path, "w")
        print("==> save " + result_pred_json_path)
        f.write(result_pred_json)
        f.close()



    eval_coco(full_name, os.path.join(PATH_PREFIX, 'result-gt-json.txt'), os.path.join(PATH_PREFIX, 'result-pred-json.txt'))

    # evaluation
    annType = ['segm','bbox','keypoints']
    annType = annType[2]
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

    print('Running demo for *%s* results.'%(annType))

    annFile = os.path.join(PATH_PREFIX, "result-gt-json.txt")
    cocoGt=COCO(annFile)
    resFile = os.path.join(PATH_PREFIX,"result-pred-json.txt")
    cocoDt=cocoGt.loadRes(resFile)
    imgIds=sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
