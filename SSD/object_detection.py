import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a function that will do the detections
# frame by frame detection
def detect(frame, net, transform):
    """
    Sets the database connection for the destination
    :param frame: where function is applied
    :param net: ssd neural network
    :param transform: transform image
    :returns: same frame with rectangle detection and label on rectangle
    """

    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x) # output od neural network
    detections = y.data # extract information we need (torch tensors and gradient)
    scale = torch.Tensor([width, height, width, height])
    
