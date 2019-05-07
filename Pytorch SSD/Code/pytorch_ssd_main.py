import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# function that will do the detection
def detect(frame, net, transform):
    # define a detect function that will take as inputs, a frame, a ssd neural network,
    # and a transformation to be applied on the images, and that will return the frame with the detector rectangle.
    height, width = frame.shape[:2] # get the height and the width of the frame
    frame_t = transform(frame)[0] # apply the transformation to our frame
    x = torch.from_numpy(frame_t).permute(2,0,1) # # convert frame into a torch tensor
    x = Variable(x.unsqueeze(0)) # add a fake dimension corresponding to the batch
    y = net(x) # feed the neural network ssd with the image and we get the utput y
    detections = y.data # create the detection tensor contained in the output y
    scale = torch.Tensor([width, height, width, height]) # We create a tensor object of dimensions [width, height, width, height].

    for i in range(detections.size(1)): # for every class:
        j = 0 # initialize the loop variable j that will correspond to the occurrences of the class
        while detections[0, i, j, 0] >= 0.6: # we take into account all the occurences j of the class i that have a matching score larger than 0.6
            pt = (detections[0, i, j, 1:] * scale).numpy() # get the corrdinates of the points at the upper left and the lower right of the detector rectangle
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) # We draw a rectangle around the detected object.
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) # We put the label of the class right above the rectangle.
            j += 1 # We increment j to get to the next occurrence.
    return frame # we return the original frame_with the detector rectangle and the label around the detected object

# Creating the SSD neural network
net = build_ssd('test') # create an object that is our neural network SSD
# we get the weights of nerual network from another one that is pretrained
net.load_state_dict(torch.load('C:/Users/Kuasn/OneDrive/10 - Scripts Treino/Computer_Vision/ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

# creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # create an object of the BaseTransform, a class that will do the required transformation so that the image can be the input of the neural network

# Doing some Object Detection on a video
reader = imageio.get_reader('C:/Users/Kuasn/OneDrive/10 - Scripts Treino/Computer_Vision/funny_dog.mp4') # We open the video.
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
writer = imageio.get_writer('C:/Users/Kuasn/OneDrive/10 - Scripts Treino/Computer_Vision/output.mp4', fps = fps) # We create an output video with this same fps frequence.
for i, frame in enumerate(reader): # We iterate on the frames of the output video:
    frame = detect(frame, net.eval(), transform) # We call our detect function (defined above) to detect the object on the frame.
    writer.append_data(frame) # We add the next frame in the output video.
    print(i) # We print the number of the processed frame.
writer.close() # We close the process that handles the creation of the output video.
