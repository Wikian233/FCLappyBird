import feedforward_closedloop_learning as fcl
import numpy as np
import matplotlib.pyplot as plt
from config import *
import random
import os
import glob
# from prepocessing import *
import cv2
import torch
import torch.nn as nn

conv_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
bn1 = nn.BatchNorm2d(16)
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2) 
conv_layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
bn2 = nn.BatchNorm2d(32)
conv_layer3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
bn3 = nn.BatchNorm2d(16)
conv_layer4 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1)
bn4 = nn.BatchNorm2d(4)

if torch.cuda.is_available():
    conv_layer = conv_layer.cuda()
    pool_layer = pool_layer.cuda()
    conv_layer2 = conv_layer2.cuda()
    conv_layer3 = conv_layer3.cuda()
    conv_layer4 = conv_layer4.cuda()
    bn1 = bn1.cuda()
    bn2 = bn2.cuda()
    bn3 = bn3.cuda()
    bn4 = bn4.cuda()



layers = [600, 75, 15, 4]
class FCLNet(fcl.FeedforwardClosedloopLearning):
    def __init__(self, learningRate):
        super().__init__(600, layers)
        self.initWeights(1., fcl.FCLNeuron.MAX_OUTPUT_RANDOM)
        print ("Initialised weights")
        for i in range(len(layers)):
            print ("hidden ", i, ": ", layers[i])
        print("learning rate: ", learningRate)
        
        self.learningRate = learningRate

        self.setBias(1)
        self.setMomentum(0.5)
        self.setActivationFunction(fcl.FCLNeuron.RELU)
        self.setLearningRate(learningRate)
        self.seedRandom(np.random.randint(low=0, high=999999))
        self.setLearningRateDiscountFactor(1)
        self.input_buff = np.zeros((600))
        self.netErr = np.zeros(layers[0])
        # self.netOutput = np.zeros(layers[-1])
        self.netOutput = None

    def train(self, input, err):
        # print(input.shape)
        self.input_buff[:] = self.process_img(input)

        # print(self.input_buff.shape)
        self.netErr[:] = err
        
        self.doStep(self.input_buff, self.netErr)
        # for i in range(len(self.netOutput)):
        #     self.netOutput[i] = self.getOutput(i)
        self.netOutput = np.tanh(self.getOutput(0)) + np.tanh(self.getOutput(1)) + np.tanh(self.getOutput(2)) + np.tanh(self.getOutput(3))#+ np.tanh(err)
        # print(self.getOutput(0), self.getOutput(1), self.getOutput(2))
        print(self.input_buff, self.netErr, self.netOutput, " , ",self.netOutput - err, self.getLayer(1).getNeuron(0).getWeight(0), self.getLayer(2).getNeuron(0).getWeight(0))
        return self.netOutput
    
    def loadBestModel(self):
        # to get all the files in the current directory which fufill the condition
        filenames = glob.glob("Models/fcl2-*.txt")

        if not filenames:
            print("file not found")
        else:  
        # find the max value in the filename
         max_value = max(int(f.split('-')[-1].split('.')[0]) for f in filenames)
    
        # get the filename with the max value
        max_filename = "Models/fcl2-" + str(max_value) + ".txt"

        if os.path.isfile(max_filename):
            print(f"file {max_filename} exists")
            self.loadModel(max_filename)
            print(f"Model loaded {max_filename}")
        else:
            print(f"file {max_filename} not exists")

        # delete all the files except the one with the max value
        for filename in filenames:
            value = int(filename.split('-')[-1].split('.')[0])
            if value != max_value:
                os.remove(filename)
                print(f"files deleted {filename}")

    def process_img(self, image):
        image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tensor_frame = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # 为batch和channel增加一个维度

        if torch.cuda.is_available():
            tensor_frame = tensor_frame.cuda()

        tensor_frame = conv_layer(tensor_frame)
        tensor_frame = bn1(tensor_frame)
        tensor_frame = nn.functional.relu(tensor_frame)
        tensor_frame = pool_layer(tensor_frame) 

        tensor_frame = conv_layer2(tensor_frame)
        tensor_frame = bn2(tensor_frame)
        tensor_frame = nn.functional.relu(tensor_frame)
        tensor_frame = pool_layer(tensor_frame)

        tensor_frame = conv_layer3(tensor_frame)
        tensor_frame = bn3(tensor_frame)
        tensor_frame = nn.functional.relu(tensor_frame)
        tensor_frame = pool_layer(tensor_frame)

        tensor_frame = conv_layer4(tensor_frame)
        tensor_frame = bn4(tensor_frame)
        tensor_frame = nn.functional.relu(tensor_frame)
        tensor_frame = pool_layer(tensor_frame)

        tensor_frame = tensor_frame.cpu().detach().numpy()
        img_to_show = tensor_frame[0, 2, :, :]
        img_to_show = np.ndarray.flatten(img_to_show)
        
        return img_to_show