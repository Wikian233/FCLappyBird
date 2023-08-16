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
import torch.optim as optim

class FeatureCNN(nn.Module):
    def __init__(self):
        super(FeatureCNN, self).__init__()
        
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        
        self.conv_layer3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(4)
        
        self.conv_layer4 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.pool_layer(x) 

        x = self.conv_layer2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.pool_layer(x)

        x = self.conv_layer3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.pool_layer(x)

        x = self.conv_layer4(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)
        x = self.pool_layer(x)
        
        return x
    
    
model = FeatureCNN()
if torch.cuda.is_available():
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.00001)
loss_function = nn.MSELoss()



layers = [630, 240 , 4]
class FCLNet(fcl.FeedforwardClosedloopLearning):
    def __init__(self, learningRate):
        super().__init__(630, layers)
        self.initWeights(1., fcl.FCLNeuron.MAX_OUTPUT_RANDOM)
        print ("Initialised weights")
        for i in range(len(layers)):
            print ("hidden ", i, ": ", layers[i])
        print("learning rate: ", learningRate)
        
        self.learningRate = learningRate

        self.setBias(1)
        self.setMomentum(0.5)
        self.setActivationFunction(fcl.FCLNeuron.TANHLIMIT)
        self.setLearningRate(learningRate)
        self.seedRandom(np.random.randint(low=0, high=999999))
        self.setLearningRateDiscountFactor(1)
        self.input_buff = np.zeros((630))
        self.netErr = np.zeros(layers[0])
        # self.netOutput = np.zeros(layers[-1])
        self.netOutput = None

    def train(self, input, err, avg_history):
        # print(input.shape)
        print(avg_history.shape)
        self.input_buff[:] = self.process_img(input, avg_history)

        # print(self.input_buff.shape)
        self.netErr[:] = err * 500
        
        self.doStep(self.input_buff, self.netErr)

        self.netOutput = np.tanh(self.getOutput(0)) + np.tanh(self.getOutput(1)) + np.tanh(self.getOutput(2)) + np.tanh(self.getOutput(3)) #+\
        
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

    def process_img(self, image, avg_history):  

        tensor_frame = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        target_tensor = torch.from_numpy(avg_history).float().unsqueeze(0)
        target_tensor = nn.functional.interpolate(target_tensor, size=(21, 15), mode='bilinear')

        # print(target_tensor.shape)
        # print(tensor_frame.shape)

        if torch.cuda.is_available():
            tensor_frame = tensor_frame.cuda()
            target_tensor = target_tensor.cuda()

        output = model(tensor_frame)
        
        loss = loss_function(output, target_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tensor_frame = output.cpu().detach().numpy()
        
        # img = tensor_frame.squeeze(0)
        # print(img.shape)
        # plt.imshow(img[1])
        # plt.pause(0.001)
        # plt.clf()
        
        img_to_show = np.ndarray.flatten(tensor_frame)
        
        return img_to_show
            
