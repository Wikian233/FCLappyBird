import feedforward_closedloop_learning as fcl
import numpy as np
import matplotlib.pyplot as plt
from config import *
import random
import os
import glob
from prepocessing import *
import cv2

layers = NEURONSPERLAYER_FCL3
class FCLNet(fcl.FeedforwardClosedloopLearning):
    def __init__(self, learningRate):
        super().__init__(130 * 50, layers)
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
        self.rawInputs = np.zeros((130, 50))
        self.input_buff = np.zeros((130 * 50))
        self.netErr = np.zeros(layers[0])
        # self.netOutput = np.zeros(layers[-1])
        self.netOutput = None

    def train(self, input, err):
        self.rawInputs = cv2.Canny(input, threshold1 = 200, threshold2=300)
        self.input_buff[:] = np.ndarray.flatten(self.rawInputs)

        print(self.input_buff.shape)
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