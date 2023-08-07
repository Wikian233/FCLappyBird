import feedforward_closedloop_learning as fcl
import numpy as np
import matplotlib.pyplot as plt
from config import *
import random

layers = NEURONSPERLAYER_FCL
class FCLNet(fcl.FeedforwardClosedloopLearning):
    def __init__(self, learningRate):
        super().__init__(layers[0], layers)
        self.initWeights(1., fcl.FCLNeuron.MAX_OUTPUT_RANDOM)
        print ("Initialised weights")
        for i in range(len(layers)):
            print ("hidden ", i, ": ", layers[i])
        print("learning rate: ", learningRate)
        
        self.learningRate = learningRate

        self.setBias(1)
        self.setMomentum(0.5)
        self.setActivationFunction(fcl.FCLNeuron.TANH)
        self.setLearningRate(learningRate)
        self.seedRandom(np.random.randint(low=0, high=999999))
        self.setLearningRateDiscountFactor(1)

        self.input_buff = np.zeros((4))
        self.netErr = np.zeros(layers[0])
        # self.netOutput = np.zeros(layers[-1])
        self.netOutput = 0

    def train(self, Input, err):
        self.input_buff[:] = Input
        self.netErr[:] = err

        self.doStep(self.input_buff, self.netErr)
        # for i in range(len(self.netOutput)):
        #     self.netOutput[i] = self.getOutput(i)
        self.netOutput = np.tanh(self.getOutput(0)) + np.tanh(self.getOutput(1)) + np.tanh(self.getOutput(2)) #+ np.tanh(self.getOutput(3))#+ np.tanh(err)
        # print(self.getOutput(0), self.getOutput(1), self.getOutput(2))
        print(self.input_buff, self.netErr, self.netOutput, " , ",self.netOutput - err, self.getLayer(1).getNeuron(0).getWeight(0), self.getLayer(2).getNeuron(0).getWeight(0))
        return self.netOutput


