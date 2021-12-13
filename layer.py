from abc import ABC, abstractclassmethod

import numpy as np 
import pandas as pd


#Load dataset
data = pd.read_csv("labels.csv")
#print(data.head())

#Implementation Design

"""
Each layer uses forward and backward propagation. 
Forward propagation: input is the output of previous layer
Backward  Propagation: chain rule,dE/dW = (dE/dY)*(dY/dW) 
"""

#Create the base layer

# Abstract base class all layers inherit from
class Layer(ABC): 
    def __init__(self):
        self.input = None
        self.output = None
    
    #given input X, calculates output Y
    @abstractclassmethod
    def forward_propagation(self,input):
        #error is here, it is calling the parent class and not the child class
        pass
    
    
    #calculates the dE/dx for a given dE/dy (and update parameters if needed)
    # chain rule: dE/dW = (dE/dY)*(dY/dW)
    @abstractclassmethod
    def backward_propagation(self,output_error, learning_rate):
        #update parameters and return input gradient
        pass
    
