import numpy as np
from layer import Layer
from activation import Activation


class Tanh(Activation): #inherit from activations
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        
        def tanh_prime(x):
            #d(tanh)/dx = 1-(tanh(x))^2
            return 1 - np.power(tanh(x),2)
        
        super(Tanh,self).__init__(tanh, tanh_prime)
        


