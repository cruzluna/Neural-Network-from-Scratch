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
        
        super().__init__(tanh, tanh_prime)
        

# class Sigmoid(Activation):#child class of Activation
#     def __init__(self):
        
#         # S(x) = 1/(1+e^-x)
#         def sigmoid(x):
#             return 1.0 / (1 +np.exp(-x))
        
#         # S'(x) = S(x) * (1-S(z))
#         def sigmoid_prime(x):
#             return sigmoid(x) * (1-sigmoid(x))
#         super().__init__(sigmoid,sigmoid_prime)
            
