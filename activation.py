import numpy as np
from layer import Layer

       
#Create Activation Layer
class Activation(Layer): #Child Class of Layer
    def __init__(self, activation, activation_prime):
        self.activation = activation 
        self.activation_prime = activation_prime
    
    def forward_propagation(self, input):
        self.input = input
        #Y = f(X)
        return self.activation(self.input)

    def backward_propagation(self, output_gradient, learning_rate):
        #dE/dX = dE/dY * f'(X)
        return np.multiply(output_gradient, self.activation_prime(self.input))