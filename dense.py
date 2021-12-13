import numpy as np
from layer import Layer

#Create Dense Layer
class Dense(Layer):#Dense is a child class of Layer
    # Constructor initializes the matrices
    def __init__(self,input_size,output_size):
        # Initialize the matrices
        self.weights = np.random.randn(output_size,input_size) #.randn(m X n)
        self.bias = np.random.randn(output_size,1) # b vector, size = j X 1
       
    
      
    def forward_propatgation(self, input): 
        self.input = input 
        #Matrix multiplication
        return np.dot(self.weights, self.input) + self.bias
        
    
    def backward_propagation(self, output_gradient, learning_rate):
        # dE/dW = dE/dY * X^T
        weights_gradient = np.dot(output_gradient,self.input.T) 
         #dE/dX = W^T * dE/dY
        input_gradient = np.dot(self.weights.T, output_gradient)
        #Update parameters -> weights & Bias  
        #subtract b/c derivatibes point in direction of steepest ascent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        
       
        
        return  input_gradient
    