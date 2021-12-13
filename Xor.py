from dense import Dense
from Activations import Tanh
from Losses import mse, mse_prime
from network import train, predict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Creating datasets (XOR table)
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2,3), #input = 2 output = 3
    Tanh(),
    Dense(3,1), #input = 3 output = 1
    Tanh()
]

# train
train(network, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1)

# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()

# epochs = 10000
# learning_rate = 0.1
# #train
# for e in range(epochs):
#     error = 0
#     for x,y in zip(X,Y):
#         #forward propagation
#         output = x
#         for layer in network: 
#             #output = input of next layer
#             output = layer.forward_propagation(output)
            
#         #error
#         error += mse(y,output)
        
#         #backward propagation
#         grad = mse_prime(y,output)
#         for layer in reversed(network):
#             grad = layer.backward_propagation(grad,learning_rate)
#     error /= len(X)
#     print('%d/%d, error %f' % (e + 1, epochs, error))
        