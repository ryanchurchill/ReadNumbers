from network import Network
import numpy as np


# training_data = [(np.matrix('1;2'), np.matrix('3'))]
training_data = [(np.array([[1], [2]]), np.array([[0], [1]])), (np.array([[2], [1]]), np.array([[1], [0]]))]
# print training_data[0][1]

epochs = 1
mini_batch_size = 2
eta = 3.0 # learning rate

net = Network([2, 3, 2])
net.sgd(training_data, epochs, mini_batch_size, eta)

# sizes = [2, 3, 1]
# print(sizes)
# print(sizes[:-1])
# print(sizes[1:])
# print(zip(sizes[:-1], sizes[1:]))