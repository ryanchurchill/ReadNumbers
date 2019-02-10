import mnist_loader
from network import Network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 30, 10])
epochs = 30
mini_batch_size = 10
eta = 3.0 # learning rate
net.sgd(training_data, epochs, mini_batch_size, eta, test_data)