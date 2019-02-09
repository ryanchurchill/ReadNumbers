import numpy as np

class Network(object):

    def __init__(self, sizes):
        print("Initializing network of")
        print(sizes)
        self.num_layers = len(sizes)

        # list of the size of each layer
        # e.g. [input_layer_size, hidden_layer_size, output_layer_size]
        self.sizes = sizes

        # list of numpy matrices storing the biases at each layer except the first
        # [[w1, w2, w3], [w_output]]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        print(self.biases)

        # list of numpy matrices storing the weights connecting the layers
        # e.g. weights[1] is a Numpy matrix storing the weights connecting
        # the second and third layers of neurons
        # It's a matrix such that wjk is the weight for the connection between
        # the kth neuron in the second layer, and the jth neuron in the third layer.
        self.weights = [np.random.randn(y, x)
                       for x, y in zip(sizes[:-1], sizes[1:])]
        # print(self.weights[0])

    # Note that when the input z is a vector or Numpy array, Numpy automatically
    # applies the function sigmoid elementwise, that is, in vectorized form.
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    # a is the input for the network
    # assume a is an (n, 1) numpy ndarray, where n is the number of inputs
    def feedforward(self, a):
        # Return the output of the network if "a" is input.
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
