import random
import numpy as np
import calendar
import time

class Network(object):
    def __init__(self, sizes):
        print("Initializing network of")
        print(sizes)
        self.num_layers = len(sizes)

        # list of the size of each layer
        # e.g. [input_layer_size, hidden_layer_size, output_layer_size]
        self.sizes = sizes

        # list of numpy matrices storing the biases at each layer except the first
        # [[[w1], [w2], [w3]], [[w_output]]]
        # for each size except the first, return an array of size Y
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]


        # list of numpy matrices storing the weights connecting the layers
        # e.g. weights[1] is a Numpy matrix storing the weights connecting
        # the second and third layers of neurons
        # It's a matrix such that wjk is the weight for the connection between
        # the kth neuron in the second layer, and the jth neuron in the third layer.
        # ex: sizes: [2, 3, 1]. The zip returns [(2, 3), (3, 1)]
        self.weights = [np.random.randn(y, x)
                       for x, y in zip(sizes[:-1], sizes[1:])]


        # self.printMe()

    def printMe(self):
        # print('biases')
        # print(self.biases)
        #
        # print('weights')
        # print(self.weights)

        np.set_printoptions(threshold=100000, linewidth=1000000)

        print('biases[0]')
        print(self.biases[0])
        print('biases[1]')
        print(self.biases[1])


        print('weights[0]')
        print(self.weights[0])

        print('weights[1]')
        print(self.weights[1])

    # Note that when the input z is a vector or Numpy array, Numpy automatically
    # applies the function sigmoid elementwise, that is, in vectorized form.
    #
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return Network.sigmoid(z)*(1-Network.sigmoid(z))

    # a is the input for the network
    # assume a is an (n, 1) numpy ndarray, where n is the number of inputs
    def feedforward(self, a):
        # Return the output of the network if "a" is input.
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)

        if test_data:
            print "Epoch -1: {0} / {1}".format(
                self.evaluate(test_data), n_test)

        for j in xrange(epochs):
            random.shuffle(training_data)

            start_time = calendar.timegm(time.gmtime())

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
                # for k in xrange(0, mini_batch_size*1000, mini_batch_size)]
            for mini_batch in mini_batches:
                # print("a")
                # print len(mini_batch)
                # print len(mini_batches)
                self.update_mini_batch(mini_batch, eta)

            end_time = calendar.timegm(time.gmtime())
            duration = end_time - start_time

            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

            print "epoch finished in {0} seconds".format(duration)

            # self.printMe()

    def update_mini_batch(self, mini_batch, eta):
        #ryan debugging
        # print "mini_batch"
        # print mini_batch

        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            #ryan debugging
            # print "delta_nabla_b:"
            # print delta_nabla_b
            # print "delta_nabla_w:"
            # print delta_nabla_w


            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            #ryan debugging
            # print "nabla_b:"
            # print nabla_b
            # print "nabla_w:"
            # print nabla_w

            # print "eta:"
            # print eta
            # print "len(mini_batch):"
            # print len(mini_batch)

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # ryan testing
        # print "final layer of activations"
        # print activations[2]

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                self.sigmoid_prime(zs[-1])

        # ryan debugging. I believe delta is the value of BP1: the error of the output layer
        # print "delta at output layer:"
        # print delta

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

            #ryan debugging
            # print "delta at next layer moving left:"
            # print delta

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)