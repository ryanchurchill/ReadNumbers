import mnist_loader
from network import Network
import calendar
import time
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = Network([784, 30, 10])

start_time = int(round(time.time() * 1000))

# [net.feedforward(x)
# for (x, y) in test_data]

#sigmoid: 14k milliseconds or 707k/sec
for i in xrange(10000000):
    sigmoid(i)

end_time = int(round(time.time() * 1000))
duration = end_time - start_time
average_duration = (10000000 / duration) * 1000

print "duration: {0} milliseconds".format(duration)
print "average rate {0} / second".format(average_duration)


