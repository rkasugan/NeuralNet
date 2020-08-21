import numpy as np
from NeuralNet import NeuralNet
from scipy.io import loadmat

# Methods for data processing
def print_terminal(digit_array):
    # Assuming array of 28x28 pixels
    for i in range(28):
        for j in range(28):
            print(f'{str(digit_array[28*i+j]): <4}', end = "")
        print()
def produce_arrays_from_labels(labels):
    # 0 -> [1,0,0,0,0,0,0,0,0,0], 5 -> [0,0,0,0,0,1,0,0,0,0], etc
    def array_from_label(x):
        arr = np.zeros(10)
        arr[x] = 1
        return arr
    result = [None]*len(labels)
    for i in range(len(labels)):
        result[i] = array_from_label(int(labels[i]))
    return result
def get_label_from_array(arr):
    # [1,0,0,0,0,0,0,0,0,0] -> 0, [0,0,0,0,0,1,0,0,0,0] -> 5, etc
    result = 0
    for i in range(1, len(arr)):
        if arr[i] > arr[result]:
            result = i
    return result

# Load training data from .mat file, downloaded from https://www.kaggle.com/avnishnish/mnist-original
mnist = loadmat("mnist-original.mat")
mnist_data = mnist["data"].T.astype('int32')
mnist_data = np.floor_divide(mnist_data,25)
mnist_label = mnist["label"][0]

# Shuffle data (originally ordered by value) and produce expected output
shuffler = np.random.permutation(len(mnist_data))
raw_data = np.array(mnist_data[shuffler])
raw_labels = np.array(mnist_label[shuffler])
raw_output = np.array(produce_arrays_from_labels(raw_labels))

# Train on the first 60,000 samples
training_data = raw_data[:60000]
training_labels = raw_labels[:60000]
training_output = raw_output[:60000]

net = NeuralNet(np.array([784,16,16,10])) 
# net = NeuralNet(np.array([784,16,16,10]), True, "weights.npy", "biases.npy") - Load neural net with preset weights and biases

net.train(training_data, training_output, 300, .1)

# Print results of testing a single random sample
i = 65000
print_terminal(raw_data[i])
print("Expected Output: " + str(int(raw_labels[i])))
output = net.forward_prop(raw_data[i])
print("Actual Output: " + str(get_label_from_array(output.ravel())))
print("Layer Output:\n" + str(output))

# Save the weights and biases to a .npy file for later use
# net.write_data("weights.npy", "biases.npy")