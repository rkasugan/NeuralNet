# NeuralNet
This is a simple neural net class built based on 3Blue1Brown's excellent video series on Deep Learning which can be found at https://youtu.be/aircAruvnKk. It uses stochastic gradient descent to train on individual trials with a standard sigmoid activation function and biases for each node. The network can be initialized with any provided layer structure, and can write/load its weights and biases to a file for convenience.

# Data
weights.npy and biases.npy represent the exported weights and biases numpy arrays. "Training" the network involves modifying these arrays, so saving these can save time on larger projects.

The provided netDriver.py driver trains the network on a collection of 70,000 hand-written digits from MNIST, which was downloaded from https://www.kaggle.com/avnishnish/mnist-original. This file must be unpacked before use, which can be done by running "tar zxvf mat_file.tgz".

# Archive
These files represent previous iterations of the network, which were included to make the process behind the neural net easier to understand. They are in chronological order, showing the development of the network:

neuralNet1.py - Network specifically built for the structure [2,2,2], with the backpropagation math being done entirely explicitly. Trains on a combination of OR and XOR

neuralNet2.py - Network specifically built for the structure [2,2,2], but now with the math being done in matrices making the code faster and easier to read.

neuralNet3.py - Generalized neural net, which can support any provided structure.

neuralNet4.py - Same generalized neural net as neuralNet3.py, this time attempting to train on the function .5*sin(x)+.5.
