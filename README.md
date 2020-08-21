# NeuralNet
This is a simple neural net class built based on 3Blue1Brown's excellent video series on Deep Learning, which can be found at https://youtu.be/aircAruvnKk. It uses stochastic gradient descent to train on individual trials with a standard sigmoid activation function and biases for each node. The network can be initialized with any provided layer structure, and can write/load its weights and biases to/from a file for convenience.

# Data
weights.npy and biases.npy represent the exported weights and biases numpy arrays. "Training" the network involves modifying these arrays, so saving these can save time on larger projects.

The provided netDriver.py driver trains the network on a collection of 70,000 hand-written digits from MNIST, which was downloaded from https://www.kaggle.com/avnishnish/mnist-original. This file must be unpacked before use, which can be done by running "tar zxvf mat_file.tgz".

# Archive
These files represent previous iterations of the network, which were included to make the process behind the neural net easier to understand. They are in chronological order, showing the development of the network:

neuralNet1.py - Network specifically built for the structure [2,2,2], with the backpropagation math being done entirely explicitly. Trains on a combination of OR and XOR

neuralNet2.py - Network specifically built for the structure [2,2,2], but now with the math being done in matrices making the code faster and easier to read.

neuralNet3.py - Generalized neural net, which can support any provided structure.

neuralNet4.py - Same generalized neural net as neuralNet3.py, this time attempting to train on the function .5*sin(x)+.5.

# Notes
These files are here out of sentiment more than anything else, but can be enlightening if trying to understand the math behind the network. 

GeneralNotes displays the core formula for calculating the partial derivative of cost with respect to an individual weight or individual bias, which is the heart of back propagation. Also included are the stochastic gradient descent algorithm and the structure of the weight and bias matrices.

BackpropCalculus displays the calculation of the previously described partial derivative for each individual weight and bias in the provided network. This took a long time. The math here lines up with that of neuralNet1.py.

MatrificationGeneralization shows how I went from calculating each individual derivative to doing one calculation for weights and one for biases per layer by stuffing everything into matrices. This math here lines up with that of neuralNet2.py. At the bottom there are some notes on Generalization, or allowing the code to handle a network of any size (instead of just [2,2,2]). The insights here got me to neuralNet3.py; after that, it was just a matter of splitting the code into a distinct class description and driver.
