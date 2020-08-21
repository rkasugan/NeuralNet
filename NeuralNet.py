import numpy as np

class NeuralNet:    
    # Initialize weights and biases arrays based on provided structure of net, or load parameters if requested
    # layer_structure[i] is the number of nodes in the ith layer of the network, and num_layers is the number of layers in the network
    # weights[i] is itself the matrix of edge weights for the ith layer
        # weights[i][j][k] is the edge weight between the jth node of the ith layer and the kth node of the (i-1)th layer
    # biases[i] is a list of the biases for the nodes of the ith layer
        # biases[i][j] is the bias for the jth node of the ith layer
    def __init__(self, layer_structure, load = False, weights_file = "", biases_file = ""):
        self.layer_structure = layer_structure
        self.num_layers = len(layer_structure)
        self.biases = np.zeros(self.num_layers, dtype='object')
        self.weights = np.zeros(self.num_layers, dtype='object')
        if load:
            self.biases = np.load(biases_file, allow_pickle=True)
            self.weights = np.load(weights_file, allow_pickle=True)
        else:
            for i in range(1,self.num_layers):
                self.biases[i] = (np.random.rand(layer_structure[i], 1)*2-1).astype('float128')
                self.weights[i] = (np.random.rand(layer_structure[i], layer_structure[i-1])*2-1).astype('float128')
        
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    def deriv_sigmoid(x):
        return np.exp(x)/(np.exp(x)+1)**2
    def cost(self, input_data, expected_output):
        actual_output = expected_output.copy()
        for i in range(len(input_data)):
            actual_output[i] = self.forward_prop(input_data[i].reshape(len(input_data[i]),1)).ravel()
        return ((expected_output-actual_output)**2).mean()
    def write_data(self, weights_file, biases_file):
        np.save(weights_file, self.weights)
        np.save(biases_file, self.biases)
    
    # Forward propagate input array x through the network
    def forward_prop(self, x):
        layer = x.reshape(len(x), 1)
        for i in range(1, self.num_layers):
            layer = NeuralNet.sigmoid(np.dot(self.weights[i], layer)+self.biases[i])
        return layer

    # Trains the network over a set of training data and output
        # Total training data and output is taken in as [ [sample A inputs]  and  [ [sample A outputs]
        #                                                 [sample B inputs] ]       [sample B outputs] ]
        # However, internally each sample is converted to [ [sample A input 1]  and  [ [sample A output 1] for individual training    
        #                                                   [sample A input 2] ]       [sample A output 2] ]
    def train(self, training_data, training_output, epochs, learning_rate):
        for epoch in range(epochs):
            for trial in range(len(training_data)):
                if trial % 1000 == 0:
                    print("Trial " + str(trial) + " of epoch " + str(epoch))
                
                trial_data = training_data[trial].reshape(len(training_data[trial]), 1)
                trial_output = training_output[trial].reshape(len(training_output[trial]), 1)
                self.train_single(trial_data, trial_output, learning_rate)
            print("Epoch: " + str(epoch) + ", Cost:" + str(self.cost(training_data, training_output)))

    # Run a single training example based on input array x and expected output y
    def train_single(self, x, y, learning_rate):
        # Manually forward propagate through the network, keeping track of values
            # layers[i]["z"] and layers[i]["a"] are the lists of layer i's sums and activations, respectively
        layers = [dict() for i in range(self.num_layers)] 
        layers[0]["a"] = x
        for i in range(1, self.num_layers):
            layers[i]["z"] = np.dot(self.weights[i], layers[i-1]["a"])+self.biases[i]
            layers[i]["a"] = NeuralNet.sigmoid(layers[i]["z"])

        # Calculate the partial derivates of total cost with respect to each weight and bias
            # derivatives[i]["b"] and derivatives[i]["w"] are the derivatives for the ith layer's biases and weights, respectively
            # The bias and weight derivative matrices mirror the structure of the actual biases[] and weights[] matrices
        derivatives = [dict() for i in range(self.num_layers)] 
        for i in reversed(range(1, self.num_layers)):
            if i == self.num_layers-1:
                derivatives[i]["b"] = NeuralNet.deriv_sigmoid(layers[i]["z"])*(2*(layers[i]["a"]-y))
            else:
                derivatives[i]["b"] = NeuralNet.deriv_sigmoid(layers[i]["z"])*np.dot(derivatives[i+1]["b"].T, self.weights[i+1]).T
            derivatives[i]["w"] = np.dot(derivatives[i]["b"], layers[i-1]["a"].T)

        # Update the weights and biases using the stochastic gradient descent algorithm
        for i in range(1, self.num_layers):
            self.biases[i] -= learning_rate * derivatives[i]["b"]
            self.weights[i] -= learning_rate * derivatives[i]["w"]