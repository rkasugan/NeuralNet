import numpy as np

# Initialize weights and biases arrays based on provided structure of net
# s[i] is the number of nodes in the ith layer of the network, and len(s) is the number of layers in the network
# weights[i] is itself the matrix of edge weights for the ith layer
    # weights[i][j][k] is the edge weight between the jth node of the ith layer and the kth node of the (i-1)th layer
# biases[i] is a list of the biases for the nodes of the ith layer
    # biases[i][j] is the bias for the jth node of the ith layer
s = [2,2,2]
biases = [None]*len(s)
weights = [None]*len(s)
for i in range(1,len(s)):
    biases[i] = np.random.rand(s[i], 1)*2-1
    weights[i] = np.random.rand(s[i], s[i-1])*2-1

def sigmoid(x):
    return 1/(1+np.exp(-x))
def deriv_sigmoid(x):
    return np.exp(x)/(np.exp(x)+1)**2

# Forward propagate input array x through the network
def forward_prop(x):
    layer = x
    for i in range(1, len(s)):
        layer = sigmoid(np.dot(weights[i], layer)+biases[i])
    return layer

# Run a single training example based on input array x and expected output y
def train(x,y):
    # Manually forward propagate through the network, keeping track of values
        # layers[i]["z"] and layers[i]["a"] are the lists of layer i's sums and activations, respectively
    layers = [dict() for i in range(len(s))] 
    layers[0]["a"] = x
    for i in range(1, len(s)):
        layers[i]["z"] = np.dot(weights[i], layers[i-1]["a"])+biases[i]
        layers[i]["a"] = sigmoid(layers[i]["z"])

    # Calculate the partial derivates of total cost with respect to each weight and bias
        # derivatives[i]["b"] and derivatives[i]["w"] are the derivatives for the ith layer's biases and weights, respectively
        # The bias and weight derivative matrices mirror the structure of the actual biases[] and weights[] matrices
    derivatives = [dict() for i in range(len(s))] 
    for i in reversed(range(1, len(s))):
        if i == len(s)-1:
            derivatives[i]["b"] = deriv_sigmoid(layers[i]["z"])*(2*(layers[i]["a"]-y))
        else:
            derivatives[i]["b"] = deriv_sigmoid(layers[i]["z"])*np.dot(derivatives[i+1]["b"].T, weights[i+1]).T
        derivatives[i]["w"] = np.dot(derivatives[i]["b"], layers[i-1]["a"].T)

    # Update the weights and biases using the stochastic gradient descent algorithm
    learning_rate = .01
    for i in range(1, len(s)):
        biases[i] -= learning_rate * derivatives[i]["b"]
        weights[i] -= learning_rate * derivatives[i]["w"]

# Network takes in two inputs and returns two outputs: the results of the OR and XOR functions
training_data = np.array([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
])
training_output = np.array([
  [0, 0],
  [1, 1],
  [1, 1],
  [1, 0],
])

# Train
for epoch in range(60000):
    for trial in range(len(training_data)):
        train(training_data[trial].reshape(len(training_data[0]),1), training_output[trial].reshape(len(training_output[0]),1))

# Print the results
for i in range(4):
    result = forward_prop(training_data[i].reshape(2,1))
    result_OR = str(result[0])
    result_XOR = str(result[1])
    print(str(training_data[i]) + " -> OR: " + result_OR + ", XOR: " + result_XOR)