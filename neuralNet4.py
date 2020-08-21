import numpy as np

# Initialize weights and biases arrays based on provided structure of net
# s[i] is the number of nodes in the ith layer of the network, and len(s) is the number of layers in the network
# weights[i] is itself the matrix of edge weights for the ith layer
    # weights[i][j][k] is the edge weight between the jth node of the ith layer and the kth node of the (i-1)th layer
# biases[i] is a list of the biases for the nodes of the ith layer
    # biases[i][j] is the bias for the jth node of the ith layer
s = [1,6,6,1]
biases = [None]*len(s)
weights = [None]*len(s)
for i in range(1,len(s)):
    biases[i] = (np.random.rand(s[i], 1)*2-1)
    weights[i] = (np.random.rand(s[i], s[i-1])*2-1)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def deriv_sigmoid(x):
    return np.exp(x)/(np.exp(x)+1)**2
def cost(input_data, expected_output):
    actual_output = expected_output.copy()
    for i in range(len(input_data)):
        actual_output[i] = forward_prop(input_data[i].reshape(1,1))
    return ((expected_output-actual_output)**2).mean()

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
    learning_rate = .2
    for i in range(1, len(s)):
        biases[i] -= learning_rate * derivatives[i]["b"]
        weights[i] -= learning_rate * derivatives[i]["w"]

# Network is trained on random values between -25 and 25, should return x -> .5*sin(x) +.5
raw_data = (np.random.rand(100000,1)*50-25)
raw_output = (.5*np.sin(raw_data[:,0]) + .5).reshape(len(raw_data),1)
training_data = raw_data[:90000] # train on first 90,000 samples
training_output = raw_output[:90000]

# Train
for epoch in range(300):
    for trial in range(len(training_data)):
        if trial % 5000 == 0:
            print("Trial " + str(trial) + " of epoch " + str(epoch))
        train(training_data[trial].reshape(len(training_data[trial]),1), training_output[trial].reshape(len(training_output[trial]),1))
    print("Epoch: " + str(epoch) + ", Cost:" + str(cost(raw_data, raw_output)))

# Output some test trials
for i in range(95000,95010):
    input_i = str(raw_data[i])
    output_i = str(forward_prop(raw_data[i].reshape(len(raw_output[i]),1)))
    expected_output_i = str(raw_output[i])
    print("Input: " + input_i + "\tOutput: " + output_i + "\tExpected Output: " + expected_output_i)

print("Overall Cost:" + str(cost(raw_data, raw_output)))

# This model doesn't do a great job, but in most cases it does seem to be getting reasonably close. Better performance
# might be gained by changing the activation function (maybe using tanh or potentially a sin form). A periodic activation
# might work well for approximating sin