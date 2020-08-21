import numpy as np

b2 = np.random.rand(2,1)*2-1
w2 = np.random.rand(2,2)*2-1
b1 = np.random.rand(2,1)*2-1
w1 = np.random.rand(2,2)*2-1

def sigmoid(x):
    return 1/(1+np.exp(-x))
def deriv_sigmoid(x):
    return np.exp(x)/(np.exp(x)+1)**2
def forward_prop(x):
    l0 = x
    l1 = sigmoid(np.dot(w1, l0)+b1)
    l2 = sigmoid(np.dot(w2, l1)+b2)
    return l2
def train(x,y):
    global b2
    global w2
    global b1
    global w1
    
    # manual forward prop
    l0_a = x
    l1_z = np.dot(w1, l0_a)+b1
    l1_a = sigmoid(l1_z)
    l2_z = np.dot(w2, l1_a)+b2
    l2_a = sigmoid(l2_z)

    # derivatives
    dC_db2 = deriv_sigmoid(l2_z)*(2*(l2_a-y))
    dC_dw2 = np.dot(dC_db2, l1_a.T)    
    dC_db1 = deriv_sigmoid(l1_z)*np.dot(dC_db2.T, w2).T
    dC_dw1 = np.dot(dC_db1, l0_a.T)

    # adjust terms
    b2 -= .01*dC_db2
    w2 -= .01*dC_dw2
    b1 -= .01*dC_db1
    w1 -= .01*dC_dw1


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
for epoch in range(300000):
    for trial in range(4):
        train(training_data[trial].reshape(2,1), training_output[trial].reshape(2,1))

# Print the results
for i in range(4):
    result = forward_prop(training_data[i].reshape(2,1))
    result_OR = str(result[0])
    result_XOR = str(result[1])
    print(str(training_data[i]) + " -> OR: " + result_OR + ", XOR: " + result_XOR)
