import numpy as np

w1_00 = np.random.normal()
w1_01 = np.random.normal()
w1_10 = np.random.normal()
w1_11 = np.random.normal()
w2_00 = np.random.normal()
w2_01 = np.random.normal()
w2_10 = np.random.normal()
w2_11 = np.random.normal()
b1_0 = np.random.normal()
b1_1 = np.random.normal()
b2_0 = np.random.normal()
b2_1 = np.random.normal()

def sigmoid(x):
    return 1/(1+np.exp(-x))
def deriv_sigmoid(x):
    return np.exp(x)/(np.exp(x)+1)**2
def forward_prop(x):
    l0 = training_data[x].reshape(2,1)
    l1 = sigmoid(np.dot(np.array([[w1_00,w1_01],[w1_10,w1_11]]), l0) + np.array([[b1_0],[b1_1]]))
    l2 = sigmoid(np.dot(np.array([[w2_00,w2_01],[w2_10,w2_11]]), l1) + np.array([[b2_0],[b2_1]]))
    return l2
def train(x):
    global w1_00
    global w1_01
    global w1_10
    global w1_11
    global w2_00
    global w2_01
    global w2_10
    global w2_11
    global b1_0
    global b1_1
    global b2_0
    global b2_1
    
    # forward prop
    l0_a = training_data[x].reshape(2,1)
    l1_z = np.dot(np.array([[w1_00,w1_01],[w1_10,w1_11]]), l0_a) + np.array([[b1_0],[b1_1]])
    l1_a = sigmoid(l1_z)
    l2_z = np.dot(np.array([[w2_00,w2_01],[w2_10,w2_11]]), l1_a) + np.array([[b2_0],[b2_1]])
    l2_a = sigmoid(l2_z)

    # derivatives
    dC_db2_0 = deriv_sigmoid(l2_z[0][0])*2*(l2_a[0][0]-training_output[x][0])
    dC_dw2_00 = l1_a[0][0]*dC_db2_0
    dC_dw2_01 = l1_a[1][0]*dC_db2_0

    dC_db2_1 = deriv_sigmoid(l2_z[1][0])*2*(l2_a[1][0]-training_output[x][1])
    dC_dw2_10 = l1_a[0][0]*dC_db2_1
    dC_dw2_11 = l1_a[1][0]*dC_db2_1

    dC_db1_0 = deriv_sigmoid(l1_z[0][0])*(w2_00*dC_db2_0 + w2_10*dC_db2_1)
    dC_dw1_00 = l0_a[0][0]*dC_db1_0
    dC_dw1_01 = l0_a[1][0]*dC_db1_0

    dC_db1_1 = deriv_sigmoid(l1_z[1][0])*(w2_01*dC_db2_0 + w2_11*dC_db2_1)
    dC_dw1_10 = l0_a[0][0]*dC_db1_1
    dC_dw1_11 = l0_a[1][0]*dC_db1_1

    #adjust terms
    w1_00 -= .01*dC_dw1_00
    w1_01 -= .01*dC_dw1_01
    w1_10 -= .01*dC_dw1_10
    w1_11 -= .01*dC_dw1_11
    w2_00 -= .01*dC_dw2_00
    w2_01 -= .01*dC_dw2_01
    w2_10 -= .01*dC_dw2_10
    w2_11 -= .01*dC_dw2_11
    b1_0 -= .01*dC_db1_0
    b1_1 -= .01*dC_db1_1
    b2_0 -= .01*dC_db2_0
    b2_1 -= .01*dC_db2_1


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
        train(trial)

# Print the results
for trial in range(4):
    result = forward_prop(trial)
    result_OR = str(result[0])
    result_XOR = str(result[1])
    print(str(training_data[trial]) + " -> OR: " + result_OR + ", XOR: " + result_XOR)
