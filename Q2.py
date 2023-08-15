import numpy as np
from keras.datasets import fashion_mnist

# Load the Fashion-MNIST dataset
((x_Train, y_Train), (x_Test, y_Test)) = fashion_mnist.load_data()

# Normalize image pixels bw 0 and 1
x_Train = x_Train / 255
x_Test = x_Test / 255

# Reshape the data to (number of samples, number of features)
(x, y, z) = x_Train.shape
(x1, y1, z1) = x_Test.shape
x_Train = np.reshape(x_Train, (x, y * z))
x_Test = np.reshape(x_Test, (x1, y1 * z1))


# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Define the sigmoid derivative function
def sigmoid_derivative(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))


# Define the softmax function
def softmax(z):
    sum_exp = np.sum(np.exp(z), axis=1, keepdims=True)
    return np.exp(z) / sum_exp


# Define the cross entropy loss
def cross_entropy(y_onehot, y_hat):
    # y_onehot: original values of y
    # y_hat: calculated probabilities

    loss_Value = -np.sum(y_onehot * np.log(y_hat)) / y_Train.shape[0]
    return loss_Value


# Define the forward propagation function
def forward_propagation(w, b, x):
    # w: weights
    # b: biases
    # x: dataset

    a = []  # preactivation
    h = []  # activation
    num_layers = len(w)

    for i in range(num_layers - 1):
        w[i] = np.array(w[i])
        b[i] = np.array(b[i])

        # preactivation at layer i
        a_i = np.dot(x, w[i].T) + b[i]

        # activation at layer i using sigmoid function
        h_i = sigmoid(a_i)
        a.append(a_i)
        h.append(h_i)
        x = h_i

    # For last layer use softmax function
    w[-1] = np.array(w[-1])
    b[-1] = np.array(b[-1])
    a_last = np.matmul(x, w[-1].T) + b[-1]
    h_last = softmax(a_last)
    a.append(a_last)
    h.append(h_last)
    return (a, h)


# Initialize weights and biases for all the layers
def initialize(num_layers, hidden_size):
    np.random.seed(0)
    layers = []
    layers.append(784)
    for i in range(num_layers):
        layers.append(hidden_size)
    layers.append(10)
    w = []
    b = []
    num_layers = len(layers) - 1
    for i in range(num_layers):
        w_i = np.random.uniform(-1, 1, (layers[i + 1], layers[i]))
        b_i = np.random.uniform(-1, 1, layers[i + 1])
        w.append(w_i)
        b.append(b_i)
    return (w, b, layers)


# Define a function that performs feedforward propagation on a neural network
def FeedForward(num_layers, hidden_size):
    # num_layers: Number of hidden layers used in feedforward neural network
    # hidden_size: Number of neurons in a hidden feedforward layer
    
    # Initialize the weights, biases and layers of the neural network
    (w, b, layers) = initialize(num_layers, hidden_size)

    # Perform forward propagation on the training dataset
    # This calculates the pre-activation and activation values for each layer of the network
    (a, h) = forward_propagation(w, b, x_Train)

    # The final output of the network is the predicted probability distribution over the 10 classes
    y_hat = h[-1]

    # Print the predicted probability distribution
    print('Probability Distribution over the 10 classes:')
    print(y_hat)

    # Verify that the predicted probability distribution is a valid distribution (i.e. the sum of probabilities is 1)
    print('Since it is a Probability Distribution, Sum of Probability for each point in the dataset is 1')
    print(np.sum(y_hat, axis=1))

# Call the FeedForward function with 3 hidden layers with each layer of 128 neurons
FeedForward(3, 128)

