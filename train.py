#!pip install wandb
import wandb
import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist


# default_p = dict(
#     num_layers=3,
#     hidden_size=32,
#     learning_rate=0.001,
#     num_epochs=5,
#     batch_size=32,
#     activation="ReLU",
#     optimizer="adam",
#     weight_init="Xavier",
# 	weight_decay=0	
# )

# num_layers = config.num_layers
# hidden_size = config.hidden_size
# learning_rate = config.learning_rate
# num_epochs = config.num_epochs
# batch_size = config.batch_size
# activation = config.activation
# optimizer = config.optimizer
# weight_init = config.weight_init
# alpha=config.weight_decay
# run = wandb.init(config=default_p, project="cs6910_assignment1",
#                  entity="cs22m029", reinit='True')
# config = wandb.config

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--wandb_project",   help="project_name",       type=str,                                                                  default="cs6910_assignment1")
parser.add_argument("--wandb_entity",    help="entity",             type=str,                                                                  default="cs22m029")
parser.add_argument("--dataset",         help="dataset_name",       type=str,   choices=["fashion_mnist","mnist"],                             default="fashion_mnist")
parser.add_argument("--momentum",        help="m",                  type=float, choices=[0.5,0.9],                                             default=0.9)
parser.add_argument("--beta",            help="beta",               type=float, choices=[0.5,0.9],                                             default=0.9)
parser.add_argument("--beta1",           help="beta1",              type=float, choices=[0.5,0.9],                                             default=0.9)
parser.add_argument("--beta2",           help="beta2",              type=float, choices=[0.999,0.5],                                           default=0.999)
parser.add_argument("--epsilon",         help="epsilon",            type=float, choices=[1e-3,1e-4],                                           default=1e-3)
parser.add_argument("--weight_decay",    help="weight_decay",       type=float, choices=[0,0.0005,0.5],                                        default=0)
parser.add_argument("--optimizer",       help="loss_function",      type=str,   choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],default="nadam")
parser.add_argument("--lr",              help="lr",                 type=float, choices=[1e-4,1e-3],                                           default=0.001)
parser.add_argument("--epochs",          help="epochs",             type=int,   choices=[5,10],                                                default=10)
parser.add_argument("--batch_size",      help="batch_size",         type=int,   choices=[1,16,32,64],                                          default=64)
parser.add_argument("--num_layers",      help="hidden_layer",       type=int,   choices=[3,4,5],                                               default=3)
parser.add_argument("--weight_init",     help="weight_init",        type=str,   choices=["random","Xavier"],                                   default="Xavier")
parser.add_argument("--activation",      help="activation_function",type=str,   choices=["ReLU","tanh","sigmoid"],                             default="ReLU")
parser.add_argument("--hidden_size",     help="hidden_layer_size",  type=int,   choices=[32,64,128],                                           default=128)
parser.add_argument("--loss",            help="loss_function",      type=str,   choices=["mean_squared_error", "cross_entropy"],               default="cross_entropy")
args=parser.parse_args()



project_name=args.wandb_project
entity=args.wandb_entity
dataset=args.dataset
m=args.momentum
epsilon=args.epsilon
alpha=args.weight_decay
num_layers=args.num_layers
activation=args.activation
hidden_size=args.hidden_size
loss_function=args.loss
beta=args.beta
beta1=args.beta1
beta2=args.beta2
learning_rate=args.lr
num_epochs=args.epochs
optimizer=args.optimizer
weight_init=args.weight_init
loss_function=args.loss
batch_size=args.batch_size


wandb.init( project=project_name,
                  entity=entity, reinit='True')

# Load the dataset
if (dataset=="fashion_mnist"):
    # Load the Fashion-MNIST dataset
    (x_Train, y_Train), (x_Test, y_Test) = fashion_mnist.load_data()
    
elif (dataset=="mnist"):
    # Load the MNIST dataset
    (x_Train, y_Train), (x_Test, y_Test) =mnist.load_data()

# Normalize image bw 0 and 1
x_Train = x_Train / 255
x_Test = x_Test / 255

# Split the data into training and validation sets (90:10)
x, y, z = (x_Train.shape)
np.random.seed(0)
position = np.arange(x)
np.random.shuffle(position)
# Validation set size as 10% of total data
val_size = int(x * 0.1)  

x_Val = x_Train[position[:val_size]]
y_Val = y_Train[position[:val_size]]
x_Train = x_Train[position[val_size:]]
y_Train = y_Train[position[val_size:]]

# Reshape the data to (number of samples, number of features)
x, y, z = (x_Train.shape)
x1, y1, z1 = (x_Test.shape)
x2, y2, z2 = (x_Val.shape)
x_Train = np.reshape(x_Train, (x, y*z))
x_Test = np.reshape(x_Test, (x1, y1*z1))
x_Val = np.reshape(x_Val, (x2, y2*z2))

# Define the 10 classes in the Fashion-MNIST dataset
class_Names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define the apply_Activation functions
def apply_Activation(z, activation):
    if activation == "sigmoid":
        return 1/(1+np.exp(-z))
    if activation == "tanh":
        return np.tanh(z)
    if activation == "ReLU":  # Rectified Linear Unit
        return np.maximum(0, z)

# Define the derivative of apply_Activation functions


def act_derivative(z, activation):
    if activation == "sigmoid":
        return np.multiply(apply_Activation(z, activation), (1-apply_Activation(z, activation)))
    if activation == "tanh":
        return 1-(np.tanh(z))**2
    if activation == "ReLU":
        return np.where(z > 0, 1, 0)

# defining softmax function


def softmax(z):
    sum_exp = np.sum(np.exp(z), axis=1, keepdims=True)
    return np.exp(z)/sum_exp


# Define the loss function
def loss(y_onehot, y_hat,y,loss_function,alpha):
# y_onehot: original values
# y_hat: calculated probabilities
# alpha: weight decay inclusion
    
    if loss_function == "cross_entropy":    
        total_loss_value=-np.sum(y_onehot * np.log(y_hat)) / y.shape[0]
        regularization_term = 0.5* alpha * np.sum(np.square(y_hat))/ y.shape[0]
        loss_value = (total_loss_value + regularization_term) 
        return loss_value

    elif loss_function == "mean_squared_error":
        total_loss_value = np.sum(np.square(y_onehot - y_hat))
        regularization_term = 0.5* alpha * np.sum(np.square(y_hat))
        loss_value = (total_loss_value + regularization_term) / y.shape[0]
        return loss_value



# Initialize weights and biases for all the layers
def initialize(num_layers, hidden_size, weight_init):
    np.random.seed(0)
    layers = []
    layers.append(784)
    for i in range(num_layers):
        layers.append(hidden_size)
    layers.append(10)
    w = []
    b = []
    num_layers = len(layers)-1
    if weight_init=="random":
        for i in range(num_layers):
            w_i = np.random.uniform(-1, 1, (layers[i+1], layers[i]))
            b_i = np.random.uniform(-1, 1, (layers[i+1]))
            w.append(w_i)
            b.append(b_i)
    elif weight_init=="Xavier":
        for i in range(num_layers):
            x=np.sqrt(6/(layers[i+1]+layers[i]))
            w_i = np.random.uniform(-x, x, (layers[i+1], layers[i]))
            b_i = np.random.uniform(-x, x, (layers[i+1]))
            w.append(w_i)
            b.append(b_i)
    return w, b, layers

# checking the accuracy of data
def accuracy(x, y, w, b, activation):
    a, h = forward_propagation(w, b, x, activation)
    y_hat_hot = h[-1]
    y_hat = np.argmax(y_hat_hot, axis=1)
    accuracy = 1 - np.mean(y_hat != y)
    return accuracy

# make one hot vector out of y
def onehot(y):
    num_op = 10
    y_onehot = np.zeros((y.shape[0], num_op))
    for i in range(y.shape[0]):
        y_onehot[i][y[i]] = 1
    return y_onehot

# Define the forward propagation function
def forward_propagation(w, b, x, activation):
    a = []  # preactivation
    h = []  # apply_Activation
    num_layers = len(w)

    for i in range(num_layers-1):
        w[i] = np.array(w[i])
        b[i] = np.array(b[i])
        a_i = np.dot(x, w[i].T) + b[i]
        h_i = apply_Activation(a_i, activation)
        a.append(a_i)
        h.append(h_i)
        x = h_i

    # for last layer use softmax function
    w[-1] = np.array(w[-1])
    b[-1] = np.array(b[-1])
    a_last = np.matmul(x, w[-1].T) + b[-1]
    h_last = softmax(a_last)
    a.append(a_last)
    h.append(h_last)
    return a, h


# Define the backward propagation function
def backward_propagation(a, h, y_Train, y_hat, y_onehot, x_Train, w, b, activation, alpha):

    dw = []
    db = []
    dh = []
    da = []

    # compute o/p gradients
    da_last = -(y_onehot-y_hat)
    dh_last = -(y_onehot/y_hat)
    da.append(da_last)
    dh.append(dh_last)

    n = len(w)-1
    for i in range(n, 0, -1):
        # compute gradients wrt params:
        dw_i = np.dot(da[-1].T, h[i-1])/y_Train.shape[0]

        dw.append(dw_i)
        db_i = np.sum(da[-1], axis=0)/y_Train.shape[0]
        db.append(db_i)

        # compute gradients wrt layer below:
        dh_i = np.dot(da[-1], w[i])
        dh.append(dh_i)

        # compute gradients wrt layer below (pre-apply_Activation) :
        da_i = np.multiply(dh[-1], act_derivative(a[i-1], activation))
        da.append(da_i)

    # computing w0 and b0
    dw_i = np.dot(da[-1].T, x_Train)/y_Train.shape[0]
    dw.append(dw_i)
    db_i = np.sum(da[-1], axis=0)/y_Train.shape[0]
    db.append(db_i)

    dw.reverse()
    db.reverse()
    #for L2 Regularization
    for i in range(len(dw)):
        dw[i]=np.add(dw[i],alpha*w[i])
    return dw, db


# training to find best weights and biases (with batches)
def gradient_descent_with_batch_size(x_Train, y_Train, layers, w, b, learning_rate, activation, num_epochs, batch_size, loss_function, alpha):

    print("Gradient Descent with Batch Size = ", batch_size)

    # make one hot vector out of y_Train and y_Val
    y_onehot = onehot(y_Train)
    y_onehot_val = onehot(y_Val)

    losses = []

    # dividing data in batches
    num_samples = x_Train.shape[0]
    num_batches = num_samples // batch_size

    # Loop through the specified number of epochs
    for epoch in range(num_epochs):
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0

        # Loop through each batch in the training data
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_Train_batch = x_Train[start:end, :]
            y_Train_batch = y_Train[start:end]
            y_onehot_batch = y_onehot[start:end, :]

            # Forward Propagation on training data batch
            a, h = forward_propagation(w, b, x_Train_batch, activation)

            y_hat = h[-1]
            train_loss_i = loss(y_onehot_batch, y_hat,y_Train,loss_function,alpha)
            train_epoch_loss += train_loss_i

            # Backward Propagation
            dw, db = backward_propagation(
                a, h, y_Train_batch, y_hat, y_onehot_batch, x_Train_batch, w, b, activation, alpha)

            # Update the weights and biases
            for i in range(len(w)):
                w[i] = w[i] - learning_rate * dw[i]
                b[i] = b[i] - learning_rate * db[i]

        # Forward Propagation on val batch
        a, h = forward_propagation(w, b, x_Val, activation)

        y_hatval = h[-1]
        val_loss_i = loss(y_onehot_val, y_hatval,y_Val, loss_function,alpha)
        val_epoch_loss = val_loss_i
        acc_train = accuracy(x_Train, y_Train, w, b, activation)
        acc_val = accuracy(x_Val, y_Val, w, b, activation)
        print("epoch: ", epoch)
        print("Train Accuracy : ", acc_train)
        print("Validation Accuracy : ", acc_val)
     
        print("train loss", train_epoch_loss)
        print("validation loss:", val_epoch_loss)

        wandb.log({'train_accuracy': acc_train,
                   'val_accuracy': acc_val,
                   'train_loss': train_epoch_loss,
                   'val_loss': val_epoch_loss})

    return w, b, losses

 # training to find best weights and biases (with momentum)


def momentum_gradient_descent(x_Train, y_Train, layers, w, b, learning_rate, activation, num_epochs, batch_size, beta, loss_function,alpha):
    print("Momentum Gradient Descent with Batch Size = ", batch_size)

    # make one hot vector out of y_Train and y_Val
    y_onehot = onehot(y_Train)
    y_onehot_val = onehot(y_Val)

    losses = []
    # dividing data in batches
    num_samples = x_Train.shape[0]
    num_batches = num_samples // batch_size

    prev_w = []
    prev_b = []
    num_layers = len(layers)-1
    for i in range(num_layers):
        prev_w_i = np.zeros((layers[i+1], layers[i]))
        prev_b_i = np.zeros(layers[i+1])
        prev_w.append(prev_w_i)
        prev_b.append(prev_b_i)

    for epoch in range(num_epochs):
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0

        # Loop through each batch in the training data
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_Train_batch = x_Train[start:end, :]
            y_Train_batch = y_Train[start:end]
            y_onehot_batch = y_onehot[start:end, :]

            # Forward Propagation on training data batch
            a, h = forward_propagation(w, b, x_Train_batch, activation)

            y_hat = h[-1]
            train_loss_i = loss(y_onehot_batch, y_hat,y_Train,loss_function,alpha)
            train_epoch_loss += train_loss_i

            # Backward Propagation
            dw, db = backward_propagation(
                a, h, y_Train_batch, y_hat, y_onehot_batch, x_Train_batch, w, b, activation, alpha)

            # update weight and biases giving importance to history as well
            for i in range(len(w)):

                prev_w[i] = beta * prev_w[i] + learning_rate * dw[i]
                prev_b[i] = beta * prev_b[i] + learning_rate * db[i]

                w[i] = w[i] - prev_w[i]
                b[i] = b[i] - prev_b[i]

    # Forward Propagation on val batch
        a, h = forward_propagation(w, b, x_Val, activation)

        y_hatval = h[-1]
        val_loss_i = loss(y_onehot_val, y_hatval,y_Val,loss_function,alpha)
        val_epoch_loss = val_loss_i
        acc_train = accuracy(x_Train, y_Train, w, b, activation)
        acc_val = accuracy(x_Val, y_Val, w, b, activation)
        print("epoch: ", epoch)
        print("Train Accuracy : ", acc_train)
        print("Validation Accuracy : ", acc_val)
     
        print("train loss", train_epoch_loss)
        print("validation loss:", val_epoch_loss)

        wandb.log({'train_accuracy': acc_train,
                   'val_accuracy': acc_val,
                   'train_loss': train_epoch_loss,
                   'val_loss': val_epoch_loss})

    return w, b, losses


# Training to find best weights and biases (with Nesterov accelerated gradient descent)
def nesterov_gradient_descent(x_Train, y_Train, w, b, layers, learning_rate, num_epochs, batch_size, beta, activation, loss_function,alpha):
    print("Nesterov accelerated gradient descent")

    # make one hot vector out of y_Train and y_Val
    y_onehot = onehot(y_Train)
    y_onehot_val = onehot(y_Val)

    losses = []
    # Divide data into batches
    num_samples = x_Train.shape[0]
    num_batches = num_samples // batch_size

    prev_w = []
    prev_b = []
    num_layers = len(layers)-1
    for i in range(num_layers):
        prev_w_i = np.zeros((layers[i+1], layers[i]))
        prev_b_i = np.zeros(layers[i+1])
        prev_w.append(prev_w_i)
        prev_b.append(prev_b_i)

    for epoch in range(num_epochs):
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0

        # Loop through each batch in the training data
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_Train_batch = x_Train[start:end, :]
            y_Train_batch = y_Train[start:end]
            y_onehot_batch = y_onehot[start:end, :]

            # Calculate the gradients using Nesterov accelerated gradient descent
            w_nesterov = []
            b_nesterov = []
            for i in range(len(w)):
                w_nesterov_i = w[i] - beta * prev_w[i]
                b_nesterov_i = b[i] - beta * prev_b[i]
                w_nesterov.append(w_nesterov_i)
                b_nesterov.append(b_nesterov_i)

            # Forward propagation on training data batch
            a, h = forward_propagation(
                w_nesterov, b_nesterov, x_Train_batch, activation)

            y_hat = h[-1]
            train_loss_i = loss(y_onehot_batch, y_hat,y_Train,loss_function,alpha)
            train_epoch_loss += train_loss_i

            # Backward propagation
            dw, db = backward_propagation(
                a, h, y_Train_batch, y_hat, y_onehot_batch, x_Train_batch, w_nesterov, b_nesterov, activation, alpha)

            # Update the weights and biases with Nesterov accelerated gradient descent
            for i in range(len(w)):
                prev_w[i] = beta * prev_w[i] + learning_rate * dw[i]
                prev_b[i] = beta * prev_b[i] + learning_rate * db[i]
                w[i] = w[i] - prev_w[i]
                b[i] = b[i] - prev_b[i]

        # Forward Propagation on val batch
        a, h = forward_propagation(w, b, x_Val, activation)

        y_hatval = h[-1]
        val_loss_i = loss(y_onehot_val, y_hatval,y_Val,loss_function,alpha)
        val_epoch_loss = val_loss_i
        acc_train = accuracy(x_Train, y_Train, w, b, activation)
        acc_val = accuracy(x_Val, y_Val, w, b, activation)
        print("epoch: ", epoch)
        print("Train Accuracy : ", acc_train)
        print("Validation Accuracy : ", acc_val)
     
        print("train loss", train_epoch_loss)
        print("validation loss:", val_epoch_loss)

        wandb.log({'train_accuracy': acc_train,
                   'val_accuracy': acc_val,
                   'train_loss': train_epoch_loss,
                   'val_loss': val_epoch_loss})

    return w, b, losses


# Training to find best weights and biases (with RMSProp)
def rmsProp(x_Train, y_Train, w, b, layers, learning_rate, num_epochs, batch_size, beta, epsilon, activation, loss_function,alpha):
    print("RMSProp algorithm")

    # make one hot vector out of y_Train and y_Val
    y_onehot = onehot(y_Train)
    y_onehot_val = onehot(y_Val)

    losses = []
    # Divide data into batches
    num_samples = x_Train.shape[0]
    num_batches = num_samples // batch_size

    prev_w = []
    prev_b = []
    num_layers = len(layers)-1
    for i in range(num_layers):
        prev_w_i = np.zeros((layers[i+1], layers[i]))
        prev_b_i = np.zeros(layers[i+1])
        prev_w.append(prev_w_i)
        prev_b.append(prev_b_i)

    for epoch in range(num_epochs):
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0

        # Loop through each batch in the training data
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_Train_batch = x_Train[start:end, :]
            y_Train_batch = y_Train[start:end]
            y_onehot_batch = y_onehot[start:end, :]

            # Forward propagation on training data batch
            a, h = forward_propagation(w, b, x_Train_batch, activation)

            y_hat = h[-1]
            train_loss_i = loss(y_onehot_batch, y_hat,y_Train,loss_function,alpha)
            train_epoch_loss += train_loss_i

            # Backward propagation
            dw, db = backward_propagation(
                a, h, y_Train_batch, y_hat, y_onehot_batch, x_Train_batch, w, b, activation, alpha)

            # Update the weights and biases with RMSProp
            for i in range(len(w)):
                prev_w[i] = beta * prev_w[i] + (1 - beta) * (dw[i] ** 2)
                prev_b[i] = beta * prev_b[i] + (1 - beta) * (db[i] ** 2)
                w[i] = w[i] - (learning_rate /
                               (np.sqrt(prev_w[i]) + epsilon)) * dw[i]
                b[i] = b[i] - (learning_rate /
                               (np.sqrt(prev_b[i]) + epsilon)) * db[i]

        # Forward Propagation on val batch
        a, h = forward_propagation(w, b, x_Val, activation)

        y_hatval = h[-1]
        val_loss_i = loss(y_onehot_val, y_hatval,y_Val,loss_function,alpha)
        val_epoch_loss = val_loss_i
        acc_train = accuracy(x_Train, y_Train, w, b, activation)
        acc_val = accuracy(x_Val, y_Val, w, b, activation)
        print("epoch: ", epoch)
        print("Train Accuracy : ", acc_train)
        print("Validation Accuracy : ", acc_val)
     
        print("train loss", train_epoch_loss)
        print("validation loss:", val_epoch_loss)

        wandb.log({'train_accuracy': acc_train,
                   'val_accuracy': acc_val,
                   'train_loss': train_epoch_loss,
                   'val_loss': val_epoch_loss})

    return w, b, losses


# Training to find best weights and biases (with Adaptive Moments)
def adam(x_Train, y_Train, w, b, layers, learning_rate, num_epochs, batch_size, beta1, beta2, epsilon, activation, loss_function,alpha):
    print("Adam : Adaptive Moments algorithm")
    # make one hot vector out of y_Train and y_Val
    y_onehot = onehot(y_Train)
    y_onehot_val = onehot(y_Val)

    losses = []
    # Divide data into batches
    num_samples = x_Train.shape[0]
    num_batches = num_samples // batch_size

    m_w = []
    m_b = []
    prev_w = []
    prev_b = []

    num_layers = len(layers)-1
    for i in range(num_layers):
        m_w_i = np.zeros((layers[i+1], layers[i]))
        m_b_i = np.zeros(layers[i+1])
        prev_w_i = np.zeros((layers[i+1], layers[i]))
        prev_b_i = np.zeros(layers[i+1])
        m_w.append(m_w_i)
        m_b.append(m_b_i)
        prev_w.append(prev_w_i)
        prev_b.append(prev_b_i)

    for epoch in range(num_epochs):
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0

        # Loop through each batch in the training data
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_Train_batch = x_Train[start:end, :]
            y_Train_batch = y_Train[start:end]
            y_onehot_batch = y_onehot[start:end, :]

            # Forward propagation on training data batch
            a, h = forward_propagation(w, b, x_Train_batch, activation)

            y_hat = h[-1]
            train_loss_i = loss(y_onehot_batch, y_hat,y_Train,loss_function,alpha)
            train_epoch_loss += train_loss_i

            # compute the gradients by Backward propagation
            dw, db = backward_propagation(
                a, h, y_Train_batch, y_hat, y_onehot_batch, x_Train_batch, w, b, activation, alpha)

            # Update the weights and biases with adam
            for i in range(len(w)):
                m_w[i] = beta1 * m_w[i] + (1 - beta1) * dw[i]
                m_b[i] = beta1 * m_b[i] + (1 - beta1) * db[i]
                prev_w[i] = beta2 * prev_w[i] + (1 - beta2) * (dw[i] ** 2)
                prev_b[i] = beta2 * prev_b[i] + (1 - beta2) * (db[i] ** 2)

                m_w_hat = m_w[i]/(1-np.power(beta1, i+1))
                m_b_hat = m_b[i]/(1-np.power(beta1, i+1))
                prev_w_hat = prev_w[i]/(1-np.power(beta2, i+1))
                prev_b_hat = prev_b[i]/(1-np.power(beta2, i+1))

                # update parameters
                w[i] = w[i] - (learning_rate * m_w_hat /
                               (np.sqrt(prev_w_hat)+epsilon))
                b[i] = b[i] - (learning_rate * m_b_hat /
                               (np.sqrt(prev_b_hat)+epsilon))

        # Forward Propagation on val batch
        a, h = forward_propagation(w, b, x_Val, activation)

        y_hatval = h[-1]
        val_loss_i = loss(y_onehot_val, y_hatval,y_Val,loss_function,alpha)
        val_epoch_loss = val_loss_i
        acc_train = accuracy(x_Train, y_Train, w, b, activation)
        acc_val = accuracy(x_Val, y_Val, w, b, activation)
        print("epoch: ", epoch)
        print("Train Accuracy : ", acc_train)
        print("Validation Accuracy : ", acc_val)
     
        print("train loss", train_epoch_loss)
        print("validation loss:", val_epoch_loss)

        wandb.log({'train_accuracy': acc_train,
                   'val_accuracy': acc_val,
                   'train_loss': train_epoch_loss,
                   'val_loss': val_epoch_loss})

    return w, b, losses

# Training to find best weights and biases (with Nesterov Adaptive Moments)
def nadam(x_Train, y_Train, w, b, layers, learning_rate, num_epochs, batch_size, beta1, beta2, epsilon, activation, loss_function,alpha):

    print("Nesterov Adam Adaptive Moments optimizer")
    # x_Train, y_Train, layers, Learning rate, max epochs, batch size, beta1, beta2, epsilon)

    # make one hot vector out of y_Train and y_Val
    y_onehot = onehot(y_Train)
    y_onehot_val = onehot(y_Val)

    losses = []
    # Divide data into batches
    num_samples = x_Train.shape[0]
    num_batches = num_samples // batch_size

    m_w = []
    m_b = []
    prev_w = []
    prev_b = []

    num_layers = len(layers)-1
    for i in range(num_layers):
        m_w_i = np.zeros((layers[i+1], layers[i]))
        m_b_i = np.zeros(layers[i+1])
        prev_w_i = np.zeros((layers[i+1], layers[i]))
        prev_b_i = np.zeros(layers[i+1])
        m_w.append(m_w_i)
        m_b.append(m_b_i)
        prev_w.append(prev_w_i)
        prev_b.append(prev_b_i)

    for epoch in range(num_epochs):
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0

        # Loop through each batch in the training data
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_Train_batch = x_Train[start:end, :]
            y_Train_batch = y_Train[start:end]
            y_onehot_batch = y_onehot[start:end, :]

            # Forward propagation on training data batch
            a, h = forward_propagation(w, b, x_Train_batch, activation)

            y_hat = h[-1]
            train_loss_i = loss(y_onehot_batch, y_hat,y_Train,loss_function,alpha)
            train_epoch_loss += train_loss_i

            # compute the gradients by Backward propagation
            dw, db = backward_propagation(
                a, h, y_Train_batch, y_hat, y_onehot_batch, x_Train_batch, w, b, activation, alpha)

            # Update the weights and biases with nadam
            for i in range(len(w)):
                m_w[i] = beta1 * m_w[i] + (1 - beta1) * dw[i]
                m_b[i] = beta1 * m_b[i] + (1 - beta1) * db[i]
                prev_w[i] = beta2 * prev_w[i] + (1 - beta2) * (dw[i] ** 2)
                prev_b[i] = beta2 * prev_b[i] + (1 - beta2) * (db[i] ** 2)

                m_w_hat = m_w[i]/(1-np.power(beta1, i+1))
                m_b_hat = m_b[i]/(1-np.power(beta1, i+1))
                prev_w_hat = prev_w[i]/(1-np.power(beta2, i+1))
                prev_b_hat = prev_b[i]/(1-np.power(beta2, i+1))

                m_w_dash = beta1 * m_w_hat + \
                    (1-beta1) * dw[i] / (1-np.power(beta1, i+1))
                m_b_dash = beta1 * m_b_hat + \
                    (1-beta1) * db[i] / (1-np.power(beta1, i+1))

                # update parameters
                w[i] = w[i] - (learning_rate * m_w_dash /
                               (np.sqrt(prev_w_hat)+epsilon))
                b[i] = b[i] - (learning_rate * m_b_dash /
                               (np.sqrt(prev_b_hat)+epsilon))

        # Forward Propagation on val batch
        a, h = forward_propagation(w, b, x_Val, activation)

        y_hatval = h[-1]
        val_loss_i = loss(y_onehot_val, y_hatval,y_Val,loss_function,alpha)
        val_epoch_loss = val_loss_i
        acc_train = accuracy(x_Train, y_Train, w, b, activation)
        acc_val = accuracy(x_Val, y_Val, w, b, activation)
        print("epoch: ", epoch)
        print("Train Accuracy : ", acc_train)
        print("Validation Accuracy : ", acc_val)
     
        print("train loss", train_epoch_loss)
        print("validation loss:", val_epoch_loss)

        wandb.log({'train_accuracy': acc_train,
                   'val_accuracy': acc_val,
                   'train_loss': train_epoch_loss,
                   'val_loss': val_epoch_loss})

    return w, b, losses


def Neural_Network(num_layers, hidden_size, learning_rate, num_epochs, batch_size, beta,
                   beta1, beta2, epsilon, activation, optimizer, loss_function, weight_init,alpha):

    # Initialize the weights w0 and biases b0 and layers of neural network
    w, b, layers = initialize(num_layers, hidden_size,weight_init)

    if optimizer == 'sgd':
        w, b, loss_history = gradient_descent_with_batch_size(x_Train, y_Train, layers, w, b, learning_rate, activation,
                                                              num_epochs, batch_size, loss_function,alpha)
    elif optimizer == 'momentum':
        w, b, loss_history = momentum_gradient_descent(x_Train, y_Train, layers, w, b, learning_rate, activation,
                                                       num_epochs, batch_size, beta, loss_function,alpha)
    elif optimizer == 'nesterov':
        w, b, loss_history = nesterov_gradient_descent(x_Train, y_Train, w, b, layers,   learning_rate,
                                                       num_epochs, batch_size, beta, activation, loss_function,alpha)
    elif optimizer == 'rmsprop':
        w, b, loss_history = rmsProp(x_Train, y_Train, w, b, layers,  learning_rate,
                                     num_epochs, batch_size, beta, epsilon,activation, loss_function,alpha)
    elif optimizer == 'adam':
        w, b, loss_history = adam(x_Train, y_Train, w, b, layers,   learning_rate,
                                  num_epochs, batch_size, beta1, beta2, epsilon,activation, loss_function,alpha)
    elif optimizer == 'nadam':
        w, b, loss_history = nadam(x_Train, y_Train, w, b, layers,  learning_rate,
                                   num_epochs, batch_size, beta1, beta2, epsilon,activation, loss_function,alpha)

    a, h = forward_propagation(w, b, x_Test, activation)
    y_hat_hot = h[-1]
    y_hat = np.argmax(y_hat_hot, axis=1)
    accuracy = 1 - np.mean(y_hat != y_Test)
    print("Test Accuracy",accuracy)


    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=y_Test, preds=y_hat,
                        class_names=class_Names)})

# run.name = "hl_"+str(num_layers)+"_bs_"+str(batch_size) + \
#     "_ac_"+activation+"_op_"+optimizer

Neural_Network ( num_layers, hidden_size, learning_rate, num_epochs, batch_size, beta, beta1,
               beta2, epsilon, activation, optimizer,loss_function, weight_init,alpha ) 
