#import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

#wandb.init(project="Q2")

# loading the Fashion-MNIST dataset
(x_Train, y_Train), (x_Test, y_Test) = fashion_mnist.load_data()

# Normalize image bw 0 and 1
x_Train = x_Train/255
x_Test  = x_Test /255

# Reshape the data to (number of samples, number of features)
x,y,z=(x_Train.shape)
x1,y1,z1=(x_Test.shape)
x_Train = np.reshape(x_Train, (x, y*z))
x_Test  = np.reshape(x_Test, (x1, y1*z1))

# defining the 10 classes in the Fashion-MNIST dataset
class_Names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Define the activation functions
def activation(z,act_fun):
    if act_fun=="sigmoid":
      return 1/(1+np.exp(-z))
    if act_fun=="tanh":
      return np.tanh(z)
    if act_fun=="ReLU": #Rectified Linear Unit
      return np.maximum(0,z)
  
#Define the derivative of activation functions
def act_derivative (z,act_fun):
  if act_fun=="sigmoid":
    return np.multiply(activation(z,act_fun),(1-activation(z,act_fun)))
  if act_fun=="tanh":
    return 1-(np.tanh(z))**2
  if act_fun=="ReLU": 
    return np.where(z > 0, 1, 0)

# defining softmax function
def softmax(z):
    sum_exp = np.sum(np.exp(z), axis=1, keepdims=True)
    return np.exp(z)/sum_exp

# Define the cross entropy loss
def cross_entropy(y_onehot, y_hat):
    # y_onehot: original values
    # y_hat: calculated probabilities 
    loss_Value = -np.sum(y_onehot * np.log(y_hat )) / y_Train.shape[0]
    return loss_Value

# Initialize weights and biases for all the layers
def initialize(layers):
    np.random.seed(0)
    w = []
    b = []
    num_layers = len(layers)-1
    for i in range(num_layers):
        w_i = np.random.uniform(-1,1,(layers[i+1], layers[i]))
        b_i = np.random.uniform(-1,1,(layers[i+1]))
        w.append(w_i)
        b.append(b_i)
    return w, b

# Define the forward propagation function
def forward_propagation(w, b, x, act_fun):
    a = [] # preactivation
    h = [] # activation
    num_layers = len(w)

    for i in range(num_layers-1):
        w[i]=np.array(w[i])
        b[i]=np.array(b[i])
        a_i =  np.dot(x, w[i].T) + b[i]
        h_i = activation(a_i,act_fun)
        a.append(a_i)
        h.append(h_i)
        x = h_i

    # for last layer use softmax function
    w[-1]=np.array(w[-1])
    b[-1]=np.array(b[-1])
    a_last = np.matmul(x,w[-1].T) + b[-1]
    h_last = softmax(a_last)
    a.append(a_last)
    h.append(h_last)
    return a, h



# Define the backward propagation function 
def backward_propagation( a, h, y_Train, y_hat,y_onehot,x_Train,w,b, act_fun):

    dw=[]
    db=[]
    dh=[]
    da=[]

    #compute o/p gradients
    da_last=-(y_onehot-y_hat)
    dh_last=-(y_onehot/y_hat)
    da.append(da_last)
    dh.append(dh_last)
    
    n=len(w)-1
    for i in range  (n,0,-1) :
      #compute gradients wrt params: 
      dw_i=np.dot(da[-1].T,h[i-1])/y_Train.shape[0]
      dw.append(dw_i)
      db_i=np.sum(da[-1],axis=0)/y_Train.shape[0]
      db.append(db_i)

      #compute gradients wrt layer below: 
      dh_i=np.dot(da[-1],w[i])
      dh.append(dh_i)

      #compute gradients wrt layer below (pre-activation) :
      da_i=np.multiply(dh[-1],act_derivative(a[i-1],act_fun))
      da.append(da_i)

    # computing w0 and b0 
    dw_i=np.dot(da[-1].T,x_Train)/y_Train.shape[0]
    dw.append(dw_i)
    db_i=np.sum(da[-1],axis=0)/y_Train.shape[0]
    db.append(db_i)

    dw.reverse()
    db.reverse()
    
    return dw, db 

#checking the accuracy of the test data
def test_accuracy(x_Test, y_Test, w, b, act_fun="sigmoid"):
    a, h = forward_propagation(w, b, x_Test, act_fun)
    y_hat_hot=h[-1]
    y_hat = np.argmax(y_hat_hot, axis=1)
    accuracy = 1 - np.mean(y_hat != y_Test)
    print(accuracy)
    

#training to find best weights and biases
def gradient_descent(x_Train, y_Train, layers, learning_rate, num_epochs, act_fun="sigmoid"):
    # Initialize the weights w0 and biases b0 
    w, b = initialize(layers)

   # make one hot vector out of y_Train
    num_op=10
    y_onehot = np.zeros((y_Train.shape[0],num_op))
    for i in range(y_Train.shape[0]):
        y_onehot[i][y_Train[i]]=1

    losses = []
    for epoch in range(num_epochs): 
        # Forward propagation on training data
        a, h = forward_propagation(w, b, x_Train, act_fun)
        y_hat = h[-1]
        
        # Calculate loss
        loss_i=cross_entropy(y_onehot,y_hat)
        losses.append(loss_i)
        
        #Backward propagation 
        dw, db = backward_propagation( a, h, y_Train, y_hat,y_onehot,x_Train,w,b, act_fun)
        
        #Update the weights and biases
        for i in range(len(w)):
            w[i] = w[i] - learning_rate * dw[i]
            b[i] = b[i] - learning_rate * db[i]
         
        #print the loss 
        print('Epoch :', epoch,'AvgLoss =', loss_i)
    
    return w, b, losses
  
#training to find best weights and biases (with batches)
def gradient_descent_with_batch_size(x_Train, y_Train, layers, learning_rate, num_epochs, batch_size,act_fun="sigmoid"):
    # Initialize the weights w0 and biases b0 
    w, b = initialize(layers)

    # make one hot vector out of y_Train
    num_op = 10
    y_onehot = np.zeros((y_Train.shape[0], num_op))
    for i in range(y_Train.shape[0]):
        y_onehot[i][y_Train[i]] = 1

    losses = []
    
    #dividing data in batches
    num_samples = x_Train.shape[0]
    num_batches = num_samples // batch_size
    
    # Loop through the specified number of epochs
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Loop through each batch in the training data
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_Train_batch=x_Train[start:end, :]
            y_Train_batch=y_Train[start:end]
            y_onehot_batch=y_onehot[start:end, :]

            # Forward Propagation on training data batch
            a, h = forward_propagation(w, b, x_Train_batch,act_fun)

            y_hat = h[-1]
            loss_i = cross_entropy(y_onehot_batch, y_hat)
            epoch_loss += loss_i

            # Backward Propagation
            dw, db = backward_propagation(a, h, y_Train_batch, y_hat, y_onehot_batch, x_Train_batch, w, b,act_fun)
            
            #Update the weights and biases
            for i in range(len(w)):
                w[i] = w[i] - learning_rate * dw[i]
                b[i] = b[i] - learning_rate * db[i]

        losses.append(epoch_loss)
        print('Epoch :', epoch, 'AvgLoss =', epoch_loss)

    return w, b, losses

 #training to find best weights and biases (with momentum)
def momentum_gradient_descent(x_Train, y_Train, layers, learning_rate, num_epochs, batch_size, beta,act_fun="sigmoid" ):
    # Initialize the weights w0 and biases b0 
    w, b = initialize(layers) 

    # make one hot vector out of y_Train
    num_op = 10
    y_onehot = np.zeros((y_Train.shape[0], num_op))
    for i in range(y_Train.shape[0]):
        y_onehot[i][y_Train[i]] = 1

    losses = []
    #dividing data in batches
    num_samples = x_Train.shape[0]
    num_batches = num_samples // batch_size
    
    prev_w = []
    prev_b=  []
    num_layers = len(layers)-1
    for i in range(num_layers):
        prev_w_i = np.zeros((layers[i+1], layers[i]))
        prev_b_i = np.zeros(layers[i+1])
        prev_w.append(prev_w_i)
        prev_b.append(prev_b_i)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Loop through each batch in the training data
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_Train_batch=x_Train[start:end, : ]
            y_Train_batch=y_Train[start:end]
            y_onehot_batch=y_onehot[start:end, : ]

            # Forward Propagation on training data batch
            a, h = forward_propagation(w, b, x_Train_batch,act_fun)

            y_hat = h[-1]
            loss_i = cross_entropy(y_onehot_batch, y_hat)
            epoch_loss += loss_i

            # Backward Propagation
            dw, db = backward_propagation(a, h, y_Train_batch, y_hat, y_onehot_batch, x_Train_batch, w, b,act_fun)

            #update weight and biases giving importance to history as well
            for i in range(len(w)):
       
                prev_w [i] = beta * prev_w [i] + learning_rate * dw[i]
                prev_b [i] = beta * prev_b [i] + learning_rate * db[i]

                w[i] = w[i] -  prev_w [i]
                b[i] = b[i] -  prev_b [i]

        losses.append(epoch_loss)
        print('Epoch :', epoch, 'AvgLoss =', epoch_loss)

    return w, b, losses

# Training to find best weights and biases (with Nesterov accelerated gradient descent)
def nesterov_gradient_descent(x_Train, y_Train, layers, learning_rate, num_epochs, batch_size, beta,act_fun="sigmoid"):
    # Initialize the weights w0 and biases b0 
    w, b = initialize(layers) 

    # Make one hot vector out of y_Train
    num_op = 10
    y_onehot = np.zeros((y_Train.shape[0], num_op))
    for i in range(y_Train.shape[0]):
        y_onehot[i][y_Train[i]] = 1

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
        epoch_loss = 0.0

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
            a, h = forward_propagation(w_nesterov, b_nesterov, x_Train_batch,act_fun)

            y_hat = h[-1]
            loss_i = cross_entropy(y_onehot_batch, y_hat)
            epoch_loss += loss_i

            # Backward propagation
            dw, db = backward_propagation(a, h, y_Train_batch, y_hat, y_onehot_batch, x_Train_batch, w_nesterov, b_nesterov,act_fun)

            # Update the weights and biases with Nesterov accelerated gradient descent
            for i in range(len(w)):
                prev_w[i] = beta * prev_w[i] + learning_rate * dw[i]
                prev_b[i] = beta * prev_b[i] + learning_rate * db[i]
                w[i] = w[i] - prev_w[i]
                b[i] = b[i] - prev_b[i]

        losses.append(epoch_loss)
        print('Epoch:', epoch, 'AvgLoss =', epoch_loss)

    return w, b, losses



# Training to find best weights and biases (with RMSProp)
def rmsProp(x_Train, y_Train, layers, learning_rate, num_epochs, batch_size, beta, epsilon,act_fun="sigmoid"):
    # Initialize the weights w0 and biases b0 
    w, b = initialize(layers) 

    # Make one hot vector out of y_Train
    num_op = 10
    y_onehot = np.zeros((y_Train.shape[0], num_op))
    for i in range(y_Train.shape[0]):
        y_onehot[i][y_Train[i]] = 1

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
        epoch_loss = 0.0

        # Loop through each batch in the training data
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_Train_batch = x_Train[start:end, :]
            y_Train_batch = y_Train[start:end]
            y_onehot_batch = y_onehot[start:end, :] 

            # Forward propagation on training data batch
            a, h = forward_propagation(w, b, x_Train_batch,act_fun)

            y_hat = h[-1]
            loss_i = cross_entropy(y_onehot_batch, y_hat)
            epoch_loss += loss_i

            # Backward propagation
            dw, db = backward_propagation(a, h, y_Train_batch, y_hat, y_onehot_batch, x_Train_batch, w, b,act_fun)

            # Update the weights and biases with RMSProp
            for i in range(len(w)):
                prev_w[i] = beta * prev_w[i] + (1 - beta) * (dw[i] ** 2)
                prev_b[i] = beta * prev_b[i] + (1 - beta) * (db[i] ** 2)
                w[i] = w[i] - (learning_rate / (np.sqrt(prev_w[i]) + epsilon)) * dw[i]
                b[i] = b[i] - (learning_rate / (np.sqrt(prev_b[i]) + epsilon)) * db[i]

        losses.append(epoch_loss)
        print('Epoch:', epoch, 'AvgLoss =', epoch_loss)

    return w, b, losses


# Training to find best weights and biases (with Adaptive Moments)
def adam(x_Train, y_Train, layers, learning_rate, num_epochs, batch_size, beta1, beta2, epsilon,act_fun="sigmoid"):
    # Initialize the weights w0 and biases b0 
    w, b = initialize(layers) 

    # Make one hot vector out of y_Train
    num_op = 10
    y_onehot = np.zeros((y_Train.shape[0], num_op))
    for i in range(y_Train.shape[0]):
        y_onehot[i][y_Train[i]] = 1

    losses = []
    # Divide data into batches
    num_samples = x_Train.shape[0]
    num_batches = num_samples // batch_size
    
    m_w=[]
    m_b=[]
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
        epoch_loss = 0.0

        # Loop through each batch in the training data
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_Train_batch = x_Train[start:end, :]
            y_Train_batch = y_Train[start:end]
            y_onehot_batch = y_onehot[start:end, :] 

            # Forward propagation on training data batch
            a, h = forward_propagation(w, b, x_Train_batch,act_fun)

            y_hat = h[-1]
            loss_i = cross_entropy(y_onehot_batch, y_hat)
            epoch_loss += loss_i

            # compute the gradients by Backward propagation
            dw, db = backward_propagation(a, h, y_Train_batch, y_hat, y_onehot_batch, x_Train_batch, w, b,act_fun)

            # Update the weights and biases with adam
            for i in range(len(w)):
                m_w[i] = beta1 * m_w[i] + (1 - beta1) * dw[i]
                m_b[i] = beta1 * m_b[i] + (1 - beta1) * db[i]
                prev_w[i] = beta2 * prev_w[i] + (1 - beta2) * (dw[i] ** 2)
                prev_b[i] = beta2 * prev_b[i] + (1 - beta2) * (db[i] ** 2)
                
                m_w_hat = m_w[i]/(1-np.power(beta1,i+1))
                m_b_hat = m_b[i]/(1-np.power(beta1,i+1))
                prev_w_hat = prev_w[i]/(1-np.power(beta2,i+1))
                prev_b_hat = prev_b[i]/(1-np.power(beta2,i+1))      

                #update parameters
                w[i] = w[i] - (learning_rate *m_w_hat / (np.sqrt(prev_w_hat)+epsilon))
                b[i] = b[i] - (learning_rate *m_b_hat / (np.sqrt(prev_b_hat)+epsilon))

        losses.append(epoch_loss)
        print('Epoch:', epoch, 'AvgLoss =', epoch_loss)

    return w, b, losses

# Training to find best weights and biases (with Nesterov Adaptive Moments)
def nadam(x_Train, y_Train, layers, learning_rate, num_epochs, batch_size, beta1, beta2, epsilon,act_fun="sigmoid"):
    # Initialize the weights w0 and biases b0 
    w, b = initialize(layers) 

    # Make one hot vector out of y_Train
    num_op = 10
    y_onehot = np.zeros((y_Train.shape[0], num_op))
    for i in range(y_Train.shape[0]):
        y_onehot[i][y_Train[i]] = 1

    losses = []
    # Divide data into batches
    num_samples = x_Train.shape[0]
    num_batches = num_samples // batch_size
    
    m_w=[]
    m_b=[]
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
        epoch_loss = 0.0

        # Loop through each batch in the training data
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_Train_batch = x_Train[start:end, :]
            y_Train_batch = y_Train[start:end]
            y_onehot_batch = y_onehot[start:end, :] 

            # Forward propagation on training data batch
            a, h = forward_propagation(w, b, x_Train_batch,act_fun)

            y_hat = h[-1]
            loss_i = cross_entropy(y_onehot_batch, y_hat)
            epoch_loss += loss_i

            # compute the gradients by Backward propagation
            dw, db = backward_propagation(a, h, y_Train_batch, y_hat, y_onehot_batch, x_Train_batch, w, b,act_fun)

            # Update the weights and biases with nadam
            for i in range(len(w)):
                m_w[i] = beta1 * m_w[i] + (1 - beta1) * dw[i]
                m_b[i] = beta1 * m_b[i] + (1 - beta1) * db[i]
                prev_w[i] = beta2 * prev_w[i] + (1 - beta2) * (dw[i] ** 2)
                prev_b[i] = beta2 * prev_b[i] + (1 - beta2) * (db[i] ** 2)
                
                m_w_hat = m_w[i]/(1-np.power(beta1,i+1))
                m_b_hat = m_b[i]/(1-np.power(beta1,i+1))
                prev_w_hat = prev_w[i]/(1-np.power(beta2,i+1))
                prev_b_hat = prev_b[i]/(1-np.power(beta2,i+1))  

                m_w_dash= beta1 *m_w_hat + (1-beta1) * dw[i] /(1-np.power(beta1,i+1))
                m_b_dash= beta1 *m_b_hat + (1-beta1) * db[i] /(1-np.power(beta1,i+1))

                #update parameters
                w[i] = w[i] - (learning_rate *m_w_dash / (np.sqrt(prev_w_hat)+epsilon))
                b[i] = b[i] - (learning_rate *m_b_dash / (np.sqrt(prev_b_hat)+epsilon))
                

        losses.append(epoch_loss)
        print('Epoch:', epoch, 'AvgLoss =', epoch_loss)

    return w, b, losses


# Define the layers of neural network of 3 hidden layers 128,128,128
layers = [784, 128, 128, 128, 10]

print("Gradient Descent")
w,b,loss_history=gradient_descent (x_Train, y_Train, layers, 0.2, 20)

test_accuracy(x_Test,y_Test,w,b)

print("Gradient Descent with Batch Size = 32")
w,b,loss_history=gradient_descent_with_batch_size (x_Train, y_Train, layers, 0.2, 5,32)

test_accuracy(x_Test,y_Test,w,b)

print("Gradient Descent with Batch Size=1, Stochastic GD")
w,b,loss_history=gradient_descent_with_batch_size (x_Train, y_Train, layers, 0.2, 2,1)

test_accuracy(x_Test,y_Test,w,b)

print("Momentum Gradient Descent with Batch Size")
w,b,loss_history=momentum_gradient_descent (x_Train, y_Train, layers, 0.2, 5,32,0.9)

test_accuracy(x_Test,y_Test,w,b)

print("Nesterov accelerated gradient descent")

w,b,loss_history=nesterov_gradient_descent (x_Train, y_Train, layers, 0.2, 5,32,0.9)

test_accuracy(x_Test,y_Test,w,b)

print("RMSProp algorithm")

w,b,loss_history=rmsProp (x_Train, y_Train, layers, 0.02, 5,32,0.9,0.0001)

test_accuracy(x_Test,y_Test,w,b)

print("Adam Adaptive Moments algorithm")
#x_Train, y_Train, layers, Learning rate, max epochs, batch size, beta1, beta2, epsilon) 
#β1 =0.9, β2=0.999

w,b,loss_history= adam(x_Train, y_Train, layers, 0.02, 5,32,0.9,0.999,0.0001)

test_accuracy(x_Test,y_Test,w,b)

print("Nesterov Adam Adaptive Moments algorithm")
#x_Train, y_Train, layers, Learning rate, max epochs, batch size, beta1, beta2, epsilon) 
#β1 =0.9, β2=0.999

w,b,loss_history= nadam(x_Train, y_Train, layers, 0.02, 5,32,0.9,0.999,0.0001)

test_accuracy(x_Test,y_Test,w,b)

