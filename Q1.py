!pip install wandb

import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

#7ea469daf0dd619314d4e3ba6f51b8c23ecfa982

wandb.init(project="Q1")

# loading the Fashion-MNIST dataset
(x_Train, y_Train), (x_Test, y_Test) = fashion_mnist.load_data()

# defining the 10 classes in the Fashion-MNIST dataset
class_Names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

images = []  # list to store image objects

#Create a grid of 10 subplots
fig, ax = plt.subplots(1, 10, figsize=(20, 20))

# select one image for each class and add to subplot
for i in range(10):
  for j in range (y_Train.shape[0]):
    if (y_Train[j]==i):
      img=x_Train[j]
      name=class_Names[y_Train[j]]
      # add img and its class to subplot
      ax[i].imshow(img, cmap='gray')
      ax[i].set_title(name)
      fig = plt.figure
      plt.title(name)
      # Remove ticks from img
      ax[i].set_xticks([])
      ax[i].set_yticks([])
      # Create an image object and append it to the list
      images.append(wandb.Image(img, caption=name))
      break
# Log all the images in a grid
wandb.log({"examples": images})
    
# Display the plot
plt.show()
