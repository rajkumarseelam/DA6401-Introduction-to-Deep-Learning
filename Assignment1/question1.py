pip install wandb

"""# Question 1

Importing the data from the fashion_mnist data set and selecting one image from each class.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


#There are 10 class labels present in mnist data set.
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


wandb.init(project="DA6401-Assignment-1", name="mnist_run", reinit=True)

#Loading data from the fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#storing images in Image_set
images_to_log = []
for class_index in range(10):

    #Find the instance of each class and append it .
    idx = np.where(train_labels == class_index)[0][0]


    image = train_images[idx]
    wandb_image = wandb.Image(image, caption=class_labels[class_index])
    images_to_log.append(wandb_image)

# Log the image to WandB
wandb.log({"Question1": images_to_log})

wandb.finish()