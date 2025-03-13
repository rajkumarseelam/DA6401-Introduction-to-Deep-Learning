
from Neural_Network_Class import *
from Optimizations_Class import *

import argparse
import wandb
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def confusionMatrix(self):
    # on the test data set
    predictions = self.predict(self.X_test)
    if self.dataset == "fashion_mnist":
      class_names = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    else:
      class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    if self.isWandb == True:
      conf_matrix = confusion_matrix(self.Y_test, predictions)
      plt.figure(figsize=(10, 7))
      sns_heatmap = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                                xticklabels=class_names, yticklabels=class_names)
      plt.title('Confusion Matrix')
      plt.ylabel('True Label')
      plt.xlabel('Predicted Label')

      # Save the plot to an image file
      heatmap_image_filename = "confusion_matrix_heatmap.png"
      plt.savefig(heatmap_image_filename)
      plt.close()  # Close the plot to avoid displaying it in the notebook/output

      # Log the image to Wandb
      wandb.log({"confusion_matrix_custom": wandb.Image(heatmap_image_filename)})

    else:
      conf_matrix = confusion_matrix(self.Y_test, predictions)
      plt.figure(figsize=(10, 7))
      sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                  xticklabels= class_names,
                  yticklabels= class_names)
      plt.title('Confusion Matrix')
      plt.ylabel('True Label')
      plt.xlabel('Predicted Label')
      plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", "-we",help = "Wandb Entity used to track experiments in the Weights & Biases dashboard.", default="cs24m042")
    parser.add_argument("--wandb_project", "-wp",help="Project name used to track experiments in Weights & Biases dashboard", default="Assignment 1")
    parser.add_argument("--dataset", "-d", help = "dataset", choices=["mnist","fashion_mnist"], default="fashion_mnist")
    parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network", type= int, default=10)
    parser.add_argument("--batch_size","-b",help="Batch size used to train neural network", type =int, default=16)
    parser.add_argument("--optimizer","-o",help="batch size is used to train neural network", default= "sgd", choices=["sgd","momentum","nag","rmsprop","adam","nadam"])
    parser.add_argument("--loss","-l", default= "cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("--learning_rate","-lr", default=0.1, type=float)
    parser.add_argument("--momentum","-m", default=0.5,type=float)
    parser.add_argument("--beta","-beta", default=0.5, type=float)
    parser.add_argument("--beta1","-beta1", default=0.5,type=float)
    parser.add_argument("--beta2","-beta2", default=0.5,type=float)
    parser.add_argument("--epsilon","-eps",type=float, default = 0.000001)
    parser.add_argument("--weight_decay","-w_d", default=0.0,type=float)
    parser.add_argument("-w","--weight_init", default="random",choices=["random","xavier"])
    parser.add_argument("--num_layers","-nhl",type=int, default=1)
    parser.add_argument("--hidden_size","-sz",type=int, default=4)
    parser.add_argument("-a","--activation",choices=["identity","sigmoid","tanh","relu"], default="sigmoid")


    wandb.login()
    wandb.init(project=args.wandb_project,entity=args.wandb_project)
    model = NeuralNetwork(inputSize = 784, hiddenLayers = args.num_layers, 
                          outputSize = 10, sizeOfHiddenLayers = args.hidden_size, 
                          batchSize = args.batch_size, learningRate = args.learning_rate, 
                          initialisationType = args.weight_init, optimiser = args.optimizer, 
                          activationFunc=args.activation,weightDecay = args.weight_decay,
                          isWandb = True, lossFunc = args.loss, epochs = 10, dataset = args.dataset)
    
    wandb.finish()
