

import argparse
import wandb
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


#This class contains all the functions for implementation the Feed Forward Neural network..

class FeedforwardNeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, hidden_layers, weight_init_type, activation, loss_function_name, weight_decay):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers
        self.activation = activation.lower()
        self.loss_function_name = loss_function_name.lower()
        self.weight_init_type = weight_init_type.lower()
        self.weight_decay = weight_decay

        # Initialize Weights
        self.weights = [np.zeros((hidden_nodes, input_nodes))]
        for _ in range(hidden_layers - 1):
            self.weights.append(np.zeros((hidden_nodes, hidden_nodes)))
        self.weights.append(np.zeros((output_nodes, hidden_nodes)))

        # Initialize Biases
        self.biases = [np.zeros((1, hidden_nodes)) for _ in range(hidden_layers)]
        self.biases.append(np.zeros((1, output_nodes)))

        # Weights Initialization
        if self.weight_init_type == "random":
            for idx in range(len(self.weights)):
                self.weights[idx] = np.random.uniform(low=-0.2, high=0.2, size=self.weights[idx].shape)
        elif self.weight_init_type == "xavier":
            for idx in range(len(self.weights)):
                d0, d1 = self.weights[idx].shape
                self.weights[idx] = np.random.randn(d0, d1) * np.sqrt(2 / (d0 + d1))

    def activate_layer(self, x):
        if self.activation == "sigmoid":
            return self.sigmoid(x)
        elif self.activation == "tanh":
            return self.tanh(x)
        elif self.activation == "relu":
            return self.relu(x)

    # Activation functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        power = np.exp(x - np.max(x, axis=1, keepdims=True))  # Prevent overflow
        return power / np.sum(power, axis=1, keepdims=True)

    #Forward
    def forwardpropagation(self, input_data):
        A = input_data
        Activation_layer = [input_data]  #output of activation function "h"
        Pre_activation_layer = [] #weighted sum + bias "a"
        for W, B in zip(self.weights[:-1], self.biases[:-1]):  # Hidden layers
            Z = np.dot(A, W.T) + B
            Pre_activation_layer.append(Z)
            A = self.activate_layer(Z)
            Activation_layer.append(A)
        Z_output = np.dot(A, self.weights[-1].T) + self.biases[-1]
        Pre_activation_layer.append(Z_output)
        A_output = self.softmax(Z_output)
        return A_output, Activation_layer, Pre_activation_layer

    #Backward
    def back_propagation(self, Activation_layer, Pre_activation_layer, y_predicted, y_actual):
        gradient_Weights = []
        gradient_biases = []

        estimated_y = np.zeros(self.output_nodes)
        estimated_y[y_actual] = 1
        layer=self.hidden_layers
        # Gradient for Output Layer
        if self.loss_function_name == "mean_squared_error":
            gradient_output_layer = self.gradient_softmax(y_predicted) * self.gradient_mse_loss(y_predicted, estimated_y)
        elif self.loss_function_name == "cross_entropy":
            gradient_output_layer = -(estimated_y - y_predicted)

        gradient_preactivation = gradient_output_layer

        #Activation -"h" , Preactivation- "a"
        while(layer>-1):
            #Current layer gradient w.r.t weights are outer product of gradient of preactivation and Activation value
            gradient_Weight_cur_layer = np.outer(gradient_preactivation, Activation_layer[layer])
            gradient_biases_cur_layer = gradient_preactivation

            gradient_Weights.append(gradient_Weight_cur_layer)
            gradient_biases.append(gradient_biases_cur_layer)

            if layer == 0:
                break
            else:
                #Gradient of preactivation is dot product of weight matric and consecutive layer's gradient_preactivation
                gradient_activation = np.dot(gradient_preactivation, self.weights[layer])
                gradient_preactivation = gradient_activation * self.gradient_activate(Pre_activation_layer[layer-1]) #element wise multiplication

            layer-=1

        gradient_Weights.reverse()
        gradient_biases.reverse()
        return gradient_Weights, gradient_biases


    def gradient_activate(self, x):
        if self.activation == "sigmoid":
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        elif self.activation == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation == "relu":
            return np.where(x > 0, 1, 0)

    def evaluate_metrics(self, X, Y):
        correct_predictions = 0
        total_predictions = len(X)
        loss = 0
        for i in range(len(X)):
            y_predicted, _, _ = self.forwardpropagation(X[i].reshape(1, -1))
            predicted_class = np.argmax(y_predicted, axis=1)[0]
            if predicted_class == Y[i]:
                correct_predictions += 1
            loss += self.loss_function(y_predicted,Y[i])

        accuracy = correct_predictions / total_predictions
        loss /= total_predictions
        return accuracy, loss

    def predict(self, x):
        y_predicted, _, _ = self.forwardpropagation(x.reshape(1, -1))
        return np.argmax(y_predicted, axis=1)[0]

    def gradient_mse_loss(self, y_pred, y_actual):
        return  (y_pred - y_actual)


    def gradient_softmax(self, y_pred):
       return y_pred * (1 - y_pred)

    def gradient_weight_loss(self):
        gradient_weight = [self.weight_decay*W for W in self.weights]
        gradient_bias = [self.weight_decay*b for b in self.biases]
        return gradient_weight, gradient_bias


    def loss_function(self, y_predicted, y_actual):
        estimated_y = np.zeros(self.output_nodes)
        estimated_y[y_actual] = 1

        if self.loss_function_name == "mean_squared_error":
            return 0.5*np.sum((y_predicted - estimated_y) ** 2)
        elif self.loss_function_name == "cross_entropy":
            return -np.log(y_predicted[0][y_actual])

    def confusion_matrix(self, X, Y,dataset,plot=True):
        class_names=[]
        if dataset == "fashion_mnist":
            class_names = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
        else:
            class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        confusion_matrix = np.zeros((self.output_nodes, self.output_nodes), dtype=int)

        for i in range(len(X)):
            y_predicted, _, _ = self.forwardpropagation(X[i].reshape(1, -1))
            predicted_class = np.argmax(y_predicted, axis=1)[0]
            confusion_matrix[predicted_class, Y[i]] += 1

        if plot:
            plt.figure(figsize=(10, 7))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="Greens", xticklabels=class_names, yticklabels=class_names)
            plt.xlabel("Actual Label")
            plt.ylabel("Predicted Label")
            plt.title("Confusion Matrix")
            plt.show()
        else:
            plt.figure(figsize=(10, 7))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="Greens", xticklabels=class_names, yticklabels=class_names)
            plt.xlabel("Actual Label")
            plt.ylabel("Predicted Label")
            plt.title("Confusion Matrix")
            Img_name="confusion_matrix.png"
            plt.savefig(Img_name)
            plt.close()
            wandb.log({"confusion_matrix": wandb.Image(Img_name)})

        return confusion_matrix


# This class contains all the Optimizers given in the assignment
# To Add any new optimization technique you can add the functionality in the class and also you can store the accuracy data in intiliazed lists which is more flexible..

class Optimizers:
    def __init__(self, model):
        self.model = model
        self.train_accuracies=[]
        self.train_losses=[]
        self.val_accuracies=[]
        self.val_losses=[]

    def sgd(self, learning_rate, batch_size, epochs, Xtrain, Ytrain, Xval, Yval):
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.arange(len(Xtrain))
            np.random.shuffle(indices)
            Xtrain_shuffled = Xtrain[indices]
            Ytrain_shuffled = Ytrain[indices]

            # Mini-batch
            for i in range(0, len(Xtrain), batch_size):

                X_batch = Xtrain_shuffled[i:i + batch_size]
                Y_batch = Ytrain_shuffled[i:i + batch_size]


                gradient_weights = [np.zeros_like(w) for w in self.model.weights]
                gradient_biases = [np.zeros_like(b) for b in self.model.biases]


                for j in range(len(X_batch)):
                    # Forward propagation
                    y_predicted, Activation_layer, Pre_activation = self.model.forwardpropagation(X_batch[j].reshape(1, -1))
                    # Backward propagation
                    propagated_Weights, propagated_biases = self.model.back_propagation(Activation_layer, Pre_activation, y_predicted, Y_batch[j])

                    # Accumulate gradients
                    model_weights_loss, model_biases_loss = self.model.gradient_weight_loss()
                    for k in range(len(propagated_Weights)):
                        gradient_weights[k] += propagated_Weights[k]+model_weights_loss[k]
                        gradient_biases[k] += propagated_biases[k]+model_biases_loss[k]

                for k in range(len(self.model.weights)):
                    # Update velocity for weights and biases
                    gradient_weights[k] /= len(X_batch)
                    gradient_biases[k] /= len(X_batch)

                    self.model.weights[k] -= learning_rate * gradient_weights[k]
                    self.model.biases[k] -= learning_rate * gradient_biases[k]


            train_acc, train_loss = self.model.evaluate_metrics(Xtrain, Ytrain)
            val_acc, val_loss = self.model.evaluate_metrics(Xval, Yval)

            print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}")


            self.train_accuracies.append(train_acc)
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)

    def momentum_sgd(self, learning_rate, batch_size, epochs, momentum, Xtrain, Ytrain, Xval, Yval):
        momentum_weights = [np.zeros_like(w) for w in self.model.weights]
        momentum_biases = [np.zeros_like(b) for b in self.model.biases]

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.arange(len(Xtrain))
            np.random.shuffle(indices)
            Xtrain_shuffled = Xtrain[indices]
            Ytrain_shuffled = Ytrain[indices]

            # Mini-batch
            for i in range(0, len(Xtrain), batch_size):

                X_batch = Xtrain_shuffled[i:i + batch_size]
                Y_batch = Ytrain_shuffled[i:i + batch_size]


                gradient_weights = [np.zeros_like(w) for w in self.model.weights]
                gradient_biases = [np.zeros_like(b) for b in self.model.biases]


                for j in range(len(X_batch)):
                    # Forward propagation
                    y_predicted, Activation_layer, Pre_activation = self.model.forwardpropagation(X_batch[j].reshape(1, -1))
                    # Backward propagation
                    propagated_Weights, propagated_biases = self.model.back_propagation(Activation_layer, Pre_activation, y_predicted, Y_batch[j])

                    # Accumulate gradients
                    model_weights_loss, model_biases_loss = self.model.gradient_weight_loss()
                    for k in range(len(propagated_Weights)):
                        gradient_weights[k] += propagated_Weights[k]+model_weights_loss[k]
                        gradient_biases[k] += propagated_biases[k]+model_biases_loss[k]

                for k in range(len(self.model.weights)):
                    # Update velocity for weights and biases
                    gradient_weights[k] /= len(X_batch)
                    gradient_biases[k] /= len(X_batch)

                    momentum_weights[k] = momentum * momentum_weights[k] + learning_rate*gradient_weights[k]
                    momentum_biases[k] = momentum * momentum_biases[k] + learning_rate*gradient_biases[k]

                    self.model.weights[k] -=momentum_weights[k]
                    self.model.biases[k] -= momentum_biases[k]


            train_acc, train_loss = self.model.evaluate_metrics(Xtrain, Ytrain)
            val_acc, val_loss = self.model.evaluate_metrics(Xval, Yval)

            print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}")


            self.train_accuracies.append(train_acc)
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)


    def nestrov(self, learning_rate, batch_size, epochs, momentum, Xtrain, Ytrain, Xval, Yval):
        momentum_weights = [np.zeros_like(w) for w in self.model.weights]
        momentum_biases = [np.zeros_like(b) for b in self.model.biases]

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.arange(len(Xtrain))
            np.random.shuffle(indices)
            Xtrain_shuffled = Xtrain[indices]
            Ytrain_shuffled = Ytrain[indices]

            # Mini-batch
            for i in range(0, len(Xtrain), batch_size):

                X_batch = Xtrain_shuffled[i:i + batch_size]
                Y_batch = Ytrain_shuffled[i:i + batch_size]

                lookahead_weights = [w - momentum * mw for w, mw in zip(self.model.weights, momentum_weights)]
                lookahead_biases = [b - momentum * mb for b, mb in zip(self.model.biases, momentum_biases)]

                original_weights = [w.copy() for w in self.model.weights]
                original_biases = [b.copy() for b in self.model.biases]
                self.model.weights = lookahead_weights
                self.model.biases = lookahead_biases

                # Compute gradients at the lookahead position
                gradient_weights = [np.zeros_like(w) for w in self.model.weights]
                gradient_biases = [np.zeros_like(b) for b in self.model.biases]

                for j in range(len(X_batch)):
                    # Forward propagation using lookahead weights
                    y_predicted, Activation_layer, Pre_activation = self.model.forwardpropagation(X_batch[j].reshape(1, -1))

                    # Backward propagation
                    propagated_Weights, propagated_biases = self.model.back_propagation(Activation_layer, Pre_activation, y_predicted, Y_batch[j])

                    model_weights_loss, model_biases_loss = self.model.gradient_weight_loss()

                    for k in range(len(propagated_Weights)):
                        gradient_weights[k] += propagated_Weights[k] + model_weights_loss[k]
                        gradient_biases[k] += propagated_biases[k] + model_biases_loss[k]

                #Restoring the values
                self.model.weights = original_weights
                self.model.biases = original_biases

                for k in range(len(self.model.weights)):
                    gradient_weights[k] /= len(X_batch)
                    gradient_biases[k] /= len(X_batch)

                    # prev_momentum_weights = momentum_weights[k]
                    # prev_momentum_biases = momentum_biases[k]

                    momentum_weights[k] = momentum * momentum_weights[k] + learning_rate * gradient_weights[k]
                    momentum_biases[k] = momentum * momentum_biases[k] +learning_rate * gradient_biases[k]

                    self.model.weights[k] -= momentum_weights[k]
                    self.model.biases[k] -= momentum_biases[k]

            train_acc, train_loss = self.model.evaluate_metrics(Xtrain, Ytrain)
            val_acc, val_loss = self.model.evaluate_metrics(Xval, Yval)

            print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}")

            self.train_accuracies.append(train_acc)
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)




    def rmsprop(self, learning_rate, batch_size, epochs, beta, epsilon, Xtrain, Ytrain, Xval, Yval):

        v_weights = [np.zeros_like(w) for w in self.model.weights]
        v_biases = [np.zeros_like(b) for b in self.model.biases]

        for epoch in range(epochs):

            indices = np.arange(len(Xtrain))
            np.random.shuffle(indices)
            Xtrain_shuffled = Xtrain[indices]
            Ytrain_shuffled = Ytrain[indices]

            for i in range(0, len(Xtrain), batch_size):
                # Get mini-batch
                X_batch = Xtrain_shuffled[i:i + batch_size]
                Y_batch = Ytrain_shuffled[i:i + batch_size]


                gradient_weights = [np.zeros_like(w) for w in self.model.weights]
                gradient_biases = [np.zeros_like(b) for b in self.model.biases]


                for j in range(len(X_batch)):
                    # Forward propagation
                    y_predicted, Activation_layer, Pre_activation = self.model.forwardpropagation(X_batch[j].reshape(1, -1))
                    # Backward propagation
                    propagated_Weights, propagated_biases = self.model.back_propagation(Activation_layer, Pre_activation, y_predicted, Y_batch[j])


                    model_weights_loss, model_biases_loss = self.model.gradient_weight_loss()
                    for k in range(len(propagated_Weights)):
                        gradient_weights[k] += propagated_Weights[k]+model_weights_loss[k]
                        gradient_biases[k] += propagated_biases[k]+model_biases_loss[k]


                for k in range(len(self.model.weights)):
                    gradient_weights[k] /= len(X_batch)
                    gradient_biases[k] /= len(X_batch)

                    v_weights[k] = beta * v_weights[k] + (1 - beta) * (gradient_weights[k] ** 2)
                    v_biases[k] = beta * v_biases[k] + (1 - beta) * (gradient_biases[k] ** 2)

                    self.model.weights[k] -= learning_rate * gradient_weights[k] / (np.sqrt(v_weights[k]) + epsilon)
                    self.model.biases[k] -= learning_rate * gradient_biases[k] / (np.sqrt(v_biases[k]) + epsilon)

            train_acc, train_loss = self.model.evaluate_metrics(Xtrain, Ytrain)
            val_acc, val_loss = self.model.evaluate_metrics(Xval, Yval)

            print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}")


            self.train_accuracies.append(train_acc)
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)


    def adam(self, learning_rate, batch_size, epochs, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval):
        m_weights = [np.zeros_like(w) for w in self.model.weights]
        m_biases = [np.zeros_like(b) for b in self.model.biases]
        v_weights = [np.zeros_like(w) for w in self.model.weights]
        v_biases = [np.zeros_like(b) for b in self.model.biases]

        for epoch in range(epochs):
            indices = np.arange(len(Xtrain))
            np.random.shuffle(indices)
            Xtrain_shuffled = Xtrain[indices]
            Ytrain_shuffled = Ytrain[indices]

            for i in range(0, len(Xtrain), batch_size):
                X_batch = Xtrain_shuffled[i:i + batch_size]
                Y_batch = Ytrain_shuffled[i:i + batch_size]

                gradient_weights = [np.zeros_like(w) for w in self.model.weights]
                gradient_biases = [np.zeros_like(b) for b in self.model.biases]

                for j in range(len(X_batch)):
                    # Forward propagation
                    y_predicted, Activation_layer, Pre_activation = self.model.forwardpropagation(X_batch[j].reshape(1, -1))
                    # Backward propagation
                    propagated_Weights, propagated_biases = self.model.back_propagation(Activation_layer, Pre_activation, y_predicted, Y_batch[j])


                    model_weights_loss, model_biases_loss = self.model.gradient_weight_loss()
                    for k in range(len(propagated_Weights)):
                        gradient_weights[k] += propagated_Weights[k]+model_weights_loss[k]
                        gradient_biases[k] += propagated_biases[k]+model_biases_loss[k]


                for k in range(len(self.model.weights)):
                    gradient_weights[k] /= len(X_batch)
                    gradient_biases[k] /= len(X_batch)

                    m_weights[k] = beta1 * m_weights[k] + (1 - beta1) * gradient_weights[k]
                    m_biases[k] = beta1 * m_biases[k] + (1 - beta1) * gradient_biases[k]
                    v_weights[k] = beta2 * v_weights[k] + (1 - beta2) * (gradient_weights[k] ** 2)
                    v_biases[k] = beta2 * v_biases[k] + (1 - beta2) * (gradient_biases[k] ** 2)

                    # Bias correction
                    m_weights_hat = m_weights[k] / (1 - beta1 ** (epoch + 1))
                    m_biases_hat = m_biases[k] / (1 - beta1 ** (epoch + 1))
                    v_weights_hat = v_weights[k] / (1 - beta2 ** (epoch + 1))
                    v_biases_hat = v_biases[k] / (1 - beta2 ** (epoch + 1))

                    # Update weights and biases
                    self.model.weights[k] -= learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + epsilon)
                    self.model.biases[k] -= learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + epsilon)

            train_acc, train_loss = self.model.evaluate_metrics(Xtrain, Ytrain)
            val_acc, val_loss = self.model.evaluate_metrics(Xval, Yval)

            print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}")

            self.train_accuracies.append(train_acc)
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)

   



    def train_model(self,optimizer, learning_rate, batch_size, epochs, momentum, beta, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval):
        if optimizer == "sgd":
            self.sgd( learning_rate, batch_size, epochs, Xtrain, Ytrain, Xval, Yval)
        elif optimizer == "momentum":
             self.momentum_sgd(learning_rate, batch_size, epochs, momentum, Xtrain, Ytrain, Xval, Yval)
        elif optimizer == "nestrov":
             self.nestrov(learning_rate, batch_size, epochs, momentum, Xtrain, Ytrain, Xval, Yval)
        elif optimizer == "rmsprop":
            self.rmsprop(learning_rate, batch_size, epochs, beta, epsilon, Xtrain, Ytrain, Xval, Yval)
        elif optimizer == "adam":
           self.adam(learning_rate, batch_size, epochs, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval)

        return self.model.weights,self.model.biases, self.train_accuracies,self.train_losses, self.val_accuracies,self.val_losses





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", "-wp",help="Project name used to track experiments in Weights & Biases dashboard", default="DA6401-Assignment-1")
    parser.add_argument("--wandb_entity", "-we",help = "Wandb Entity used to track experiments in the Weights & Biases dashboard.", default="cs24m042-iit-madras-foundation")
    parser.add_argument("--dataset", "-d", help = "dataset", choices=["mnist","fashion_mnist"], default="fashion_mnist")
    parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network", type= int, default=10)
    parser.add_argument("--batch_size","-b",help="Batch size used to train neural network", type =int, default=16)
    parser.add_argument("--optimizer","-o",help="batch size is used to train neural network", default= "adam", choices=["sgd","momentum","nestrov","rmsprop","adam"])
    parser.add_argument("--loss","-l", default= "cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("--learning_rate","-lr", default=0.001, type=float)
    parser.add_argument("--momentum","-m", default=0.9,type=float)
    parser.add_argument("--beta","-beta", default=0.9, type=float)
    parser.add_argument("--beta1","-beta1", default=0.9,type=float)
    parser.add_argument("--beta2","-beta2", default=0.999,type=float)
    parser.add_argument("--epsilon","-eps",type=float, default = 1e-8)
    parser.add_argument("--weight_decay","-w_d", default=0.0,type=float)
    parser.add_argument("-w","--weight_init", default="random",choices=["random","xavier"])
    parser.add_argument("--num_layers","-nhl",type=int, default=5)
    parser.add_argument("--hidden_size","-sz",type=int, default=128)
    parser.add_argument("-a","--activation",choices=["sigmoid","tanh","relu"], default="tanh")

    args = parser.parse_args()

    wandb.login()
    wandb.init(project=args.wandb_project)
    if(args.dataset=="fashion_mnist"):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    train_val_split = int(0.9 * len(x_train))
    Xtrain, Ytrain = x_train[:train_val_split], y_train[:train_val_split]
    Xval, Yval = x_train[train_val_split:], y_train[train_val_split:]
    Xtest, Ytest = x_test, y_test

    print("Train Data Dimensions : ", Xtrain.shape)
    print("Validation Data Dimensions :", Xval.shape)
    print("Test Data Dimensions :", Xtest.shape)

    model = FeedforwardNeuralNetwork(input_nodes=784, hidden_nodes=args.hidden_size, output_nodes=10, hidden_layers=args.num_layers, weight_init_type=args.weight_init, activation=args.activation, loss_function_name=args.loss, weight_decay=args.weight_decay)
    optimizer = Optimizers(model)
    weights, biases, train_accuracy, train_loss, val_accuracy, val_loss = optimizer.train_model(args.optimizer,args.learning_rate,args.batch_size,args.epochs,args.momentum, args.beta, args.beta1,args.beta2, args.epsilon, Xtrain, Ytrain, Xval, Yval)
    
    for epoch in range(args.epochs):
            wandb.log({
                'train_accuracy': train_accuracy[epoch]*100,
                'train_loss': train_loss[epoch],
                'val_accuracy': val_accuracy[epoch]*100,
                'val_loss': val_loss[epoch],
                'epoch' : epoch
            })

    model.confusion_matrix(Xtest,Ytest,args.dataset,False) #False indicates plot in wandb
    wandb.finish()
