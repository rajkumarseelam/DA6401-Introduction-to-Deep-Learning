
# This class contains all the Optimizers given in the assignment
# To Add any new optimization technique you can add the functionality in the class and also you can store the accuracy data in intiliazed lists which is more flexible..

import numpy as np

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





