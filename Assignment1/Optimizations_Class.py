class Optimizers:
    def __init__(self, model):
        self.model = model
        self.train_accuracies=[]
        self.train_losses=[]
        self.val_accuracies=[]
        self.val_losses=[]

    def sgd(self, learning_rate, batch_size, epochs, Xtrain, Ytrain, Xval, Yval):
        for epoch in range(epochs):
            gradient_Weights = [np.zeros_like(w) for w in self.model.weights]
            gradient_biases = [np.zeros_like(bias) for bias in self.model.biases]

            indices = np.arange(len(Xtrain))
            np.random.shuffle(indices)

            
            for i in range(len(Xtrain)):
 
                y_predicted,Activation_layer, Pre_activation = self.model.forwardpropagation(Xtrain[indices[i]])
                propagated_Weights, Propagated_biases = self.model.back_propagation(Activation_layer, Pre_activation, y_predicted, Ytrain[indices[i]])
                for j in range(len(propagated_Weights)):
                    gradient_Weights[j] += propagated_Weights[j]
                    gradient_biases[j] += Propagated_biases[j]
                    
                # Performing update when batch is completed
                if (i + 1) % batch_size == 0 or i == len(Xtrain) - 1:

                    #L2 Regularization
                    self.model.weights_loss, self.model.biases_loss = self.model.gradient_weight_loss()

                    gradient_Weights/=batch_size
                    gradient_biases/=batch_size

                    for j in range(len(gradient_Weights)):
                        self.model.weights[j] -= learning_rate * (gradient_Weights[j] + self.model.weights_loss[j])
                        self.model.biases[j] -= learning_rate * (gradient_biases[j] + self.model.biases_loss[j])

                    #Reset gradients
                    gradient_Weights = [np.zeros_like(w) for w in gradient_Weights]
                    gradient_biases = [np.zeros_like(bias) for bias in gradient_biases]

            train_acc, train_loss = self.model.evaluate_metrics(Xtrain, Ytrain)
            val_acc, val_loss = self.model.evaluate_metrics(Xval, Yval)

            print(f"Epoch {epoch + 1}: Train Acc = {train_acc}, Val Acc = {val_acc}")

            self.train_accuracies.append(train_acc)
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)

        return self.model.weights, self.model.biases , self.train_accuracies, self.train_losses, self.val_accuracies, self.val_losses


  


    def momentum_sgd(self, learning_rate, batch_size, epochs, momentum, Xtrain, Ytrain, Xval, Yval):
        momentum_weights = [np.zeros_like(w) for w in self.model.weights]
        momentum_biases = [np.zeros_like(b) for b in self.model.biases]

        for epoch in range(epochs):
            indices = np.arange(len(Xtrain))
            np.random.shuffle(indices)

            gradient_Weights = [np.zeros_like(w) for w in self.model.weights]
            gradient_biases = [np.zeros_like(b) for b in self.model.biases]

            for i in range(len(Xtrain)):  
                index = indices[i]

                y_predicted, Activation_layer, Pre_activation = self.model.forwardpropagation(Xtrain[index])
                propagated_Weights, Propagated_biases = self.model.back_propagation(Activation_layer, Pre_activation, y_predicted, Ytrain[index])

                for j in range(len(propagated_Weights)):
                    gradient_Weights[j] += propagated_Weights[j]
                    gradient_biases[j] += Propagated_biases[j]

                # Performing update when batch is completed
                if (i + 1) % batch_size == 0 or i == len(Xtrain) - 1:
                    # L2 regularization
                    Model_weights_loss, Model_biases_loss = self.model.gradient_weight_loss()

                    for j in range(len(gradient_Weights)):
                        gradient_Weights[j] /= batch_size
                        gradient_biases[j] /= batch_size

                        # Momentum Updates
                        momentum_weights[j] = momentum * momentum_weights[j] + (gradient_Weights[j] + Model_weights_loss[j])
                        momentum_biases[j] = momentum * momentum_biases[j] + (gradient_biases[j] + Model_biases_loss[j])

                        # Storing back
                        self.model.weights[j] -= learning_rate * momentum_weights[j]
                        self.model.biases[j] -= learning_rate * momentum_biases[j]

                    # Reset gradients
                    gradient_Weights = [np.zeros_like(w) for w in self.model.weights]
                    gradient_biases = [np.zeros_like(b) for b in self.model.biases]

            
            train_acc, train_loss = self.model.evaluate_metrics(Xtrain, Ytrain)
            val_acc, val_loss = self.model.evaluate_metrics(Xval, Yval)

            print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")


            self.train_accuracies.append(train_acc)
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)

        return self.model.weights, self.model.biases, self.train_accuracies, self.train_losses, self.val_accuracies, self.val_losses



