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

                
                gradient_weights = [np.zeros_like(w) for w in self.model.weights]
                gradient_biases = [np.zeros_like(b) for b in self.model.biases]

                
                for j in range(len(X_batch)):
                    # Forward propagation
                    y_predicted, Activation_layer, Pre_activation = self.model.forwardpropagation(X_batch[j].reshape(1, -1))
                    # Backward propagation
                    grad_W, grad_B = self.model.back_propagation(Activation_layer, Pre_activation, y_predicted, Y_batch[j])

                    # Accumulate gradients
                    for k in range(len(grad_W)):
                        gradient_weights[k] += grad_W[k]
                        gradient_biases[k] += grad_B[k]

                # Average gradients over the batch
                for k in range(len(gradient_weights)):
                    gradient_weights[k] /= len(X_batch)
                    gradient_biases[k] /= len(X_batch)

                # Update weights and biases using NAG
                for i in range(len(self.model.weights)):
                    # Update velocity for weights and biases
                    momentum_weights[i] = momentum * momentum_weights[i] + gradient_weights[i]
                    momentum_biases[i] = momentum * momentum_biases[i] + gradient_biases[i]
                    
                    
                    self.model.weights[i] -= learning_rate * (momentum * momentum_weights[i] + gradient_weights[i])
                    self.model.biases[i] -= learning_rate * (momentum * momentum_biases[i] + gradient_biases[i])

            
            train_acc, train_loss = self.model.evaluate_metrics(Xtrain, Ytrain)
            val_acc, val_loss = self.model.evaluate_metrics(Xval, Yval)

            print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}")

            
            self.train_accuracies.append(train_acc)
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)

        return self.model.weights, self.model.biases, self.train_accuracies, self.train_losses, self.val_accuracies, self.val_losses


    def rmsprop(self, learning_rate, batch_size, epochs, beta, epsilon, Xtrain, Ytrain, Xval, Yval):
        
        local_weights = [np.zeros_like(w) for w in self.model.weights]
        local_biases = [np.zeros_like(b) for b in self.model.biases]

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
                    grad_W, grad_B = self.model.back_propagation(Activation_layer, Pre_activation, y_predicted, Y_batch[j])

                    
                    for k in range(len(grad_W)):
                        gradient_weights[k] += grad_W[k]
                        gradient_biases[k] += grad_B[k]

               
                for k in range(len(gradient_weights)):
                    gradient_weights[k] /= len(X_batch)
                    gradient_biases[k] /= len(X_batch)

                
                for k in range(len(self.model.weights)):
                    local_weights[k] = beta * local_weights[k] + (1 - beta) * (gradient_weights[k] ** 2)
                    local_biases[k] = beta * local_biases[k] + (1 - beta) * (gradient_biases[k] ** 2)
                    self.model.weights[k] -= learning_rate * gradient_weights[k] / (np.sqrt(local_weights[k]) + epsilon)
                    self.model.biases[k] -= learning_rate * gradient_biases[k] / (np.sqrt(local_biases[k]) + epsilon)

            train_acc, train_loss = self.model.evaluate_metrics(Xtrain, Ytrain)
            val_acc, val_loss = self.model.evaluate_metrics(Xval, Yval)

            print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}")

           
            self.train_accuracies.append(train_acc)
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)

        return self.model.weights, self.model.biases, self.train_accuracies, self.train_losses, self.val_accuracies, self.val_losses

    def adam(self, learning_rate, batch_size, epochs, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval):
        m_w = [np.zeros_like(w) for w in self.model.weights]
        m_b = [np.zeros_like(b) for b in self.model.biases]
        v_w = [np.zeros_like(w) for w in self.model.weights]
        v_b = [np.zeros_like(b) for b in self.model.biases]

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
                    grad_W, grad_B = self.model.back_propagation(Activation_layer, Pre_activation, y_predicted, Y_batch[j])


                    for k in range(len(grad_W)):
                        gradient_weights[k] += grad_W[k]
                        gradient_biases[k] += grad_B[k]

                for k in range(len(gradient_weights)):
                    gradient_weights[k] /= len(X_batch)
                    gradient_biases[k] /= len(X_batch)


                for k in range(len(self.model.weights)):
                    m_w[k] = beta1 * m_w[k] + (1 - beta1) * gradient_weights[k]
                    m_b[k] = beta1 * m_b[k] + (1 - beta1) * gradient_biases[k]
                    v_w[k] = beta2 * v_w[k] + (1 - beta2) * (gradient_weights[k] ** 2)
                    v_b[k] = beta2 * v_b[k] + (1 - beta2) * (gradient_biases[k] ** 2)

                    # Bias correction
                    m_w_hat = m_w[k] / (1 - beta1 ** (epoch + 1))
                    m_b_hat = m_b[k] / (1 - beta1 ** (epoch + 1))
                    v_w_hat = v_w[k] / (1 - beta2 ** (epoch + 1))
                    v_b_hat = v_b[k] / (1 - beta2 ** (epoch + 1))

                    # Update weights and biases
                    self.model.weights[k] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                    self.model.biases[k] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

            train_acc, train_loss = self.model.evaluate_metrics(Xtrain, Ytrain)
            val_acc, val_loss = self.model.evaluate_metrics(Xval, Yval)

            print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}")

            self.train_accuracies.append(train_acc)
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)

        return self.model.weights, self.model.biases, self.train_accuracies, self.train_losses, self.val_accuracies, self.val_losses

    def nadam(self, learning_rate, batch_size, epochs, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval):
        m_w = [np.zeros_like(w) for w in self.model.weights]
        m_b = [np.zeros_like(b) for b in self.model.biases]
        v_w = [np.zeros_like(w) for w in self.model.weights]
        v_b = [np.zeros_like(b) for b in self.model.biases]

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
                    grad_W, grad_B = self.model.back_propagation(Activation_layer, Pre_activation, y_predicted, Y_batch[j])

                    for k in range(len(grad_W)):
                        gradient_weights[k] += grad_W[k]
                        gradient_biases[k] += grad_B[k]

                for k in range(len(gradient_weights)):
                    gradient_weights[k] /= len(X_batch)
                    gradient_biases[k] /= len(X_batch)

                for k in range(len(self.model.weights)):
                    m_w[k] = beta1 * m_w[k] + (1 - beta1) * gradient_weights[k]
                    m_b[k] = beta1 * m_b[k] + (1 - beta1) * gradient_biases[k]
                    v_w[k] = beta2 * v_w[k] + (1 - beta2) * (gradient_weights[k] ** 2)
                    v_b[k] = beta2 * v_b[k] + (1 - beta2) * (gradient_biases[k] ** 2)

                    # Bias correction
                    m_w_hat = m_w[k] / (1 - beta1 ** (epoch + 1))
                    m_b_hat = m_b[k] / (1 - beta1 ** (epoch + 1))
                    v_w_hat = v_w[k] / (1 - beta2 ** (epoch + 1))
                    v_b_hat = v_b[k] / (1 - beta2 ** (epoch + 1))

                    # Nesterov momentum
                    m_w_hat_nadam = beta1 * m_w_hat + (1 - beta1) * gradient_weights[k]
                    m_b_hat_nadam = beta1 * m_b_hat + (1 - beta1) * gradient_biases[k]

                    # Update weights and biases
                    self.model.weights[k] -= learning_rate * m_w_hat_nadam / (np.sqrt(v_w_hat) + epsilon)
                    self.model.biases[k] -= learning_rate * m_b_hat_nadam / (np.sqrt(v_b_hat) + epsilon)

            train_acc, train_loss = self.model.evaluate_metrics(Xtrain, Ytrain)
            val_acc, val_loss = self.model.evaluate_metrics(Xval, Yval)

            print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}")

            self.train_accuracies.append(train_acc)
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)

        return self.model.weights, self.model.biases, self.train_accuracies, self.train_losses, self.val_accuracies, self.val_losses




