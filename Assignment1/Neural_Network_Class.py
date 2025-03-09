import numpy as np

class FeedforwardNeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, hidden_layers, learning_rate, weight_init_type, activation):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.activation = activation.lower()
        self.weight_init_type = weight_init_type.lower()
        

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
        Activation_layer = [input_data]
        Pre_activation_layer = [input_data]
        for W, B in zip(self.weights[:-1], self.biases[:-1]):  # Hidden layers
            Z = np.dot(A, W.T) + B
            Pre_activation_layer.append(Z)
            A = self.activate_layer(Z)
            Activation_layer.append(A)
        Z_output = np.dot(A, self.weights[-1].T) + self.biases[-1]
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
      
        gradient_output_layer = -(estimated_y - y_predicted)

        gradient_activation = gradient_output_layer

        while(layer>-1):
            #Current layer gradient w.r.t weights are outer product of gradient of activation and Activation value
            gradient_Weight_cur_layer = np.outer(gradient_activation, Activation_layer[layer])
            gradient_biases_cur_layer = gradient_activation

            gradient_Weights.append(gradient_Weight_cur_layer)
            gradient_biases.append(gradient_biases_cur_layer)

            if layer == 0:
                break
            else:
                #Gradient of preactivation is dot product of weight matric and consecutive layer's gradient_activation
                gradient_preactivation = np.dot(gradient_activation, self.weights[layer])
                gradient_activation = gradient_preactivation * self.gradient_activate(Pre_activation_layer[layer])

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


    

