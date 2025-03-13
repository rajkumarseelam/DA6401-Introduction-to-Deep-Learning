#Tuning hyper parameters for my model on fashion_mnist dataset using the help of wandb Config..

from Neural_Network_Class import *
from Optimizations_Class import *

import numpy as np
from keras.datasets import fashion_mnist
import wandb

wandb.login()

(x_train_original, y_train_original), (x_test_original, y_test_original) = fashion_mnist.load_data()

# Normalize and reshape the images
x_train = x_train_original.reshape(x_train_original.shape[0], -1) / 255.0  # (60000, 784)
y_train = np.copy(y_train_original)
x_test = x_test_original.reshape(x_test_original.shape[0], -1) / 255.0  # (10000, 784)
y_test = np.copy(y_test_original)

# Split train into train + validation
train_val_split = 0.9
total_samples = x_train.shape[0]
split_index = int(train_val_split * total_samples)

# Shuffle indices for randomness
indices = np.arange(total_samples)
np.random.shuffle(indices)
x_train, y_train = x_train[indices], y_train[indices]

Xtrain, Ytrain = x_train[:split_index], y_train[:split_index]
Xval, Yval = x_train[split_index:], y_train[split_index:]
Xtest, Ytest = x_test, y_test

#Default parameters
input_size=28*28
output_size=10
momentum=0.9
beta=0.9
beta1=0.9
beta2=0.999
epsilon=1e-8
loss="cross_entropy"
project_name = "DA6401-Assignment-1"


sweep_configuration = {
    'name' : 'hyper_parameters_test',
    'method' : 'bayes',
    'metric': {'name' : 'val_loss', 'goal' : 'minimize'},
    'parameters' : {
        'epochs': { "values" : [5, 10] },
        'num_layers': { "values" : [3, 4, 5] },
        'hidden_size': { "values" : [32, 64, 128] },
        'weight_decay': { "values" : [0, 0.0005, 0.5] },
        'learning_rate': { "values" : [1e-3, 1e-4] },
        'optimizer' : { "values" : ["sgd", "momentum", "nestrov", "rmsprop", "adam"] },
        'batch_size' : { "values" : [16, 32, 64] },
        'weight_init' : { "values" : ["random", "Xavier"] },
        'activation' : { "values" : ["sigmoid", "tanh", "ReLU"] }
    }
}

sweep_id = wandb.sweep(sweep_configuration, project = project_name)


def wandb_connect():
    with wandb.init(config = config, project = project_name) as run:
        config = wandb.config

        # assign name of run
        run.name = "epochs {} hidden_layers {} hidden_size {} learning_rate {} opt {} batch_size {} init {} activation {} weight_decay {}".format(
        config.epochs, config.num_layers, config.hidden_size, config.learning_rate, config.optimizer, config.batch_size, config.weight_init, config.activation, config.weight_decay)

        # proceed with the run
        model = FeedforwardNeuralNetwork(input_size,config['hidden_size'],output_size,config['num_layers'],config['weight_init'],config['activation'],loss,config['weight_decay'])
        optimizer_call = Optimizers(model)
        model.weights, model.bias, train_acc, train_loss, val_acc, val_loss = optimizer_call.train_model(config['optimizer'], config['learning_rate'], config['batch_size'],config['epochs'], momentum, beta, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval)
        for epoch in range(config['epochs']):
            wandb.log({
                'train_accuracy': train_acc[epoch] *100,
                'train_loss': train_loss[epoch],
                'val_accuracy': val_acc[epoch] *100,
                'val_loss': val_loss[epoch],
                'epoch' : epoch
            })

agent = wandb.agent(sweep_id, function = wandb_connect, project = project_name, count = 50)
wandb.finish()

