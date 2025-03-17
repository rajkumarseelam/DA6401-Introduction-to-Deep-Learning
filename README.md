## GitHub Repository
You can find the complete project on [GitHub](https://github.com/rajkumarseelam/DA6401-Introduction-to-Deep-Learning.git).

## Wandb Report Link
You can find the Report on [WandB Report](https://wandb.ai/cs24m042-iit-madras-foundation/DA6401-Assignment-1/reports/DA6401-Assignment-1--VmlldzoxMTgyMDMzNQ).

# Feedforward Neural Network with Various Optimizers

## Overview
This project implements a feedforward neural network from scratch using NumPy. The network supports multiple hidden layers, various weight initialization techniques, activation functions, and loss functions. It also integrates multiple optimization techniques for training. The model can be trained on the MNIST and Fashion-MNIST datasets.

## Features
- Implements a customizable Feedforward Neural Network
- Supports activation functions: Sigmoid, Tanh, ReLU
- Supports loss functions: Mean Squared Error, Cross Entropy
- Includes multiple optimizers:
  - Stochastic Gradient Descent (SGD)
  - Momentum-based SGD
  - Nesterov Accelerated Gradient (NAG)
  - RMSprop
  - Adam
- Uses Weights & Biases (WandB) for experiment tracking
- Generates confusion matrix plots

## Code Organization
I have added all the individual work to `train.py` file.

Our `train.py` contains two classes:
- `FeedforwardNeuralNetwork`
- `Optimizer`

The `FeedforwardNeuralNetwork` class contains functions for forward pass, backward pass, required gradient functions, and confusion matrix generation for test data.

The `Optimizer` class contains all the optimizer functionalities. A `FeedforwardNeuralNetwork` object is passed to implement the selected optimization technique.

Creating a model object and intializing the optimizers class
```bash
model = FeedforwardNeuralNetwork(input_nodes, hidden_nodes, output_nodes, hidden_layers, weight_init_type, activation, loss_function_name, weight_decay)
optimizer = Optimizers(model)
```

Now you can choose your required optimizer and pass the necessary values.
```bash
weights, biases, train_accuracy, train_loss, val_accuracy, val_loss = optimizer.train_model(optimizer, learning_rate, batch_size, epochs, momentum, beta, beta1, beta2, epsilon, Xtrain, Ytrain, Xval, Yval)

```
For confusion matrix for testdata
```bash
model.confusion_matrix(X, Y,dataset)  # for printing the confusion matrix
```

`train_accuracy`, `train_loss`, `val_accuracy`, `val_loss` - these are lists that contain their respective values at each epoch and are used in generating reports.

## Usage
### Running the Training Script
Execute the training script with default parameters:
```bash
python train.py
```

### Command-line Arguments
You can customize the training process using the following arguments:
```bash
python train.py --dataset fashion_mnist --epochs 20 --batch_size 32 --optimizer adam --learning_rate 0.001 --activation relu
```

### Argument Details
- `--dataset`: Choose between `mnist` or `fashion_mnist`
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 16)
- `--optimizer`: Select an optimizer from [`sgd`, `momentum`, `nestrov`, `rmsprop`, `adam`]
- `--learning_rate`: Learning rate for optimization (default: 0.001)
- `--activation`: Choose activation function from [`sigmoid`, `tanh`, `relu`]
- `--loss`: Choose loss function from [`mean_squared_error`, `cross_entropy`]
- `--num_layers`: Number of hidden layers (default: 5)
- `--hidden_size`: Number of neurons per hidden layer (default: 128)
- `--weight_init`: Weight initialization method (`random`, `xavier`)
- `--wandb_project`: Project name for logging results in WandB

## Output
- Training and validation accuracy/loss for each epoch
- Confusion matrix for the test set
- Model performance logged in WandB

## Example
To train a model with ReLU activation and Adam optimizer:
```bash
python train.py --dataset mnist --epochs 15 --batch_size 32 --optimizer adam --activation relu
```

## Results Tracking with WandB
Ensure you are logged into WandB before running the script:
```bash
wandb login
```

