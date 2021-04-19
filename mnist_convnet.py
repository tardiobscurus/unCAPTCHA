"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2020/04/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
"""

"""
## Setup
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Print reassuring progress messages
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model

Our deep convolutional neural network model consists of the following layers:

- Input layer to accept training and test data (postage stamp images of single digits)
- 2D convolutional layer with 32 kernels, 3x3 pixels each to be learned with RELU activation function
- 2D max pooling layer to basically block average the output from previous layer in areas 2x2 pixels
- Another 2D convolutional layer with 64 kernels, 3x3 pixels each to be learned with RELU activation function
- Another 2D max pooling layer to basically block average the output from previous layer in areas 2x2 pixels
- Flattening step to go from 2D to 1D output
- A dropout layer that simply forgets randomly selected half of all inputs
- A dense layer to go from remaining inputs to a softmax probability estimate for each possible output

When presented with a postage stamp image of a single character,
the model will output a vector of probabilities for each possible
character 0-9. The character with the highest probability is the
predicted character.

"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
#epochs = 15
epochs = 5

# We train our deep convolutional network model above by minimizing a loss function
# called categorical crossentropy (meaning crossentropy for predicting class membership).
# Adam is an optimization algorithm similar to stochastic gradient descent,
# but it is much more sophisticated and has adaptive features to improve efficiency,
# convergence speed etc. We track progress using a classification accuracy metric
# (the fraction of correct classification predictions), which is more intuitive to
# humans than crossentropy.

# This makes our selections
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# This does the actual training
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# This saves a trained model to a set of files in a folder for later use without retraining
model.save('mnist_convnet.model')

"""
## Evaluate the trained model
"""

# This uses the trained model to predict all test cases
# and compares with expected results to test accuracy.
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
