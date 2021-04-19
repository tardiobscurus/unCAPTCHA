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

import matplotlib.pyplot as plt

"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
print(y_train.dtype)
print(y_train.shape)
print(y_train[0])
y_train = keras.utils.to_categorical(y_train, num_classes)
print(y_train.shape)
print(y_train.dtype)
print(y_train[0])
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

#model = keras.Sequential(
#    [
#        keras.Input(shape=input_shape),
#        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#        layers.MaxPooling2D(pool_size=(2, 2)),
#        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#        layers.MaxPooling2D(pool_size=(2, 2)),
#        layers.Flatten(),
#        layers.Dropout(0.5),
#        layers.Dense(num_classes, activation="softmax"),
#    ]
#)

model = keras.models.load_model('mnist_convnet.model')

model.summary()

"""
## Train the model
"""

batch_size = 128
#epochs = 15
epochs = 5

#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_class = model.predict_classes(x_test[:10])
y_pred  = model.predict(x_test[:10])

for yc, yp, yt in zip(y_class, y_pred, y_test):
    print(yc, 10*' %.3f' % tuple(yp), yt)
    
for i in range(10):
    stamp = x_test[i,:,:]
    plt.imshow(stamp)
    plt.show()
