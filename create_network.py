from keras.utils import np_utils
import tensorflow as tf
import numpy as np
import utils
from keras.preprocessing.image import ImageDataGenerator
from network import CNNetwork



def prepareData():
"""
Prepare data into tensors and normalize it by using mean and standard deviation, so the data can be used unbiased.

Returns:
    Training and validation (data, labels)
"""
        # Load the data and allocate it into correct matrices.
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        categories = 100

        # Ensuring that the values are in 32 decimal float
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        # Z-score normalization
        mean = np.mean(x_train)
        std = np.std(x_train)
        x_train = (x_train - mean) / (std + 1e-7)
        x_test = (x_test - mean) / (std + 1e-7)

        # Allocate validation data into categories to be used with crossentropy
        y_train = np_utils.to_categorical(y_train, categories)
        y_test = np_utils.to_categorical(y_test, categories)

        return (x_train, y_train, x_test, y_test)

def createNetwork():
"""
Creates a new network, calls runNetwork to pass the new model for compilation.
"""
        batch_size = 64  # no. of training examples per iteration
        epochs = 20  # no. of iterations the neurons update their weights
        categories = 100  # CIFAR100 has 100 categories
        img_dimensions = 32  # image dimensions (n*n)

        # Create a new instance of CNNetwork class, which creates the network model and image data generator
        cifar100 = CNNetwork(batch_size, categories, img_dimensions, (32, 32, 3))
        model = cifar100.model
        model.summary()  # Show a table containing details of the created neural network
        runNetwork(model, batch_size, epochs)

def runNetwork(model, batch_size, epochs):
"""
Compiles and runs the network with ImageDataGenerator.

Args:
    model: A neural network model structure
    batch_size: no. of training examples per iteration
    epochs: no. of iterations the neurons update their weights
    
Returns:
    printed results of the model's running accuracy
"""
        x_train, y_train, x_test, y_test = prepareData()

        # Compile the network
        model.compile(loss="categorical_crossentropy",  # use the loss function to take into note the probabilities of each category
                optimizer="rmsprop",                    
                metrics=["accuracy"])                   # use accuracy as a metric of progress during training

        # Training
        model.fit_generator(createGenerator().flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs, verbose=1,
                            validation_data=(x_test, y_test))
        
        # Evaluate performance. If training accuracy >> evaluation accuracy, overfitting is present.
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print("Accuracy: ", str(test_acc))
        utils.saveModel(model)


def createGenerator():   
"""
Generator to process images and apply different methods on them to normalize data

Returns:
    ImageDataGenerator of fixed type
"""
        generator = ImageDataGenerator(
            rotation_range=15,          # Apply random rotation to images, from 0 to 180 degrees
            width_shift_range=0.1,      # Randomly shift images horizontally, portion of the total width
            height_shift_range=0.1,     # -||- vertically
            horizontal_flip=True,       # Randomly flip images horizontally
        )

        return generator