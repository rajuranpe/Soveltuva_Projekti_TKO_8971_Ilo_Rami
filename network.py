from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D

class CNNetwork():
"""
A class for the convolutional neural network.
"""
    def __init__(self, batch_size, categories, img_dimensions, x_train_shape):
        """
        Initialize
        
        Args:
            batch_size: amount of training examples per iteration
            categories: possible categories of images, needed for the output layer
            img_dimensions: amount of pixels in the image + colors
            x_train_shape: shape of the training data to determine size of the model's input
            model: convolutional neural network. The method createModel is called to build it.
        """
        self.batch_size = batch_size
        self.categories = categories
        self.img_dimensions = img_dimensions        # (32, 32, 3), where 3 is the length of the color vector (RGB)
        self.x_train_shape = x_train_shape
        self.model = self.createModel()

    
    def createModel(self):
    """
    Convolutional neural network suited for the task. Uses multitude of methods to build a sufficient model, such as Dropout and BatchNormalization. 
    Used activation function for hidden layers, relu, has achieved best results in similiar tasks.
    
    Returns:
        The model structure readable by Keras
    """
        dim = self.img_dimensions
        model = Sequential()
        model.add(Conv2D(dim, (3, 3), padding="same", input_shape=self.x_train_shape))
        model.add(Activation("relu"))                   # linear activation function
        model.add(BatchNormalization())                 # keep values withing reasonable range and reduce overfitting
        model.add(Conv2D(dim, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))     # dismiss a portion of images from training to reduce overfitting

        model.add(Conv2D(2 * dim, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(2 * dim, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(4 * dim, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(4 * dim, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(2 * dim, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(2 * dim, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(dim, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(dim, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(self.categories, activation="softmax"))  # softmax to get the probabilities of each category

        return model

