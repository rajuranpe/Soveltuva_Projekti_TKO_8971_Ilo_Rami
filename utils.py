from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import os
import create_network
import sys

def saveModel(model):
"""
save the model and weights

Args:
    model: a model to be saved and parsed weights from
"""
    json = model.to_json()  # convert the model into .json
    with open("cifar100_model.json", "w") as json_file:
        json_file.write(json)
    model.save_weights("cifar100_model.h5")  # save the weights to HDF5 format

def loadModel():
"""
convert the model from .json file and apply the weight from HDF5 into it

Return:
    a model with applied weights in it
    
Raises:
    FileNotFoundError
"""
    try:
        json = open('cifar100_model.json', 'r')
        model_json = json.read()
        json.close()
        model = model_from_json(model_json)
        model.load_weights("cifar100_model.h5")
        return model
    except FileNotFoundError:
        print("No saved models yet! Please create a new network.")
        sys.exit()

def categoryName(n):
"""
Contains a list of all the 100 categories 

Returns:
    value at index n of the
"""
    categories = [
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
        "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
        "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
        "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
        "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
        "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
        "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
        "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
        "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
        "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
        "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
        "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
        "worm"
    ]
    return categories[n]

def preview(model, img_dimensions, index):
    """
    Visualize a small, 10 image long preview of evaluated predictions and their true values in Jupyter notebook or similiar app
    
    Args:
        model: a trained model
        img_dimensions: size of the image
        index: desired starting positions for predictions from the validation data
    
    Require:
        index <= len(x_test) - 11
    """
    x_train, y_train, x_test, y_test = create_network.prepareData()
    fig = plt.figure(figsize=(50, 50)) # create a new matplot figure
    for i in range(0, 10): # figure length is 10 images
        image_index = index + i
        sub = fig.add_subplot(10, 1, i + 1) # one subpart of the figure to hold 
        # a single image
        
        pred = model.predict(x_test[image_index].reshape(1, img_dimensions, img_dimensions, 3)) # receive a vector of the probability distribution of each class 
        title = "Subject of the picture is probably " + categoryName(pred.argmax()) # Write a caption for the subpart. The largest probability of pred is the predicted category
        sub.set_title(title) # attach the text to the subpart
        sub.imshow(x_test[image_index].reshape(img_dimensions, img_dimensions, 3), interpolation="nearest") # show the subpart