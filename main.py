"""
Execute this file to be prompted to choose wherever to train a new network or
retrain an existing one
"""

import utils
import create_network

choice = input("Type \"n\" to create a new network or \"t\" to continue training an existing one: ")
if choice == "n":
    create_network.createNetwork()

elif choice == "t":
    while True:
        try:
            epochs = int(input("How many epochs? Recommended 5-30 "))
            if epochs > 0:
                break
            else:
                print("Please give a positive whole number.")
        except ValueError:
            print("Please give a positive whole number.")

    
    model = utils.loadModel() # load the model with utils, apply weights
    create_network.runNetwork(model, 64, epochs)

else:
    print("Please read the instruction message.")