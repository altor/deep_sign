# USAGE
# python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages
from __future__ import division
from lenet import LeNet
from arch2 import Arch2
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LambdaCallback
from scipy import ndimage
import numpy as np
import argparse
import cv2
import helper
import dataset 

ap = argparse.ArgumentParser()
ap.add_argument("--nbConv1", type=int, default=20,
                help="(optional) number of convolution in the first layer")
ap.add_argument("--verbose", type=int, default=0,
                help="(optional) active verbose mode")
ap.add_argument("--szConv1", type=int, default=5,
                help="(optional) size of receptive field in the first layer")
ap.add_argument("--nbConv2", type=int, default=50,
                help="(optional) number of convolution in the second layer")
ap.add_argument("--szConv2", type=int, default=5,
                help="(optional) size of receptive field in the second layer")
ap.add_argument("--activationFun", type=str, default="relu",
                help="(optional) number of convolution in the first layer")
ap.add_argument("--nbFull", type=int, default=500,
                help="(optional) number of neurons in the fully-connected layer")
ap.add_argument("-e", "--epochs", type=int, default=20,
                help="(optional) number of training epochs")
ap.add_argument("-s", "--save-model", type=int, default=-1,
                help="(optional) whether or not model should be saved to disk")
ap.add_argument("-c", "--converge", type=int, default=-1,
                help="(optional) active convergence")
ap.add_argument("-l", "--load-model", type=int, default=-1,
            help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
            help="(optional) path to weights file")

ap.add_argument("--arch", type=str, default="lenet",
                help="(optional) architecture used for the network : lenet(default) or arch2")
args = vars(ap.parse_args())


dataset = dataset.get_gtsrb(verbose=args["verbose"])

# Separation des ensembles d'apprentissage et de validations
data = dataset.data[:, np.newaxis, :, :]
trainData, testData, trainLabels, testLabels = None, None, None, None

# Si l'on cherche juste a entrainer le réseau on ne sépare pas les données d'apprentissage des données de test
if args["save-model"] == 1:
    trainData = data / (65536 * 255)
    trainLabel = np_utils.to_categorical(dataset.label.astype("int"), 43)
else:
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        data / (65536 * 255), dataset.label.astype("int"), test_size=0.33)
    trainLabels = np_utils.to_categorical(trainLabels, 43)
    testLabels = np_utils.to_categorical(testLabels, 43)

# initialize the optimizer and model
if args["verbose"] == 1:
    print("[INFO] compiling model")
opt = SGD(lr=0.01)

model = None
if args["arch"] == "lenet":
    model = LeNet.build(
        width=28, height=28, depth=1, classes=43,
        weightsPath=args["weights"] if args["load_model"] > 0 else None,
        nbConv1=args["nbConv1"], conv1size=args["szConv1"],
        nbConv2=args["nbConv2"], conv2size=args["szConv2"],
        activationFun=args["activationFun"],
        nbN=args["nbFull"]
    )

elif args["arch"] == "arch2":
    model = Arch2.build(
        width=28, height=28, depth=1, classes=43,
        weightsPath=args["weights"] if args["load_model"] > 0 else None,
        nbConv1=args["nbConv1"], conv1size=args["szConv1"],
        nbConv2=args["nbConv2"], conv2size=args["szConv2"],
        activationFun=args["activationFun"]
    )
    
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
    if args["verbose"] == 1:
        print("[INFO] training...")
    if args["converge"] > 0:
        # Fonction utilise pour lancer la validation du model apres chaque periode d'apprentissage
        def f(epochs, logs):
            (_, accuracy) = model.evaluate(testData, testLabels,batch_size=128, verbose=args["verbose"])
            print(logs)
            print("LOG_ACC;"+format(accuracy))
            
        model.fit(trainData, trainLabels, batch_size=128, nb_epoch=args["epochs"],verbose=2,
                  callbacks=[LambdaCallback(on_epoch_end=f)])
        
    elif args["save-model"] == 1:
        model.fit(trainData, trainLabels, batch_size=128,
                  nb_epoch=args["epochs"], verbose=2,
                  validation_data=(testData, testLabels))
    else:
        model.fit(trainData, trainLabels, batch_size=128,
                  nb_epoch=args["epochs"], verbose=2,
                  validation_data=(testData, testLabels))

    # show the accuracy on the testing set
    # print("[INFO] evaluating...")
    if args["verbose"] == 1:
        (loss, accuracy) = model.evaluate(testData, testLabels,
                                          batch_size=128, verbose=1)
        print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)
 
