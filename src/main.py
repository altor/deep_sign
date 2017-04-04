# USAGE
# python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages
from __future__ import division
from lenet import LeNet
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD
from keras.utils import np_utils
from scipy import ndimage
import numpy as np
import argparse
import cv2
import helper
import dataset 

ap = argparse.ArgumentParser()
ap.add_argument("--nbConv1", type=int, default=20,
                help="(optional) number of convolution in the first layer")
ap.add_argument("--szConv1", type=int, default=5,
                help="(optional) size of receptive field in the first layer")
ap.add_argument("--nbConv2", type=int, default=50,
                help="(optional) number of convolution in the second layer")
ap.add_argument("--szConv2", type=int, default=5,
                help="(optional) size of receptive field in the second layer")
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
args = vars(ap.parse_args())


dataset = dataset.get_gtsrb()

# Separation des ensembles d'apprentissage et de validations
data = dataset.data[:, np.newaxis, :, :]
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data / (65536 * 255), dataset.label.astype("int"), test_size=0.33)

trainLabels = np_utils.to_categorical(trainLabels, 43)
testLabels = np_utils.to_categorical(testLabels, 43)

# initialize the optimizer and model
print("[INFO] compiling model")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=43,
                    weightsPath=args["weights"] if args["load_model"] > 0 else None,
                    nbConv1=args["nbConv1"], conv1size=args["szConv1"],
                    nbConv2=args["nbConv2"], conv2size=args["szConv2"]
)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
    print("[INFO] training...")
    if args["converge"] > 0:
        model.fit(trainData, trainLabels, batch_size=128, nb_epoch=args["epochs"],
                  verbose=1)
    else:
        model.fit(trainData, trainLabels, batch_size=128,
                  nb_epoch=args["epochs"], verbose=2,
                  validation_data=(testData, testLabels))

    # show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
        batch_size=128, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)

# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # classify the digit
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)

    image = (helper.rgb_to_img(testData[i][0] * 255 * 65536)).astype(np.uint8)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, str(prediction[0]), (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
    # show the image and prediction
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
        np.argmax(testLabels[i])))
    cv2.imshow("Digit", image)
    cv2.waitKey(0)
