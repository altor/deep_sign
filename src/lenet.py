# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class LeNet:
        # """
        # The width of our input images.
        # The height of our input images.
        # The depth (i.e., number of channels) of our input images.
        # And the number of classes (i.e., unique number of class labels) in our dataset.
        # """
        @staticmethod
        def build(width, height, depth, classes, weightsPath=None):
		# initialize the model
                model = Sequential()

		# first set of CONV => RELU => POOL
                # la première couche est composé de 20 filtres de convolutions de taille 5x5
                model.add(Convolution2D(20, 5, 5, border_mode="same",
			                input_shape=(depth, height, width)))
                # fonction d'activation = relu
                model.add(Activation("relu"))
                # couche de max-pooling, taille des fenêtres : 2x2, pas de déplacement de la fenêtre (stride) : 2x2
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => RELU => POOL
                model.add(Convolution2D(50, 5, 5, border_mode="same"))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# set of FC => RELU layers
                # ajout d'une "couche" permetant de transformer la sortie de la couche de max-pooling en un unique vecteur
                model.add(Flatten())
                # couche de 500 neurones complétement connecté avec fonction d'activation relu
                model.add(Dense(500))
                model.add(Activation("relu"))

		# softmax classifier
                # couche de neurone complétement connecté avec autant de neurones que de classes
                model.add(Dense(classes))
                model.add(Activation("softmax"))

		# if a weights path is supplied (inicating that the model was
		# pre-trained), then load the weights
                if weightsPath is not None:
                        model.load_weights(weightsPath)

		# return the constructed network architecture
                return model
