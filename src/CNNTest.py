# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class CNNTest:
        """
        The width of our input images.
        The height of our input images.
        The depth (i.e., number of channels) of our input images.
        And the number of classes (i.e., unique number of class labels) in our dataset.
        """
        @staticmethod
        def build(width, height, depth, classes, weightsPath=None, 
                  nbConv1=100, conv1size=7, nbConv2=150, conv2size=4,
				  nbConv3=250, conv3size=4, activationFun="relu"):
		# initialize the model
                model = Sequential()

		# first set of CONV => activation => POOL
                # la premiere couche est compose de 20 filtres de convolutions de taille 5x5
                model.add(
                        Convolution2D(
                                nbConv1,
                                (conv1size, conv1size),
                                padding="same",
			        input_shape=(depth, height, width))
                )
                # fonction d'activation 
                model.add(Activation(activationFun))
                # couche de max-pooling, taille des fenetres : 2x2, pas de deplacement de la fenetre (stride) : 2x2
                model.add(
                        MaxPooling2D(
                                pool_size=(2, 2),
                                strides=(2, 2), dim_ordering="th")
                )

		# second set of CONV => Activation => POOL
                model.add(
                        Convolution2D(nbConv2, conv2size, conv2size,
                                        border_mode="same")
                )
                model.add(Activation(activationFun))
                model.add(
                        MaxPooling2D(pool_size=(2, 2),
                                     strides=(2, 2), dim_ordering="th")
                )
				
		# third set of CONV => Activation => POOL
                model.add(
                        Convolution2D(nbConv3, conv3size, conv3size,
                                        border_mode="same")
                )
                model.add(Activation(activationFun))
                model.add(
                        MaxPooling2D(pool_size=(2, 2),
                                     strides=(2, 2), dim_ordering="th")
                )

		# set of FC => RELU layers
                # ajout d'une "couche" permetant de transformer la sortie de la couche de max-pooling en un unique vecteur
                model.add(Flatten())
                # couche de 300 neurones completement connecte avec fonction d'activation relu
                model.add(Dense(300))
                model.add(Activation(activationFun))

		# softmax classifier
                # couche de neurone completement connecte avec autant de neurones que de classes
                model.add(Dense(classes))
                model.add(Activation("softmax"))

		# if a weights path is supplied (inicating that the model was
		# pre-trained), then load the weights
                if weightsPath is not None:
                        model.load_weights(weightsPath)

		# return the constructed network architecture
                return model
