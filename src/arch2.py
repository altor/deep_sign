import keras
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input


class Arch2:
        """
        The width of our input images.
        The height of our input images.
        The depth (i.e., number of channels) of our input images.
        And the number of classes (i.e., unique number of class labels) in our dataset.
        """
        @staticmethod
        def build(width, height, depth, classes, weightsPath=None,
                  nbConv1=20, conv1size=5, nbConv2=50, conv2size=5,
                  activationFun="relu", nbN=500):

            print("[INFO] using Arch2")
            
            input = Input(shape=(depth, height, width), name='input')
            
	    # Premier set of CONV => RELU => POOL
            conv1 = Convolution2D(
                nbConv1,
                (conv1size, conv1size),
                padding="same",
                name="conv1")(input)
            
            act1 = Activation(activationFun, name="act1")(conv1)
            pool1 = MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2), dim_ordering="th",
                name="pool1")(act1)

            pool1_2 = MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2), dim_ordering="th",
                name="pool1_2")(pool1)
            
		# second set of CONV => RELU => POOL
            conv2 = Convolution2D(
                nbConv2, conv2size, conv2size,
                border_mode="same", name="conv2")(pool1)
            act2 = Activation(activationFun, name="act2")(conv2)
            pool2 = MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2), dim_ordering="th",
                name="pool2")(act2)

	    # set of FC => RELU layers
            # ajout d'une "couche" permetant de transformer la sortie de la couche de max-pooling en un unique vecteur

            concat = keras.layers.concatenate([pool1_2, pool2])
            
            x=Flatten()(concat)
            # couche de 500 neurones completement connecte avec fonction d'activation relu
            x = Dense(nbN)(x)
            x = Activation(activationFun)(x)

	    # softmax classifier
            # couche de neurone completement connecte avec autant de neurones que de classes
            x = Dense(classes)(x)
            output = Activation("softmax")(x)

	    # if a weights path is supplied (inicating that the model was
	    # pre-trained), then load the weights
            if weightsPath is not None:
                model.load_weights(weightsPath)

	    # return the constructed network architecture
            return Model(inputs=[input], outputs=[output])

