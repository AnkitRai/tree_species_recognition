#!usr/bin/python
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D
from keras import backend as K

import config

class ConvolutionNet:
	def __init__(self):
		pass

	@staticmethod
	def build(name, *args, **kargs):
		# define the network (i.e., string => function) mappings
		mappings = {
			"net": ConvolutionNet.ConvNet}

		# grab the builder function from the mappings dictionary
		builder = mappings.get(name, None)

		# if the builder is None, then return None
		if builder is None:
			return None

		# otherwise, build the network architecture
		return builder(*args, **kargs)

	@staticmethod
	def ConvNet():
		# initialize the model
		model = Sequential()

		# define the layes
		model.add(Convolution2D(32, 3, 3, border_mode="same",
			input_shape=(config.numChannels, config.imgRows, config.imgCols)))
		model.add(Activation("relu"))
		model.add(Convolution2D(32, 3, 3))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# check to see if dropout should be applied to reduce overfitting
		if config.dropout:
			model.add(Dropout(0.25))

		model.add(Convolution2D(64, 3, 3, border_mode="same"))
		model.add(Activation("relu"))
		model.add(Convolution2D(64, 3, 3))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# check to see if dropout should be applied to reduce overfitting
		if config.dropout:
			model.add(Dropout(0.25))

		# define the set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))

		# check to see if dropout should be applied to reduce overfitting
		if config.dropout:
			model.add(Dropout(0.5))

		# define the soft-max classifier
		model.add(Dense(numClasses))
		model.add(Activation("softmax"))

		# return the model base architecture
		return model
