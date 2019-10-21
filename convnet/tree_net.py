#!usr/bin/python
import sys
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from convnet import config

# define the model
def shallow_net():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu'
		, kernel_initializer='he_uniform'
		, padding='same'
		, input_shape=(config.imgRows, config.imgCols, config.numChannels)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	
	# compile model
	opt = SGD(lr=config.learning_rate, momentum=config.momentum)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	
	return model

