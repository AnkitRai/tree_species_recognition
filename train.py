#!usr/bin/python
# import the necessary packages
import cv2
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split

from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import np_utils

from convnet import config
from convnet.tree_net import ConvolutionNet

seed =42
np.random.seed(seed=seed)

# load the training and testing data, then scale it into the range [0, 1]
print("[INFO] loading training and validation data...")

data = pd.read_csv('leafsnap_data.csv')

print("The number of images: {} and label: {}".format(data.shape[0], data.shape[1]))

X,Y = data.iloc[:,0],data.iloc[:,1]

X = X.values
Y = Y.values
# create lists
x = []
y = []

#reading the images off the disk -
for i in X:
    img = cv2.imread(i)
    x.append(img)

y = list(Y)

print('[INFO]: Splitting train and test set..')

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)

trainData,testData = np.asarray(X_train), np.asarray(X_test)
trainLabel,testLabel = np.asarray(Y_train) , np.asarray(Y_test)

print('[INFO]: Normalize images between 0 to 1..')
trainData = trainData/255
testData = testData/255

num_classes = 184
trainData = np.ndarray(shape=(799,3,800,600),dtype=np.float32)
testData = np.ndarray(shape=(200,3,800,600),dtype=np.float32)

# collect the keyword arguments to the network
kargs = {"dropout": 0.25, "activation": "softmax"}

# train the model using SGD
print("[INFO] compiling model...")
model = ConvolutionNet.build("net", 3, 32, 32, 10, **kargs)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

filepath ="weights.best.hdf5"

#Create checkpoint for best model only
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')

callbacks_list = [checkpoint]

# start the training process
print("[INFO] starting training...")
history = model.fit(trainData,trainLabel, validation_data=(testData,testLabel),callbacks=callbacks_list,
         nb_epoch=40, batch_size=32, verbose=2)


# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(testData, testLabels, verbose=0)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
#model.save_weights(config.MODEL_WEIGHTS)
model.save_weights("CNN_PlantClassify.h5")

## Plotting and evaluating model history -
print(history.history.keys())

# Checking the model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.legend(['training','validation'], loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('model-accuracy.png')


# Checking the model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.legend(['training','validation'], loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('model loss')
plt.savefig('model_loss.png') 
