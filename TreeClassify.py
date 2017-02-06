import numpy as np
import pandas as pd
import cv2
import keras
from matplotlib import pyplot as plt
import h5py

data = pd.read_csv("data.csv")

print "The number of images and label:"
print data.shape


# Setting the seed for reproducibility 
seed= 10
np.random.seed(seed)

X,Y = data_subset.iloc[:,0],data_subset.iloc[:,1]

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

# Split the dataset - training
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)

trainData,testData = np.asarray(X_train), np.asarray(X_test)
trainLabel,testLabel = np.asarray(Y_train) , np.asarray(Y_test)

# normalize the data - 
trainData,testData = trainData/255 , testData/255

# Building the Model
unique_val = set(y)

print len(unique_val)

### importing the required packages
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.constraints import maxnorm
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

num_classes = 184

trainData = np.ndarray(shape=(799,3,800,600),dtype=np.float32)

testData = np.ndarray(shape=(200,3,800,600),dtype=np.float32)

from keras.optimizers import SGD
sgd = SGD(lr=0.1,decay=1e-3,momentum=0.9,nesterov=True)

def base_model():
    model = Sequential()
    model.add(Convolution2D(32,3,3, border_mode='same',input_shape=(3,800,600),
                           activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation ='relu',W_constraint=maxnorm(3)))
    model.add(Dense(num_classes, activation='softmax'))
    
#Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    return model

filepath ="weights.best.hdf5"

#Create checkpoint for best model only
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')

callbacks_list = [checkpoint]

#build the CNN model
model = base_model()


## Load the best ckeckpoints - 

model.load_weights("weights-improvement-03--0.19.hdf5")


#Fit the model
history = model.fit(trainData,trainLabel, validation_data=(testData,testLabel),callbacks=callbacks_list,
         nb_epoch=40, batch_size=100, verbose=2)

print model.summary()


#Final scores
scores = model.evaluate(testData,testLabel,verbose=0)


# In[73]:

#Save the model
model.save_weights("CNN_PlantClassify.h5")

## Plotting and evaluating model history -
print history.history.keys()


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



