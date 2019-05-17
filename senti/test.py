import keras
import numpy as np 
import sklearn as sk 
import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import *
from keras import layers
from keras.callbacks import TensorBoard
from time import time
from keras.models import Model
import os
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from sklearn import preprocessing


df= pd.read_csv('har.csv')
#df= pd.read_csv('./labeled data/combined/5_stay_walk_jog_skip_stUp_stDown.csv')
#df= pd.read_csv('1_stay.csv')

columns= ['acc_x', 'acc_y', 'acc_z']

x=df[columns].values

prediction=['label']

y= df[prediction].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2,random_state=42)
#x_train=preprocessing.normalize(x_train)
#x_test=preprocessing.normalize(x_test)
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
#print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 3, 1)
x_test = x_test.reshape(x_test.shape[0], 3,1)

print(x_train.shape)
print(y_train.shape)
print(y_train_hot.shape)

model= Sequential()
'''
model.add(Conv1D(512,2, activation='relu', input_shape=(3,1)))
#model.add(MaxPooling1D(1, padding='same'))
model.add(Conv1D(256,2, activation='relu'))
model.add(Conv1D(100,1, activation='relu'))


model.add(Flatten())
model.add(Dense(500, activation='relu'))

model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
'''

model.add(LSTM(100, return_sequences=True, input_shape=(3,1)))
#model.add(LSTM(100, return_sequences=True))
#model.add(LSTM(100, return_sequences=False))
#model.add(Dropout(0.30))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(x_train, y_train_hot, batch_size=100, epochs=1000, verbose=1,validation_data=(x_test,y_test_hot), callbacks=[keras.callbacks.TensorBoard(log_dir="newlogs/raw1/{}".format(time()), histogram_freq=0, write_graph=False, write_images=False)]
)


score,acc=model.evaluate(x_test, y_test_hot, batch_size=100)

print(acc)







