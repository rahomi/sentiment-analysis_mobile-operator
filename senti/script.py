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
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_excel("Book2.xlsx")
print(df.shape)
df = df.dropna()
print("After Drop: ",df.shape)

columns = ["message"]
x = df[columns]
cv = CountVectorizer(binary=True)
cv.fit_transform(x)
'''
x_train = cv.transform(x_train.ravel())
x_test = cv.transform(x_test.ravel())

x = df[columns].values
'''
x = np.asarray(x) 
print(x.shape)

print(x)

label = ["label"]

y = df[label].values
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# x_train = np.array(x_train)

# x_train = np.asarray(x_train,dtype=np.float32)
# x_test = np.asarray(x_test,dtype=np.float32)

print("PPPPP",x_train.shape)
# x_train=preprocessing.normalize(x_train)
# x_test=preprocessing.normalize(x_test)

print(y_train[2])
# print(y_test)

y_train_hot = to_categorical(y_train)
print(y_train_hot[0])
y_test_hot = to_categorical(y_test)


x_train = x_train.reshape(x_train.shape[0], 1, 1)

print(x_train.shape)

x_test = x_test.reshape(x_test.shape[0], 1, 1)

model = Sequential()
model.add(LSTM(20, return_sequences=True, input_shape=(1,1))) 
model.add(Flatten())
model.add(Dense(10, activation='relu'))
# model.add(Dense(25, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train_hot, batch_size=20, epochs=100, verbose=1,validation_data=(x_test,y_test_hot))
