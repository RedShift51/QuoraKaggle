#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras.layers import Conv1D, Conv2D, Dense, Input, Flatten, MaxPooling2D
from keras.layers import Concatenate, BatchNormalization
from keras.models import Model, Sequential
import keras

def keras_model(len_question = 30):
    
    X10 = Input(shape = (len_question,300,1))
    X20 = Input(shape = (len_question,300,1))
    
    X1 = Conv2D(filters = 64, kernel_size = (3,300), strides = (1,1), \
                activation = 'relu', padding = 'valid')(X10)
    X1 = Conv2D(filters = 32, kernel_size = (4,1), activation = 'relu', \
                padding = 'same')(X1)
    X1 = MaxPooling2D(pool_size = (4,1))(X1)
    X1 = BatchNormalization()(X1)
    
    X1 = Flatten()(X1)
    X1 = Dense(units = 100, activation = 'relu')(X1)
    
    X2 = Conv2D(filters = 64, kernel_size = (3,300), strides = (1,1), \
                activation = 'relu', padding = 'valid')(X20)
    X2 = Conv2D(filters = 32, kernel_size = (4,1), activation = 'relu', \
                padding = 'same')(X2)
    X2 = MaxPooling2D(pool_size = (4,1))(X2)
    X2 = BatchNormalization()(X2)
    
    X2 = Flatten()(X2)
    X2 = Dense(units = 100, activation = 'relu')(X2)
    
    X = Concatenate(axis = -1)([X1, X2])
    X = Dense(units = 100, activation = 'relu')(X)
    X = Dense(units = 500, activation = 'relu')(X)
    X = Dense(units = 100, activation = 'relu')(X)
    X = Dense(units = 1, activation = 'sigmoid')(X)
    
    return Model(inputs = [X10, X20], outputs = X)

"""
example = keras_model()
opt = keras.optimizers.Adam(lr = 0.001)
example.compile(optimizer = opt, loss = 'binary_crossentropy', batch_size = 1, \
                          metrics = ['accuracy'])
example.summary()
"""

""" Embeddings train """

def word2vec(size_of_dict):
    #X0 = Input(shape = (size_of_dict,))
    model = Sequential()
    model.add(Dense(units = 300, input_shape = (size_of_dict,),\
              activation = 'relu'))
    model.add(Dense(units = size_of_dict, activation = 'softmax'))
    return model


