#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, pandas as pd, numpy as np, re
import keras
from keras.layers import Conv1D, Dense, Input
import matplotlib.pyplot as plt
from model import word2vec, keras_model
from keras.models import Sequential

data = pd.read_csv('/home/alex/kaggle/quora/train.csv')

print('Number of 1s is ', sum(data['is_duplicate']))
print('Number of 0s is ', len(data) - sum(data['is_duplicate']))

""" Firstly let's balance class is_duplicate for training """

aux_numbers_0 = np.array([i for i in range(len(data)) \
                          if data['is_duplicate'][i] == 0])
aux_numbers_1 = np.array([i for i in range(len(data)) \
                          if data['is_duplicate'][i] == 1])

np.random.shuffle(aux_numbers_0)
np.random.shuffle(aux_numbers_1)

train_numbers = np.concatenate([aux_numbers_1,aux_numbers_0[:len(aux_numbers_1)]])
np.random.shuffle(train_numbers)

""" Preprocessing """
#x1 = np.array([len(data['question1'][i].split(' ')) for i in range(len(data))])
#x2 = np.array([len(data['question2'][i].split(' ')) for i in range(len(data)) \
#               if type(data['question2'][i]) is str])

#plt.hist(x1, bins = 50)
#plt.hist(x2, bins = 50)

""" From graphs we know that optimal len of input is 30 words """
""" Let's transform our questions to vectors """
""" But firstly we will get words frequences """

""" Making words storage """
storage_question1, storage_question2 = {}, {}
for i in range(len(data)):
    if type(data['question1'][i]) is str:
        s = data['question1'][i].lower()
        s = re.sub(r'[^\w]', ' ', s).split()
        storage_question1[str(i)] = s
    else:
        storage_question1[str(i)] = []
    if type(data['question2'][i]) is str:
        s = data['question2'][i].lower()
        s = re.sub(r'[^\w]', ' ', s).split()
        storage_question2[str(i)] = s
    else:
        storage_question2[str(i)] = []
    print(i)

""" Let's count number of appearances for each word """
words = {}
for i in range(len(data)):
    s = storage_question1[str(i)]
    for j in s:
        if words.get(j, -1) == -1:
            words[j] = 1
        else:
            words[j] += 1
    s = storage_question2[str(i)]
    for j in s:
        if words.get(j, -1) == -1:
            words[j] = 1
        else:
            words[j] += 1
    print(i, len(data))
""" Here we sort words per their numbers of appearances """
words = pd.DataFrame({'words':list(words.keys()), 'counts':list(words.values())})
words = words.sort_values('counts', ascending=False).reset_index(drop=True)

""" Take first 10000 the most frequent words """
actual_words = words['words'][:10000]
actual_words = {actual_words[i]:i for i in range(len(actual_words))}
""" Training embeddings """
"""
The embedding small net is based on  
https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b
"""
proc2vec = word2vec(10000)
opt = keras.optimizers.Adam()
proc2vec.compile(optimizer = opt, loss = 'categorical_crossentropy')
proc2vec.summary()

def onehot(pos, len_actual=10000):
    ans = np.zeros((len_actual,1))
    ans[pos,0] = 1
    return ans

def train_append(s):
    train = []
    positions = {str(i):1 for i in range(len(s))}
    for j0, j in enumerate(s):
        if positions.get(str(j0-1),-1) != -1:
            train.append((s[j0], s[j0-1]))
        if positions.get(str(j0-2),-1) != -1:
            train.append((s[j0], s[j0-2]))
        if positions.get(str(j0+2),-1) != -1:
            train.append((s[j0], s[j0+2]))
        if positions.get(str(j0+1),-1) != -1:
            train.append((s[j0], s[j0+1]))
    return train
    
for i in range(len(data))[:25000]:
    s = storage_question1[str(i)]
    train = train_append(s)
    if len(train) > 0:
        train = [(actual_words[k1],actual_words[k2]) for k1,k2 in train \
             if actual_words.get(k1,-1) != -1 \
                                     and actual_words.get(k2,-1) != -1]
        if len(train) > 0:
            X = np.concatenate([onehot(t0) for t0,t in train],axis=1).T
            Y = np.concatenate([onehot(t) for t0,t in train],axis=1).T
            proc2vec.fit(X,Y,epochs=3)
    s = storage_question2[str(i)]
    train = train_append(s)
    if len(train) > 0:
        train = [(actual_words[k1],actual_words[k2]) for k1,k2 in train \
             if actual_words.get(k1,-1) != -1 \
                                     and actual_words.get(k2,-1) != -1]
        if len(train) > 0:
            X = np.concatenate([onehot(t0) for t0,t in train],axis=1).T
            Y = np.concatenate([onehot(t) for t0,t in train],axis=1).T
            proc2vec.fit(X,Y,epochs=3)
    print(i, len(data), 'embeddings training')

#proc2vec.save('/home/alex/kaggle/quora/proc2vec1.h5')
#proc2vec = keras.models.load_model('/home/alex/kaggle/quora/proc2vec1.h5')
""" Let's train model """

low_dim_model = Sequential()
low_dim_model.add(Dense(units = 300, input_shape = (10000,),\
              weights = proc2vec.layers[0].get_weights(), activation = 'relu'))
    
def question2mat(s, len_actual = 1500):
    start_embed = [low_dim_model.predict(onehot(actual_words[i], \
                    len_actual = len_actual).T) for i in s if \
                    actual_words.get(i,-1) != -1]
    if len(start_embed) == 0:
        return np.zeros((30,300))
    else:    
        start_embed = np.concatenate(start_embed, axis = 0)
        if len(start_embed[:,0]) >= 30:
            return start_embed[:30,:]
        elif len(start_embed[:,0]) == 0:
            return np.zeros((30,len(start_embed[0,:])))
        else:
            return np.concatenate([start_embed, np.zeros(\
            (30-len(start_embed[:,0]),len(start_embed[0,:])))], axis = 0)    
    
#def equal(dict1, dict2):
#    actual_components = []
#    for i in list(dict1):
#        if 

for i in range(2900):
    batch = train_numbers[i*100:(i+1)*100]
    X1 = np.concatenate([np.expand_dims(question2mat(storage_question1[str(j)], \
                        len_actual=10000),axis=0)
                  for j in batch], axis=0)
    X2 = np.concatenate([np.expand_dims(question2mat(storage_question2[str(i)], \
                        len_actual=10000),0)
                         for i in batch], axis=0)
    np.save(open('/home/ubuntu/alexey-interview/batches/'+str(i)+'1', \
                 'w'), X1)
    np.save(open('/home/ubuntu/alexey-interview/batches/'+str(i)+'2', \
                 'w'), X2)
    print('We have', i, 'of', 2900, 'embeddings')    
    
"""
We have two parallel convolution blocks at the entry, then they
are flattened and concated and then we have block of dense layers
"""
end_model = keras_model()
end_model.compile(optimizer = opt, loss = 'binary_crossentropy', \
                          metrics = ['accuracy'])
end_model.summary()

history = []
for i in range(2900):
    batch = train_numbers[i*100:(i+1)*100]
    #np.random.randint(low=0,high=len(train_numbers),size=100)]
    X1 = np.load(open('/home/ubuntu/alexey-interview/batches/'+str(i)+'1', \
                 'w'))
    X2 = np.load(open('/home/ubuntu/alexey-interview/batches/'+str(i)+'2', \
                 'w'))
    """
    [np.expand_dims(question2mat(storage_question1[str(j)], \
                        len_actual=10000),0)
                  for j in batch]
    
    X2 = [np.expand_dims(question2mat(storage_question2[str(i)], \
                        len_actual=10000),0)
                         for i in batch]
    """
    
    #X1 = np.concatenate(X1, axis = 0)
    #X2 = np.concatenate(X2, axis = 0)
    Y = np.array(data['is_duplicate'][batch])  
    history.append(end_model.fit([np.expand_dims(X1,3), np.expand_dims(X2,3)], Y, \
                                  epochs = 5, batch_size = 50, validation_split = 0.2))
    #if (i+1) % 150 == 0:
    #    end_model.save('/home/alex/kaggle/quora/end_model1.h5')        
    print('Final model is training', i, 3000)

val_acc = []
val_loss = []
for i in range(history):
    val_acc += history[i].history['val_acc']
    val_loss += history[i].history['val_loss']


import json
json.dump({'acc':val_acc, 'loss':val_loss}, open('results_model.json','w'))
