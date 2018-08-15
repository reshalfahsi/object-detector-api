#importing dependencies
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, RNN, GRU
from keras.utils import np_utils
import os

#load the data
ws_dir = './'
weight_dir = 'saved_weight.h5'
text = open(ws_dir+'chairilanwar_poem.txt').read()
text = text.lower()

#create charachter/word mappings
charachters = sorted(list(set(text)))
n_to_char = {n:char for n, char in enumerate(charachters)}
char_to_n = {char:n for n, char in enumerate(charachters)}

#data preprocessing
X = []
Y = []
X_modified = 0.0
Y_modified = 0.0
model = None
epoch = 10
batch = 200
length = len(text)
seq_length = 120

def initialization() :
    global length, seq_length, text,char_to_n, n_to_char, X_modified, Y_modified, X, Y, charachters
    for i in range(0,length-seq_length,1):
        sequence  = text[i : i+seq_length]
        label = text[i+seq_length]
        X.append([char_to_n[char] for char in sequence])
        Y.append(char_to_n[label])

    X_modified = np.reshape(X,(len(X),seq_length,1))
    X_modified = X_modified / float(len(charachters))
    Y_modified = np_utils.to_categorical(Y)

#model
def runPoem() :
    global model, X_modified, Y_modified, epoch, batch, ws_dir, weight_dir
    model = Sequential()
    model.add(LSTM(200, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(100,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(200))
    model.add(Dropout(0.25))
    model.add(Dense(Y_modified.shape[1],activation='softmax'))
    if os.path.exists(ws_dir+weight_dir) :
        model.load_weights(ws_dir+weight_dir)
    model.compile(loss='categorical_crossentropy',optimizer='adam')
    model.fit(X_modified, Y_modified, epochs=epoch, batch_size=batch)
    model.save_weights(ws_dir+weight_dir)

def savePoem() :
    global model,X,Y, charachters, char_to_n, n_to_char, seq_length
#generating text
    string_mapped = X[seq_length-1]
    full_string = [n_to_char[value] for value in string_mapped]
#generating characters
    for i in range(seq_length) :
        x = np.reshape(string_mapped,(1,len(string_mapped),1))
        x = x / float(len(charachters))
        pred_index = np.argmax(model.predict(x,verbose=0))
        full_string.append(n_to_char[pred_index])
        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]

#combining text
    txt = ''
    for char in full_string :
        txt = txt + char
    file = open('mypoem.txt','w')
    file.write(txt)
    file.close() 

if __name__ == '__main__' :
    initialization()
    runPoem()
    savePoem()
