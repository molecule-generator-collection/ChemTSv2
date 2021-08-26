import csv
import itertools
import operator
import numpy as np
import nltk
import os
import sys
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation,TimeDistributed
from keras.layers import LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers import Dropout
import numpy as np
import random
import sys
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from make_smile import ziprocess_organic,process_zinc_data
from make_smile import zinc_data_with_bracket_original,zinc_processed_with_bracket

from keras.layers import Conv1D, MaxPooling1D
#from combine_bond_atom import organic, process_organic,bond_atom

import yaml

import matplotlib as mpl
mpl.use('Agg')  ## fix the "Invalid DISPLAY variable" Error
import matplotlib.pyplot as plt

def load_data():

    sen_space=[]
    #f = open('/Users/yang/smiles.csv', 'rb')
    f = open(dataset, 'rb')
    reader = csv.reader(f)
    for row in reader:
        #word_space[row].append(reader[row])
        #print word_sapce
        sen_space.append(row)
    #print sen_space
    f.close()

    element_table=["Cu","Ti","Zr","Ga","Ge","As","Se","Br","Si","Zn","Cl","Be","Ca","Na","Sr","Ir","Li","Rb","Cs","Fr","Be","Mg",
            "Ca","Sr","Ba","Ra","Sc","La","Ac","Ti","Zr","Nb","Ta","Db","Cr","Mo","Sg","Mn","Tc","Re","Bh","Fe","Ru","Os","Hs","Co","Rh",
            "Ir","Mt","Ni","Pd","Pt","Ds","Cu","Ag","Au","Rg","Zn","Cd","Hg","Cn","Al","Ga","In","Tl","Nh","Si","Ge","Sn","Pb","Fl",
            "As","Sb","Bi","Mc","Se","Te","Po","Lv","Cl","Br","At","Ts","He","Ne","Ar","Kr","Xe","Rn","Og"]
    #print sen_space
    word1=sen_space[0]
    word_space=list(word1[0])
    end="\n"
    #start="st"
    #word_space.insert(0,end)
    word_space.append(end)
    #print word_space
    #print len(sen_space)
    all_smile=[]
    #print word_space
    #for i in range(len(all_smile)):

    for i in range(len(sen_space)):
        word1=sen_space[i]
        word_space=list(word1[0])
        word=[]
        #word_space.insert(0,end)
        j=0
        while j<len(word_space):
            word_space1=[]
            word_space1.append(word_space[j])
            if j+1<len(word_space):
                word_space1.append(word_space[j+1])
                word_space2=''.join(word_space1)
            else:
                word_space1.insert(0,word_space[j-1])
                word_space2=''.join(word_space1)
            if word_space2 not in element_table:
                word.append(word_space[j])
                j=j+1
            else:
                word.append(word_space2)
                j=j+2

        word.append(end)
        all_smile.append(list(word))
    #print all_smile
    val=[]
    for i in range(len(all_smile)):
        for j in range(len(all_smile[i])):
            if all_smile[i][j] not in val:
                val.append(all_smile[i][j])
    #print val
    val.remove("\n")
    val.insert(0,"\n")

    return val, all_smile


def organic_data():
    sen_space=[]
    #f = open('/Users/yang/smiles.csv', 'rb')
    f = open('/Users/yang/LSTM-chemical-project/make_sm.csv', 'rb')
    #f = open('/Users/yang/LSTM-chemical-project/smile_trainning.csv', 'rb')
    reader = csv.reader(f)
    for row in reader:
        #word_space[row].append(reader[row])
        #print word_sapce
        sen_space.append(row)
    #print sen_space
    f.close()

    element_table=["Cu","Ti","Zr","Ga","Ge","As","Se","Br","Si","Zn","Cl","Be","Ca","Na","Sr","Ir","Li","Rb","Cs","Fr","Be","Mg",
            "Ca","Sr","Ba","Ra","Sc","La","Ac","Ti","Zr","Nb","Ta","Db","Cr","Mo","Sg","Mn","Tc","Re","Bh","Fe","Ru","Os","Hs","Co","Rh",
            "Ir","Mt","Ni","Pd","Pt","Ds","Cu","Ag","Au","Rg","Zn","Cd","Hg","Cn","Al","Ga","In","Tl","Nh","Si","Ge","Sn","Pb","Fl",
            "As","Sb","Bi","Mc","Se","Te","Po","Lv","Cl","Br","At","Ts","He","Ne","Ar","Kr","Xe","Rn","Og"]
    #print sen_space
    word1=sen_space[0]
    word_space=list(word1[0])
    end="\n"
    #start="st"
    #word_space.insert(0,end)
    word_space.append(end)
    #print word_space
    #print len(sen_space)
    all_smile=[]
    #print word_space
    #for i in range(len(all_smile)):

    for i in range(len(sen_space)):
        word1=sen_space[i]
        word_space=list(word1[0])
        word=[]
        #word_space.insert(0,end)
        j=0
        while j<len(word_space):
            word_space1=[]
            word_space1.append(word_space[j])
            if j+1<len(word_space):
                word_space1.append(word_space[j+1])
                word_space2=''.join(word_space1)
            else:
                word_space1.insert(0,word_space[j-1])
                word_space2=''.join(word_space1)
            if word_space2 not in element_table:
                word.append(word_space[j])
                j=j+1
            else:
                word.append(word_space2)
                j=j+2

        word.append(end)
        all_smile.append(list(word))
    #print all_smile
    val=[]
    for i in range(len(all_smile)):
        for j in range(len(all_smile[i])):
            if all_smile[i][j] not in val:
                val.append(all_smile[i][j])
    #print val
    val.remove("\n")
    val.insert(0,"\n")

    return val, all_smile


def prepare_data(smiles,all_smile):
    all_smile_index=[]
    for i in range(len(all_smile)):
        smile_index=[]
        for j in range(len(all_smile[i])):
            smile_index.append(smiles.index(all_smile[i][j]))
        all_smile_index.append(smile_index)
    X_train=all_smile_index
    y_train=[]
    for i in range(len(X_train)):

        x1=X_train[i]
        x2=x1[1:len(x1)]
        x2.append(0)
        y_train.append(x2)

    return X_train,y_train


def generate_smile(model,val):
    end="\n"
    start_smile_index= [val.index("C")]
    new_smile=[]

    while not start_smile_index[-1] == val.index(end):
        predictions=model.predict(start_smile_index)
        ##next atom probability
        smf=[]
        for i in range (len(X)):
            sm=[]
            for j in range(len(X[i])):
                #if np.argmax(predictions[i][j])=!0
                sm.append(np.argmax(predictions[i][j]))
            smf.append(sm)

        #print sm
        #print smf
        #print len(sm)

        new_smile.append(sampled_word)
    #sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    #return new_sentence



def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    #with open("model.json", "w") as json_file:
    with open(output_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    #model.save_weights("model.h5")
    model.save_weights(output_weight)
    print("Saved model to disk")

def save_model_ES(model):
    model_json = model.to_json()
    with open(output_json, "w") as json_file:
        json_file.write(model_json)
    print("Saved model to disk")

if __name__ == "__main__":
    argvs = sys.argv
    argc = len(argvs)
    if argc == 1:
        print("input configuration file")
        exit()
    f = open(str(argvs[1]), "r+")
    conf = yaml.load(f)
    f.close()
    dataset = conf.get('dataset')
    output_json = conf.get('output_json')
    output_weight = conf.get('output_weight')
    dropout_rate = conf.get('dropout_rate',0.2)
    lr_val = conf.get('learning_rate',0.01)
    epochs = conf.get('epoch',100)
    batch_size = conf.get('batch_size',512)
    validation_split = conf.get('validation_split', 0.1)
    units = conf.get('units', 256)    
    rec_dropout_rate = conf.get('rec_dropout_rate', 0.2)
 
    print('========== display configuration ==========')
    print('dataset = ',dataset)
    print('output_json = ',output_json)
    print('output_weight = ',output_weight)
    print('dropout_rate = ',dropout_rate)
    print('learning_rate = ',lr_val)
    print('epoch = ',epochs)
    print('batch_size = ',batch_size)
    print('validation_split = ',validation_split)
    print('units = ', units)    
    print('rec_dropout_rate = ', rec_dropout_rate)

    smile=zinc_data_with_bracket_original(dataset)
    valcabulary,all_smile=zinc_processed_with_bracket(smile)
    print(valcabulary)
    print(len(all_smile))
    X_train,y_train=prepare_data(valcabulary,all_smile) 
  
    maxlen=82

    X= sequence.pad_sequences(X_train, maxlen=82, dtype='int32',
        padding='post', truncating='pre', value=0.)
    y = sequence.pad_sequences(y_train, maxlen=82, dtype='int32',
        padding='post', truncating='pre', value=0.)
    
    y_train_one_hot = np.array([to_categorical(sent_label, num_classes=len(valcabulary)) for sent_label in y])
    print(y_train_one_hot.shape)

    vocab_size=len(valcabulary)
    embed_size=len(valcabulary)

    
    N=X.shape[1]


    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=len(valcabulary), input_length=N,mask_zero=False))
    model.add(GRU(output_dim=units, input_shape=(82,64),activation='tanh', dropout = dropout_rate, recurrent_dropout = rec_dropout_rate,return_sequences=True))
    #model.add(LSTM(output_dim=256, input_shape=(82,64),activation='tanh',return_sequences=True))
    #model.add(Dropout(dropout_rate))
    model.add(GRU(units,activation='tanh',dropout = dropout_rate, recurrent_dropout = rec_dropout_rate,return_sequences=True))
    #model.add(LSTM(output_dim=1000, activation='sigmoid',return_sequences=True))
    #model.add(Dropout(dropout_rate))
    model.add(TimeDistributed(Dense(embed_size, activation='softmax')))
    optimizer=Adam(lr_val)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    if epochs == -1:
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
        checkpointer = ModelCheckpoint(filepath = output_weight, verbose = 1, save_weights_only = True, save_best_only = True)
        result = model.fit(X,y_train_one_hot,batch_size = batch_size,epochs=100,verbose = 1, callbacks = [early_stopping, checkpointer], validation_split = validation_split, shuffle = True)
        save_model_ES(model)
    else:
        result = model.fit(X,y_train_one_hot,batch_size,epochs,1,None,validation_split, shuffle = True)
        save_model(model)

    ## plot the training acc
    #%matplotlib inline
    try:
        # before keras 2.0.5
        plt.plot(range(1, len(result.history['acc'])+1), result.history['acc'], label="training")
        plt.plot(range(1, len(result.history['val_acc'])+1), result.history['val_acc'], label="validation")
    except KeyError:
        # after keras-2.3.1
        plt.plot(range(1, len(result.history['accuracy'])+1), result.history['accuracy'], label="training")
        plt.plot(range(1, len(result.history['val_accuracy'])+1), result.history['val_accuracy'], label="validation")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('learning_curve_GRU'+dataset.split('/')[-1]+'_units'+str(units)+'_dropout'+str(dropout_rate)+'_recDP'+str(rec_dropout_rate)+'_lr'+str(lr_val)+'_batchsize'+str(batch_size)+'.png', dpi = 300,  bbox_inches='tight', pad_inches=0)
                     
