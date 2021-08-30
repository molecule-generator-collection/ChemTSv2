import csv
import sys

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  ## fix the "Invalid DISPLAY variable" Error
import matplotlib.pyplot as plt
import yaml

sys.path.append("../")
from utils.make_smiles import zinc_data_with_bracket_original, zinc_processed_with_bracket


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


def save_model(model):
    model_json = model.to_json()
    with open(output_json, "w") as json_file:
        json_file.write(model_json)
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
    model.add(GRU(units, input_shape=(82,64),activation='tanh', dropout = dropout_rate, recurrent_dropout = rec_dropout_rate,return_sequences=True))
    model.add(GRU(units, activation='tanh',dropout = dropout_rate, recurrent_dropout = rec_dropout_rate,return_sequences=True))
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
