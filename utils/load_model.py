import os
from keras.models import model_from_json
os.environ["CUDA_VISIBLE_DEVICES"]= "1" #"0,1,2,3,"                                                                                                                  


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


def loaded_model(filename):
    json_file = open(filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(filename+'.h5')
    print("Loaded model from disk")
    return loaded_model


# TODO: no longer used?
def loaded_activity_model():
    json_file = open('/Users/yang/LSTM-chemical-project/protein-ligand/ppara_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights('/Users/yang/LSTM-chemical-project/protein-ligand/ppara_model.hdf5')
    print("Loaded model from disk")

    return loaded_model
