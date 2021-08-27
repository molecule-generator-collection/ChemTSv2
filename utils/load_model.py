import os
from keras.models import model_from_json
os.environ["CUDA_VISIBLE_DEVICES"]= "1" #"0,1,2,3,"                                                                                                                  


def loaded_model(filename):
    with open(f"{filename}.json", 'r') as f:
        loaded_model_json = f.read()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(filename+'.h5')
    print("Loaded model from disk")
    return loaded_model

