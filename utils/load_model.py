import os
from tensorflow.keras.models import model_from_json
os.environ["CUDA_VISIBLE_DEVICES"]= "-1" #"0,1,2,3,"                                                                                                                  


def loaded_model(model_json, model_weight, logger):
    with open(model_json, 'r') as f:
        loaded_model_json = f.read()
    loaded_model = model_from_json(loaded_model_json)
    logger.info(f"Loaded model_json from {model_json}")

    # load weights into new model
    loaded_model.load_weights(model_weight)
    logger.info(f"Loaded model_weight from {model_weight}")
    return loaded_model

