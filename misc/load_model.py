from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Embedding, GRU



def _loaded_model(model_json, model_weight, logger):
    with open(model_json, 'r') as f:
        loaded_model_json = f.read()
    loaded_model = model_from_json(loaded_model_json)
    logger.info(f"Loaded model_json from {model_json}")

    # load weights into new model
    loaded_model.load_weights(model_weight)
    logger.info(f"Loaded model_weight from {model_weight}")
    return loaded_model


def loaded_model(_, model_weight, logger):
    VOCAB_SIZE = 64
    EMBED_SIZE = 64

    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=VOCAB_SIZE,
                        mask_zero=False, batch_size=1))
    model.add(GRU(256, batch_input_shape=(1, None, VOCAB_SIZE), activation='tanh',
                  return_sequences=True, stateful=True))
    model.add(GRU(256, activation='tanh', return_sequences=False, stateful=True))
    model.add(Dense(EMBED_SIZE, activation='softmax'))
    model.load_weights(model_weight)
    logger.info(f"Loaded model_weight from {model_weight}")

    return model