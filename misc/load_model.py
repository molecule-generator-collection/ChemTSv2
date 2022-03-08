import sys

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Embedding, GRU


def get_model_structure_info(model_json, logger):
    with open(model_json, 'r') as f:
        loaded_model_json = f.read()
    loaded_model = model_from_json(loaded_model_json)
    logger.info(f"Loaded model_json from {model_json}")
    input_shape = None
    vocab_size = None
    output_size = None
    for layer in loaded_model.get_config()['layers']:
        config = layer.get('config')
        if layer.get('class_name') == 'InputLayer':
            input_shape = config['batch_input_shape'][1]
        if layer.get('class_name') == 'Embedding':
            vocab_size = config['input_dim']
        if layer.get('class_name') == 'TimeDistributed':
            output_size = config['layer']['config']['units']
    if input_shape is None or vocab_size is None or output_size is None:
        logger.error('Consult with ChemTSv2 developers on the GitHub repository. At that time, please attach the file specified as `model_json`')
        sys.exit()
            
    return input_shape, vocab_size, output_size

    
def loaded_model(model_weight, logger, conf):
    model = Sequential()
    model.add(Embedding(input_dim=conf['rnn_vocab_size'], output_dim=conf['rnn_vocab_size'],
                        mask_zero=False, batch_size=1))
    model.add(GRU(256, batch_input_shape=(1, None, conf['rnn_vocab_size']), activation='tanh',
                  return_sequences=True, stateful=True))
    model.add(GRU(256, activation='tanh', return_sequences=False, stateful=True))
    model.add(Dense(conf['rnn_output_size'], activation='softmax'))
    model.load_weights(model_weight)
    logger.info(f"Loaded model_weight from {model_weight}")

    return model