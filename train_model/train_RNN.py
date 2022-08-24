import argparse
import os
import pickle
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
import numpy as np
import yaml

sys.path.append("../chemtsv2")
from preprocessing import read_smiles_dataset, tokenize_smiles


def get_parser():
    parser = argparse.ArgumentParser(
        description="",
        usage=f"python {os.path.basename(__file__)} -c CONFIG_FILE"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="path to a config file"
    )
    return parser.parse_args()


def prepare_data(smiles, all_smiles):
    all_smiles_index = []
    for i in range(len(all_smiles)):
        smiles_index = []
        for j in range(len(all_smiles[i])):
            smiles_index.append(smiles.index(all_smiles[i][j]))
        all_smiles_index.append(smiles_index)
    X_train = all_smiles_index
    y_train = []
    for i in range(len(X_train)):
        x1 = X_train[i]
        x2 = x1[1:len(x1)]
        x2.append(0)
        y_train.append(x2)
    return X_train, y_train


def save_model(model, output_dir):
    output_json = os.path.join(output_dir, "model.tf25.json")
    output_weight = os.path.join(output_dir, "model.tf25.h5")
    model_json = model.to_json()

    with open(output_json, "w") as json_file:
        json_file.write(model_json)
    print(f"[INFO] Save a model structure to {output_json}")

    model.save_weights(output_weight)
    print(f"[INFO] Save model weights to {output_weight}")


def update_config(conf):
    conf.setdefault("dataset", "../data/250k_rndm_zinc_drugs_clean.smi")
    conf.setdefault('output_model_dir', "../model")
    conf.setdefault('output_token', "../model/tokens.pkl")
    conf.setdefault('dropout_rate', 0.2)
    conf.setdefault('learning_rate', 0.01)
    conf.setdefault('epoch', 100)
    conf.setdefault('batch_size', 512)
    conf.setdefault('validation_split', 0.1)
    conf.setdefault('units', 256)    
    conf.setdefault('rec_dropout_rate', 0)
    conf.setdefault('maxlen', 82)


def main():
    args = get_parser()
    
    # Setup configuration
    with open(args.config, "r") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    update_config(conf)
    print(f"========== Configuration ==========")
    for k, v in conf.items():
        print(f"{k}: {v}")
    print(f"===================================")

    os.makedirs(conf['output_model_dir'], exist_ok=True)

    # Prepare training dataset
    original_smiles = read_smiles_dataset(conf["dataset"])
    vocabulary, all_smiles = tokenize_smiles(original_smiles)
    with open(conf['output_token'], 'wb') as f:
        pickle.dump(vocabulary, f)
    print(f"[INFO] Save generated tokens to {conf['output_token']}")
        
    print(f"vocabulary:\n{vocabulary}\n"
          f"size of SMILES list: {len(all_smiles)}")
    X_train, y_train = prepare_data(vocabulary, all_smiles) 
    X = sequence.pad_sequences(X_train, maxlen=conf['maxlen'], dtype='int32', padding='post', truncating='pre', value=0.)
    y = sequence.pad_sequences(y_train, maxlen=conf['maxlen'], dtype='int32', padding='post', truncating='pre', value=0.)
    y_train_one_hot = np.array([to_categorical(sent_label, num_classes=len(vocabulary)) for sent_label in y])
    print(f"shape of y_train_one_hot: {y_train_one_hot.shape}")

    # Build model
    model = Sequential()
    model.add(Embedding(input_dim=len(vocabulary), output_dim=len(vocabulary), input_length=X.shape[1], mask_zero=False))
    model.add(GRU(conf['units'], input_shape=(X.shape[1], len(vocabulary)), activation='tanh', dropout=conf['dropout_rate'], recurrent_dropout=conf['rec_dropout_rate'], return_sequences=True))
    model.add(GRU(conf['units'], activation='tanh', dropout=conf['dropout_rate'], recurrent_dropout=conf['rec_dropout_rate'], return_sequences=True))
    model.add(TimeDistributed(Dense(len(vocabulary), activation='softmax')))
    model.summary()
    # Create callbacks
    log_path = os.path.join(conf['output_model_dir'], "training_log.csv")
    logger = CSVLogger(log_path)
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=5
    )
    model_ckpt = ModelCheckpoint(
        filepath = os.path.join(conf['output_model_dir'], "model.tf25.best.ckpt.h5"),
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        save_freq='epoch'
    )
    callbacks = [logger, early_stopping, model_ckpt]

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=conf['learning_rate']),
        metrics=['accuracy'],)

    # Training
    _ = model.fit(
        X, 
        y_train_one_hot,
        batch_size=conf['batch_size'],
        epochs=conf['epoch'],
        verbose=1,
        callbacks=callbacks,
        validation_split=conf['validation_split'],
        shuffle=True,)
    save_model(model, conf["output_model_dir"])
    print(f"[INFO] Save a training log to {log_path}")


if __name__ == "__main__":
    main()
