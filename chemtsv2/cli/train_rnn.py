import argparse
import os
import pickle

from tensorflow.keras.models import Sequential  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Dense, Embedding, GRU, TimeDistributed  # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam  # pyright: ignore[reportMissingImports]
from tensorflow.keras.utils import to_categorical  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing import sequence  # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint  # pyright: ignore[reportMissingImports]
import numpy as np
import yaml

from chemtsv2.preprocessing import read_smiles_dataset, tokenize_smiles


def get_parser():
    parser = argparse.ArgumentParser(description="", usage="chemtsv2-train-rnn -c CONFIG_FILE")
    parser.add_argument("-c", "--config", type=str, required=True, help="path to a config file")
    return parser.parse_args()


def prepare_data(smiles, all_smiles):
    """TODO: need to be refactored"""
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
        x2 = x1[1 : len(x1)]
        x2.append(0)
        y_train.append(x2)
    return X_train, y_train


def save_model(model, output_dir, use_selfies=False):
    output_json = os.path.join(
        output_dir, "model_sf.tf25.json" if use_selfies else "model.tf25.json"
    )
    output_weight = os.path.join(output_dir, "model_sf.tf25.h5" if use_selfies else "model.tf25.h5")
    model_json = model.to_json()

    with open(output_json, "w") as json_file:
        json_file.write(model_json)
    print(f"[INFO] Save a model structure to {output_json}")

    model.save_weights(output_weight)
    print(f"[INFO] Save model weights to {output_weight}")


def update_config(conf):
    conf.setdefault("dataset", "data/250k_rndm_zinc_drugs_clean.smi")
    conf.setdefault("output_model_dir", "model/user_trained_model")
    conf.setdefault("output_token", "model/user_trained_model/tokens.pkl")
    conf.setdefault("dropout_rate", 0.2)
    conf.setdefault("learning_rate", 0.01)
    conf.setdefault("epoch", 100)
    conf.setdefault("batch_size", 512)
    conf.setdefault("validation_split", 0.1)
    conf.setdefault("units", 256)
    conf.setdefault("rec_dropout_rate", 0)
    conf.setdefault("maxlen", 82)
    conf.setdefault("use_selfies", False)


def main():
    args = get_parser()

    # Setup configuration
    with open(args.config, "r") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    update_config(conf)
    print("========== Configuration ==========")
    for k, v in conf.items():
        print(f"{k}: {v}")
    print("===================================")

    os.makedirs(conf["output_model_dir"], exist_ok=True)

    # Prepare training dataset
    original_smiles_list = read_smiles_dataset(conf["dataset"])
    token_list, tokenized_smiles_list = tokenize_smiles(original_smiles_list, use_selfies=conf["use_selfies"])
    assert len(original_smiles_list) == len(tokenized_smiles_list)
    print(f"[INFO] Size of training dataset: {len(original_smiles_list)}")
    if conf["use_selfies"]:
        base, ext = os.path.splitext(conf["output_token"])
        conf["output_token"] = f"{base}_sf{ext}"
    with open(conf["output_token"], "wb") as f:
        pickle.dump(token_list, f)
    print(f"[INFO] Generated tokens: {token_list}")
    if conf["use_selfies"]:
        print(
            f"[INFO] Save generated tokens to {conf['output_token']}. "
            "Note that the file name was modified because `use_selfies` was specified."
        )
    else:
        print(f"[INFO] Save generated tokens to {conf['output_token']}")

    X_train, y_train = prepare_data(token_list, tokenized_smiles_list)
    X = sequence.pad_sequences(
        X_train,
        maxlen=conf["maxlen"],
        dtype="int32",
        padding="post",
        truncating="pre",
        value=0.0,
    )
    y = sequence.pad_sequences(
        y_train,
        maxlen=conf["maxlen"],
        dtype="int32",
        padding="post",
        truncating="pre",
        value=0.0,
    )
    y_train_one_hot = np.array([
        to_categorical(sent_label, num_classes=len(token_list)) for sent_label in y
    ])
    print(f"[DEBUG] Shape of y_train_one_hot: {y_train_one_hot.shape}")

    # Build model
    model = Sequential()
    model.add(
        Embedding(
            input_dim=len(token_list),
            output_dim=len(token_list),
            input_length=X.shape[1],
            mask_zero=False,
        )
    )
    model.add(
        GRU(
            conf["units"],
            input_shape=(X.shape[1], len(token_list)),
            activation="tanh",
            dropout=conf["dropout_rate"],
            recurrent_dropout=conf["rec_dropout_rate"],
            return_sequences=True,
        )
    )
    model.add(
        GRU(
            conf["units"],
            activation="tanh",
            dropout=conf["dropout_rate"],
            recurrent_dropout=conf["rec_dropout_rate"],
            return_sequences=True,
        )
    )
    model.add(
        TimeDistributed(
            Dense(
                len(token_list),
                activation="softmax",
            )
        )
    )
    model.summary()
    # Create callbacks
    log_path = os.path.join(conf["output_model_dir"], "training_log.csv")
    logger = CSVLogger(log_path)
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=5)
    model_ckpt = ModelCheckpoint(
        filepath=os.path.join(
            conf["output_model_dir"],
            "model_sf.tf25.best.ckpt.h5" if conf["use_selfies"] else "model.tf25.best.ckpt.h5",
        ),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        save_freq="epoch",
    )
    callbacks = [logger, early_stopping, model_ckpt]

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=conf["learning_rate"]),
        metrics=["accuracy"],
    )

    # Training
    _ = model.fit(
        X,
        y_train_one_hot,
        batch_size=conf["batch_size"],
        epochs=conf["epoch"],
        verbose=1,
        callbacks=callbacks,
        validation_split=conf["validation_split"],
        shuffle=True,
    )
    save_model(model, conf["output_model_dir"], use_selfies=conf["use_selfies"])
    print(f"[INFO] Save a training log to {log_path}")


if __name__ == "__main__":
    main()
