import deepchem as dc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plot
import copy, yaml, argparse
from hyperopt import hp, fmin, tpe, rand, atpe, Trials, STATUS_OK
import statistics, time
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, DEBUG
from tqdm import tqdm
import csv

## functions
class DataLoader:
    def loadCSV(config):
        infile = config["infile"]
        target = config["target"] #説明変数（SMILES）
        tasks = config["tasks"] #目的変数（活性値等）
        featurizer = config['featurizer']
    
        # initizalize the dataloader
        loader = dc.data.CSVLoader(tasks, feature_field=target, featurizer=featurizer)
        # load and featurize the data from the CSV file
        dataset = loader.create_dataset(infile)
        return dataset

class DataSplitter:
    def train_valid_test_split(dataset, config):
       
        train_dataset, valid_dataset, test_dataset = config['type'].train_valid_test_split(
            dataset=dataset, frac_train=config['rate']['train'], frac_valid=config['rate']['valid'], frac_test=config['rate']['test']
        )
        datasets = dict(train=train_dataset, valid=valid_dataset, test=test_dataset)
        
        return datasets

    def train_test_split(dataset, config):

        train_dataset, test_dataset = config['type'].train_test_split(
            dataset=dataset, frac_train=config['rate']['train']
        )
        datasets = dict(train=train_dataset, test=test_dataset)
        
        return datasets

    def k_fold_split(dataset, config):

        folds = config['type'].k_fold_split(
            dataset=dataset, k=config['rate']['k']
        )
        
        return folds

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config", type=str, default=None, help="Config file path")
parser.add_argument("-opt", "--optimize", action="store_true", help="Perform Hyperparameter Optimization")
parser.add_argument("-v", "--verbose", action="store_true", help="For debug")


args = parser.parse_args()
# args = parser.parse_args(args=['-c', 'predict.yaml', '-v'])

logger = getLogger(__name__)
ch = StreamHandler()
formatter = Formatter("%(asctime)s : %(levelname)s : %(message)s ")

if args.verbose:
    print("verbosity turned on")
    logger.setLevel(DEBUG)
    ch.setLevel(DEBUG)

else:
    logger.setLevel(INFO)
    ch.setLevel(INFO)

ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info("Loading config...")

with open(args.config,'r') as f:
    config = yaml.safe_load(f)

params = copy.deepcopy(config)

params['dataloader']['featurizer'] = getattr(dc.feat,config['dataloader']['featurizer']['type'])(config['dataloader']['featurizer']['kwargs'])
params['splitter']['type'] = getattr(dc.splits,config['splitter']['type'])()

logger.debug("config:\n %s", config)
logger.info("params:\n %s",params)


logger.info("Loading data...")

infile = params["dataloader"]["infile"]

df = pd.read_csv(infile)
smiles = df['smiles'].values

featurizer = params['dataloader']['featurizer']
features = featurizer.featurize(smiles)

dataset = dc.data.NumpyDataset(X=features,ids=smiles)

# dataset.to_dataframe()

metrics = [ dc.metrics.Metric(getattr(dc.metrics, metric)) for metric in params['training']['metrics'] ]
valid_metric = dc.metrics.Metric(getattr(dc.metrics, params['training']['valid_metric'])) 

model = dc.models.GraphConvModel(
            n_tasks = params['training']['n_tasks'],
            graph_conv_layers = params['training']['graph_conv_layers'],
            dense_layer_size = params['training']['dense_layer_size'],
            dropout = params['training']['dropout'],
            mode = params['training']['mode'],
            number_atom_features = params['training']['number_atom_features'],
            n_classes = params['training']['n_classes'],
            batch_size = params['training']['batch_size'],
            batch_normalize = params['training']['batch_normalize'],
            uncertainty = params['training']['uncertainty'],
            model_dir = params['training']['model_dir']
)

model.restore()

logger.info(dataset)

y_pred, y_std = model.predict_uncertainty(dataset)
# y_pred = transformers[0].untransform(y_pred)

df = pd.DataFrame(list(zip(dataset.ids, y_pred)), columns=['smiles', 'predict_value'])
print(df)
df.to_csv(params['training']['model_dir']+'/predict.csv', index=False)

#model.restore()

