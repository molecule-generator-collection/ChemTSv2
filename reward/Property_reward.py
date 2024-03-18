import deepchem as dc
import numpy as np
import pandas as pd
import tensorflow as tf
#import matplotlib.pyplot as plot
import copy, yaml
import statistics, time
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, DEBUG
from tqdm import tqdm
import os
import sklearn
from sklearn.ensemble import *

import mordred.descriptors
from rdkit import Chem
import tempfile

# import torch

## functions
class DataLoader:
    def loadCSV(config):
        infile = config["infile"]
        target = config["target"]
        tasks = config["tasks"]
        featurizer = config['featurizer']
    
        # initizalize the dataloader
        if params['dataloader']['tasks'] is not None and type(params['dataloader']['tasks']) is list and len(params['dataloader']['tasks']) >0:
            loader = dc.data.CSVLoader(tasks, feature_field=target, featurizer=featurizer)
            # load and featurize the data from the CSV file
            dataset = loader.create_dataset(infile)
        else:
            print('Load dataset without any task')              
            loader = dc.data.CSVLoader(tasks=[], feature_field=target, id_field=target, featurizer=featurizer)
            # load and featurize the data from the CSV file
            dataset = loader.create_dataset(infile)
        return dataset

class DataSplitter:
    def train_valid_test_split(dataset, config):
        seed = None
        if 'seed' in config.keys():
            if config['seed'] is not None:
                seed = config['seed']
        print('seed for splitter = ', seed)
        train_dataset, valid_dataset, test_dataset = config['type'].train_valid_test_split(
            dataset=dataset, seed=seed, frac_train=config['rate']['train'], frac_valid=config['rate']['valid'], frac_test=config['rate']['test']
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

class ModelBuilder:
    def buildModel(config,model_dir):
        if config['model']['type'] == 'SklearnModel' and 'sklearn_model' in config['model']['kwargs']:
            config['model']['kwargs']['model'] = getattr(sklearn.ensemble, config['model']['kwargs']['sklearn_model'])()

        config['model']['kwargs']['model_dir'] = model_dir # for hyperparam opt
        model = getattr(dc.models,config['model']['type'])(**config['model']['kwargs'])
        
        return model


class Property_reward(Reward):
    def get_objective_functions(conf):
        def PropValue(mol):
            
            if mol is None:
                return None
            
            params = copy.deepcopy(conf['deepchem'])
            
            MODEL_DIR = model_dir = params['training']['model']['kwargs']['model_dir']
            print(f"[INFO] loaded model from {MODEL_DIR}")
            
            # Model
            model = ModelBuilder.buildModel(params['training'], model_dir)
            model.restore()
            
            ### Fearutirzerの
            if config['dataloader']['featurizer']['kwargs'] is not None:
                if config['dataloader']['featurizer']['type'] in ['RobertaFeaturizer','HuggingFaceFeaturizer']:
                    featurizer = getattr(dc.feat,config['dataloader']['featurizer']['type']).from_pretrained(config['dataloader']['featurizer']['kwargs']['pretrained'])
                else:
                    featurizer = getattr(dc.feat,config['dataloader']['featurizer']['type'])(**config['dataloader']['featurizer']['kwargs'])
               
            else:
                featurizer = getattr(dc.feat,params['dataloader']['featurizer']['type'])()

            ### configure RDKit　descriptors
            if config['dataloader']['featurizer']['type'] == 'RDKitDescriptors':
                featurizer.descriptors = config['dataloader']['featurizer']['descriptors']
                desc_list = Chem.Descriptors.descList
                featurizer.descList = []
                try:
                    for desc_name, function in desc_list:
                        if desc_name in featurizer.descriptors:
                            featurizer.descList.append((desc_name, function))
                            
                except ModuleNotFoundError:
                    raise ImportError("This class requires RDKit to be installed.")

            #### configure Mordred descriptors
            if config['dataloader']['featurizer']['type'] == 'MordredDescriptors':
                featurizer.descriptors = config['dataloader']['featurizer']['descriptors']
                try:
                    featurizer.is_missing = mordred.is_missing
                    featurizer.calc = mordred.Calculator(mordred.descriptors, ignore_3D=True)
                    featurizer.calc.descriptors = [d for d in featurizer.calc.descriptors if str(d) in featurizer.descriptors]
                    
                except ModuleNotFoundError:
                    raise ImportError("This class requires Mordred to be installed.")

            params['dataloader']['featurizer'] = featurizer
            print("params:\n %s",params)
            
            smiles = [Chem.MolToSmiles(mol)]
            featurizer = params['featurizer']
            features = featurizer.featurize(smiles)
            data_for_pred = dc.data.NumpyDataset(X=features,ids=smiles)
            
            print('preprocessed_data:', data_for_pred)
            
            
            if params['training']['model']['type'] in ['KerasModel', 'GraphConvModel']:
                if params['training']['model']['kwargs']['mode'] == 'regression':
                    y_pred, y_std = model.predict_uncertainty(data_for_pred)
                    for transform in transformers:
                        y_pred = transform.untransform(y_pred)

                    df = pd.DataFrame(list(zip(data_for_pred.ids, y_pred[:,0])), columns=['smiles', 'predict'])
                    logger.info('%s', df)
                    
                    pred_value = y_pred[:,0]
                
                elif params['training']['model']['kwargs']['mode'] == 'classification':
                    y_prob = model.predict(data_for_pred,transformers=transformers)
                    y_score = y_prob[:,:,1]
                    threshold = 0.5
                    y_pred  = (y_score > threshold).astype(np.int8)
                    print('threshold:', threshold)
                    df = pd.DataFrame(list(zip(data_for_pred.ids, y_pred[:,0], y_prob[:,0,:])), columns=['smiles', 'predict', 'probability'])
                    logger.info('%s', df.to_string(index=False))
                    
                    pred_value = y_pred[:,0]
                        

            if params['training']['model']['type'] == 'SklearnModel':
                if params['training']['model']['kwargs']['mode'] == 'regression':
                    df = pd.DataFrame(list(zip(data_for_pred.ids, y_pred)), columns=['smiles', 'predict'])
                    logger.info('%s', df.to_string(index=False))
                    
                    pred_value = y_pred
                        
                elif params['training']['model']['kwargs']['mode'] == 'classification':
                    y_prob = model.predict(data_for_pred,transformers=transformers)
                    y_score = y_prob[:,:,1]
                    threshold = 0.5            
                    y_pred  = (y_score > threshold).astype(np.int8)
                    print('threshold:', threshold)
                    df = pd.DataFrame(list(zip(data_for_pred.ids, y_pred, y_prob)), columns=['smiles', 'predict', 'probability'])
                    logger.info('%s', df.to_string(index=False))
                   
                    pred_value = y_pred
                    

            return pred_value
        return [PropValue]

    def calc_reward_from_objective_values(values, conf):
        #return np.tanh(values[0]/10) if None not in values else -1
        return values[0] if None not in values else -999
















 parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config", type=str, default=None, help="Config file path")
parser.add_argument("-opt", "--optimize", action="store_true", help="Perform Hyperparameter Optimization")
parser.add_argument("-o", "--out_csv", type=str, default='./predict.csv', help="Path to prediction result file name (CSV)")
parser.add_argument("-v", "--verbose", action="store_true", help="For debug")


args = parser.parse_args()
# args = parser.parse_args(args=['-c', './config/predict_KerasGCN-reg.yaml', '-v'])

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

# params['dataloader']['featurizer'] = getattr(dc.feat,config['dataloader']['featurizer']['type'])(config['dataloader']['featurizer']['kwargs'])
## FeaturizerはCallableに変換する
### Fearutirzerの引数有り無し
if config['dataloader']['featurizer']['kwargs'] is not None:
    if config['dataloader']['featurizer']['type'] in ['RobertaFeaturizer','HuggingFaceFeaturizer']:
        featurizer = getattr(dc.feat,config['dataloader']['featurizer']['type']).from_pretrained(config['dataloader']['featurizer']['kwargs']['pretrained'])
        # out = featurizer(smiles, add_special_tokens=True, truncation=True)
    else:
        # params['dataloader']['featurizer'] = getattr(dc.feat,config['dataloader']['featurizer']['type'])(config['dataloader']['featurizer']['kwargs'])
        featurizer = getattr(dc.feat,config['dataloader']['featurizer']['type'])(**config['dataloader']['featurizer']['kwargs'])
   
else:
    featurizer = getattr(dc.feat,params['dataloader']['featurizer']['type'])()

### RDKit記述子の選択
if config['dataloader']['featurizer']['type'] == 'RDKitDescriptors':
    featurizer.descriptors = config['dataloader']['featurizer']['descriptors']
    desc_list = Chem.Descriptors.descList
    featurizer.descList = []
    try:
        for desc_name, function in desc_list:
            if desc_name in featurizer.descriptors:
                featurizer.descList.append((desc_name, function))
                
    except ModuleNotFoundError:
        raise ImportError("This class requires RDKit to be installed.")

#### Mordred記述子の選択
if config['dataloader']['featurizer']['type'] == 'MordredDescriptors':
    featurizer.descriptors = config['dataloader']['featurizer']['descriptors']
    try:
        featurizer.is_missing = mordred.is_missing
        featurizer.calc = mordred.Calculator(mordred.descriptors, ignore_3D=True)
        featurizer.calc.descriptors = [d for d in featurizer.calc.descriptors if str(d) in featurizer.descriptors]
        
    except ModuleNotFoundError:
        raise ImportError("This class requires Mordred to be installed.")

### featurizerをparamsに代入
params['dataloader']['featurizer'] = featurizer

## SplitterをCallableに変換
params['splitter']['type'] = getattr(dc.splits,config['splitter']['type'])()
params['training']['model']['kwargs']['optimizer'] = getattr(dc.models.optimizers, config['training']['model']['kwargs']['optimizer'])()

logger.debug("config:\n %s", config)
logger.info("params:\n %s",params)



# params['splitter']['type'] = getattr(dc.splits,config['splitter']['type'])()
# params['training']['optimizer'] = getattr(dc.models.optimizers, config['training']['optimizer'])()

# logger.debug("config:\n %s", config)
# logger.info("params:\n %s",params)

logger.info("Loading data...")
dataset = DataLoader.loadCSV(params['dataloader'])
logger.debug(dataset)

metrics = [ dc.metrics.Metric(getattr(dc.metrics, metric)) for metric in params['training']['metrics'] ]
valid_metric = dc.metrics.Metric(getattr(dc.metrics, params['training']['valid_metric'])) 

# Model
model_dir = params['training']['model']['kwargs']['model_dir']
model = ModelBuilder.buildModel(params['training'], model_dir)

model.restore()

logger.info(dataset)

load_dataset_dir = params['dataloader']['save_dataset']

loaded, all_dataset, transformers = dc.utils.load_dataset_from_disk(save_dir=load_dataset_dir)

data_for_pred = dataset
if params['training']['model']['type'] in ['KerasModel', 'GraphConvModel']:
    if params['training']['model']['kwargs']['mode'] == 'regression':
        y_pred, y_std = model.predict_uncertainty(data_for_pred)
        for transform in transformers:
            y_pred = transform.untransform(y_pred)

        if params['dataloader']['tasks'] is not None and type(params['dataloader']['tasks']) is list and len(params['dataloader']['tasks']) >0:
            
            # df = pd.DataFrame(list(zip(data_for_pred.ids, data_for_pred.y[:,0], y_pred[:,0], y_std[:,0])), columns=['smiles', 'actural', 'predict', 'std_dev'])
            df = pd.DataFrame(list(zip(data_for_pred.ids, data_for_pred.y[:,0], y_pred[:,0], y_std[:,0])), columns=['smiles', 'actual', 'predict', 'std_dev'])
            logger.info('%s', df)
        else:
            df = pd.DataFrame(list(zip(data_for_pred.ids, y_pred[:,0], y_std[:,0])), columns=['smiles', 'predict', 'std_dev'])
            
            logger.info('%s', df)
    
    elif params['training']['model']['kwargs']['mode'] == 'classification':
        if params['dataloader']['tasks'] is not None and type(params['dataloader']['tasks']) is list and len(params['dataloader']['tasks']) >0:
            y_true = data_for_pred.y[:,0]
            y_prob = model.predict(data_for_pred,transformers=transformers)
            y_score = y_prob[:,:,1]
            roc_auc_score = dc.metrics.roc_auc_score(y_true,y_pred)
            print('roc_auc_score:',roc_auc_score)
            threshold = 0.5
            y_pred  = (y_score > threshold).astype(np.int8)
            print('threshold:', threshold)
            df = pd.DataFrame(list(zip(data_for_pred.ids, y_true.astype(np.int8), y_pred[:,0], y_prob[:,0,:])), columns=['smiles', 'true', 'predict', 'probability'])
            logger.info('%s', df)
        else:
            y_prob = model.predict(data_for_pred,transformers=transformers)
            y_score = y_prob[:,:,1]
            threshold = 0.5
            y_pred  = (y_score > threshold).astype(np.int8)
            print('threshold:', threshold)
            df = pd.DataFrame(list(zip(data_for_pred.ids, y_pred[:,0], y_prob[:,0,:])), columns=['smiles', 'predict', 'probability'])
            logger.info('%s', df.to_string(index=False))
            

if params['training']['model']['type'] == 'SklearnModel':
    if params['training']['model']['kwargs']['mode'] == 'regression':
        if params['dataloader']['tasks'] is not None and type(params['dataloader']['tasks']) is list and len(params['dataloader']['tasks']) >0:
            y_pred = model.predict(data_for_pred,transformers=transformers)
            df = pd.DataFrame(list(zip(data_for_pred.ids, data_for_pred.y, y_pred)), columns=['smiles', 'actural', 'predict'])
            logger.info('%s', df)
        else:
            # df = pd.DataFrame(list(zip(data_for_pred.ids, data_for_pred.y[:,0], y_pred[:,0], y_std[:,0])), columns=['smiles', 'actural', 'predict', 'std_dev'])
            df = pd.DataFrame(list(zip(data_for_pred.ids, y_pred)), columns=['smiles', 'predict'])
            logger.info('%s', df.to_string(index=False))
            
    elif params['training']['model']['kwargs']['mode'] == 'classification':
        if params['dataloader']['tasks'] is not None and type(params['dataloader']['tasks']) is list and len(params['dataloader']['tasks']) >0:
            y_true = data_for_pred.y[:,0]
            y_prob = model.predict(data_for_pred,transformers=transformers)
            y_score = y_prob[:,1]
            roc_auc_score = dc.metrics.roc_auc_score(y_true,y_pred)
            print('roc_auc_score:',roc_auc_score)
            threshold = 0.5
            y_pred  = (y_score > threshold).astype(np.int8)
            print('threshold:', threshold)
            df = pd.DataFrame(list(zip(data_for_pred.ids, y_true.astype(np.int8), y_pred, y_prob)), columns=['smiles', 'true', 'predict', 'probability'])
            logger.info('%s', df)
        else:

            y_prob = model.predict(data_for_pred,transformers=transformers)
            y_score = y_prob[:,:,1]
            threshold = 0.5            
            y_pred  = (y_score > threshold).astype(np.int8)
            print('threshold:', threshold)
            df = pd.DataFrame(list(zip(data_for_pred.ids, y_pred, y_prob)), columns=['smiles', 'predict', 'probability'])
            logger.info('%s', df.to_string(index=False))

df.to_csv(args.out_csv, index=False)