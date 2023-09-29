# GCN Prediction using Keras
##### ConvMolFeaturizer, GraphConvModel

import deepchem as dc
import numpy as np
import pandas as pd
import tensorflow as tf
#import matplotlib.pyplot as plot
import copy, yaml, argparse
from hyperopt import hp, fmin, tpe, rand, atpe, Trials, STATUS_OK
import statistics, time
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, DEBUG
from tqdm import tqdm

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

def cv(args):
    
    print('Hyperparameter Optimization Trial', trials.tids[-1])

    train_scores = {}
    valid_scores = {}

    # All metrics
    metrics = [ dc.metrics.Metric(getattr(dc.metrics, metric)) for metric in params['training']['metrics'] ]
    
    # metrics for optimize
    valid_metric = dc.metrics.Metric(getattr(dc.metrics, params['training']['valid_metric'])) 

    # Training/Evaluate for each subsets
    for i, fold in enumerate(subsets):
       
        model_dir = params['training']['model_dir']+'/trial_'+str(trials.tids[-1])+'/fold_'+str(i)
        train_set, valid_set = fold

        # transformers
        transformers = [ getattr(dc.trans, t)(transform_y=True, dataset=train_set) for t in params["dataloader"]["transformers"]]

        for transformer in transformers:
            train_set = transformer.transform(train_set)
            valid_set = transformer.transform(valid_set)
        
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
                    model_dir = model_dir,
                    learning_rate=params['training']['learning_rate'],
                    optimizer = params['training']['optimizer']

        )
        
        if params['training']['valid_metric'] in ["r2_score","pearson_r2_score"]:
            save_on_minimum=False
        elif params['training']['valid_metric'] in ["mean_squared_error","mean_absolute_error","rms_score","mae_score"]:
            save_on_minimum=True
        
        
        validation = dc.models.ValidationCallback(dataset=valid_set, 
                                                  interval=params['training']['valid_interval'], metrics=metrics,
                                                  save_metric=params['training']['metrics'].index(params['training']['valid_metric']),
                                                  save_dir=model_dir, 
                                                  #transformers=transformers,
                                                  save_on_minimum=save_on_minimum
                                                  #output_file=open(params['training']['model_dir']+'/fold_'+str(i)+'/all_results.txt', 'a')
                                                 )
        
        model.fit(train_set, nb_epoch=params['training']["nb_epoch"], callbacks=validation)

        for j in tqdm(range(params['training']["nb_epoch"])):
            print('[epoch '+str(j+1)+'/'+str(params['training']["nb_epoch"])+']')
            loss = model.fit(train_set, nb_epoch=1, callbacks=validation)
            print(f'loss= {loss}')

        model.restore(model_dir=model_dir)

        train_score = model.evaluate(train_set, metrics, transformers=transformers)[params['training']['valid_metric']]
        valid_score = model.evaluate(valid_set, metrics, transformers=transformers)[params['training']['valid_metric']]
        
        print("fold_"+str(i), #"loss:", loss,
              ", score (train):", train_score,
              ", score (valid):", valid_score)
        
        # losses['fold_'+str(i)] = loss
        train_scores['fold_'+str(i)] = train_score
        valid_scores['fold_'+str(i)] = valid_score
    
    # print("losses", losses)
    print("train_scores", train_scores)
    print("valid_scores", valid_scores)
    
    # loss_mean = statistics.mean(losses[k] for k in losses)
    train_score_mean = statistics.mean(train_scores[k] for k in train_scores)
    valid_score_mean = statistics.mean(valid_scores[k] for k in valid_scores)
    
    # print("loss (mean)", loss_mean)
    print("train_score (mean)", train_score_mean)
    print("valid_score (mean)", valid_score_mean)
    
    if params['training']['valid_metric'] in ["r2_score","pearson_r2_score"]:
        return {
            'loss': -1*valid_score_mean,
            'status': STATUS_OK,
            # 'eval_time': time.time(),
            'other_stuff': {
                #'loss_mean': loss_mean,
                'train_score_mean': train_score_mean, 
                'valid_score_mean': valid_score_mean,
                'train_scores': train_scores,
                'valid_scores': valid_scores
            }
        }
        
    elif params['training']['valid_metric'] in ["mean_squared_error","mean_absolute_error","rms_score","mae_score"]:
        return {
            'loss': valid_score_mean,
            'status': STATUS_OK,
            # 'eval_time': time.time(),
            'other_stuff': {
                #'loss_mean': loss_mean,
                'train_score_mean': train_score_mean, 
                'valid_score_mean': valid_score_mean,
                'train_scores': train_scores,
                'valid_scores': valid_scores
            }
        }

def train(params, datasets):
    metrics = [ dc.metrics.Metric(getattr(dc.metrics, metric)) for metric in params['training']['metrics'] ]
    valid_metric = dc.metrics.Metric(getattr(dc.metrics, params['training']['valid_metric'])) 

    if params['splitter']['method'] == 'train_valid_test_split':
        train_set = datasets['train']
        valid_set = datasets['valid']
        test_set = datasets['test']    

    if params['splitter']['method'] == 'train_test_split':
        train_set = datasets['train']
        valid_set = datasets['test']  

    # transformers
    transformers = [ getattr(dc.trans, t)(transform_y=True, dataset=train_set) for t in params["dataloader"]["transformers"]]

    for transformer in transformers:
        train_set = transformer.transform(train_set)
        valid_set = transformer.transform(valid_set)

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
                model_dir = params['training']['model_dir'],
                learning_rate=params['training']['learning_rate'],
                optimizer = params['training']['optimizer']

    )
    
    if params['training']['valid_metric'] in ["r2_score","pearson_r2_score"]:
        save_on_minimum=False
    elif paramst['training']['valid_metric'] in ["mean_squared_error","mean_absolute_error","rms_score","mae_score"]:
        save_on_minimum=True


    validation = dc.models.ValidationCallback(dataset=valid_set, 
                                              interval=params['training']['valid_interval'], metrics=metrics,
                                              save_metric=params['training']['metrics'].index(params['training']['valid_metric']),
                                              save_dir=params['training']['model_dir'], 
                                              # transformers=transformers,
                                              save_on_minimum=save_on_minimum
                                              # output_file=open(params['training']['model_dir']+'/train.log', 'a')
                                             )

    for i in tqdm(range(params['training']["nb_epoch"])):
        print('[epoch '+str(i+1)+'/'+str(params['training']["nb_epoch"])+']')
        loss = model.fit(train_set, nb_epoch=1, callbacks=validation)
        print(f'loss= {loss}')
    
    return model, loss, metrics, valid_metric, transformers



## Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config", type=str, default=None, help="Config file path")
parser.add_argument("-opt", "--optimize", action="store_true", help="Perform Hyperparameter Optimization")
parser.add_argument("-o", "--out_csv", type=str, default='./predict.csv', help="Path to prediction result file name (CSV)")
parser.add_argument("-v", "--verbose", action="store_true", help="For debug")


args = parser.parse_args()


## Logging
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

## Config
logger.info("Loading config...")

with open(args.config) as f:
    config = yaml.safe_load(f)

params = copy.deepcopy(config)

params['dataloader']['featurizer'] = getattr(dc.feat,config['dataloader']['featurizer']['type'])(config['dataloader']['featurizer']['kwargs'])
params['splitter']['type'] = getattr(dc.splits,config['splitter']['type'])()
params['training']['optimizer'] = getattr(dc.models.optimizers, config['training']['optimizer'])()

logger.debug("config:\n %s", config)
logger.info("params:\n %s",params)


## Create Dataset
logger.info("Loading data...")

# Read CSV File
dataset = DataLoader.loadCSV(params['dataloader'])

logger.debug(dataset)

## Hypterparameter optimization (hyperopt)

if args.optimize:
    logger.info('### Hyperparameter optimization with hyperopt ####')
    logger.info('Splitting...')

    datasplitter = DataSplitter

    subsets = getattr(datasplitter, 'k_fold_split')(dataset,  params['splitter'])
    logger.info('%s, k_fold_split, k=%s', params['splitter']['method'], params['splitter']['rate']['k'])
    
    logger.info('num_subsets: %i', len(subsets))

    if config['hyperopt']['algo'] == 'tpe': params['hyperopt']['algo'] = tpe.suggest
    elif config['hyperopt']['algo'] == 'atpe': params['hyperopt']['algo'] = atpe.suggest
    elif config['hyperopt']['algo'] == 'rand': config['hyperopt']['algo'] = rand.suggest

    search_space = {}
    hp_ignore = {}
    for k, v in params['hyperopt']['params'].items():
        if type(v) is list:
            search_space[k] = hp.choice(k,v)       
        elif type(v) is dict:
            search_space[k] = hp.uniform(k,low=float(v['low']), high=float(v['high']))

        else:
            logger.warning("hyperparameter", k, v, ", which is invalid format, will be ignored." )
            hp_ignore[k] = v

    params['hyperopt'], hp_ignore
    
    trials=Trials()
    best = fmin(cv,
        space= search_space,
        algo=params['hyperopt']['algo'],
        max_evals=params['hyperopt']['max_evals'],
        trials = trials)
    
    logger.info("Best: %s", best)
    logger.info(trials.best_trial)

    ## Update input parameters with optimized values
    params_init = copy.deepcopy(params) # backup
    
    for k,v in search_space.items():
        if v.name == 'switch':
            params['training'][k] = params['hyperopt']['params'][k][best[k]]
        elif v.name == 'float':
            params['training'][k] = best[k]

    logger.info('Optimized paramerers:')        
    logger.info(params)
    logger.info('### Hyperparameter optimization finished. ###')


logger.info('### Training ####')
logger.info('Splitting...')
datasplitter = DataSplitter
    
datasets = getattr(datasplitter, params['splitter']['method'])(dataset, params['splitter'])

model, loss, metrics, valid_metric, transformers = train(params, datasets)
logger.info('datasets: %s', datasets)

model.restore(model_dir=params['training']['model_dir'])

if params['splitter']['method'] == 'train_test_split':
    train_score = model.evaluate(datasets['train'], metrics, transformers=transformers)[params['training']['valid_metric']]
    valid_score = model.evaluate(datasets['test'], metrics, transformers=transformers)[params['training']['valid_metric']]
    
    logger.info("loss: %f, score (train): %f, score (valid): %f", loss, train_score, valid_score)

if params['splitter']['method'] == 'train_valid_test_split':
    train_score = model.evaluate(datasets['train'], metrics, transformers=transformers)[params['training']['valid_metric']]
    valid_score = model.evaluate(datasets['valid'], metrics, transformers=transformers)[params['training']['valid_metric']]
    test_score = model.evaluate(datasets['test'], metrics, transformers=transformers)[params['training']['valid_metric']]

    logger.info("loss: %f, score (train): %f, score (valid): %f, score (test): %f", loss, train_score, valid_score, test_score)

if params['splitter']['method'] == 'train_test_split':
    data_for_pred = datasets['test']

if params['splitter']['method'] == 'train_valid_test_split':
    data_for_pred = datasets['valid']
    
params['splitter']['method']


y_pred, y_std = model.predict_uncertainty(data_for_pred)
y_pred = transformers[0].untransform(y_pred)

df = pd.DataFrame(list(zip(data_for_pred.ids, data_for_pred.y, y_pred, y_std)), columns=['smiles', 'actural', 'predict', 'std_dev'])
logger.info('%s', df)

df.to_csv(args.out_csv, index=False)
