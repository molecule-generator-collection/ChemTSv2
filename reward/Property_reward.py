from rdkit import Chem
from reward.reward import Reward

import deepchem as dc
import numpy as np
import pandas as pd
import copy

## functions
class DataLoader:
    def loadCSV(config):
        infile = config["infile"]
        target = config["target"]
        tasks = config["tasks"]
        featurizer = config['featurizer']
    
        loader = dc.data.CSVLoader(tasks, feature_field=target, featurizer=featurizer)
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


class Property_reward(Reward):
    def get_objective_functions(conf):
        def PropValue(mol):
            
            if mol is None:
                return None
            
            params = copy.deepcopy(conf['deepchem'])
            
            MODEL_DIR = params["model_dir"]
            print(f"[INFO] loaded model from {MODEL_DIR}")

            model = dc.models.GraphConvModel(
                        n_tasks = params['n_tasks'],
                        graph_conv_layers = params['graph_conv_layers'],
                        dense_layer_size = params['dense_layer_size'],
                        dropout = params['dropout'],
                        mode = params['mode'],
                        number_atom_features = params['number_atom_features'],
                        n_classes = params['n_classes'],
                        batch_size = params['batch_size'],
                        batch_normalize = params['batch_normalize'],
                        uncertainty = params['uncertainty'],
                        model_dir = params['model_dir']
            )

            model.restore()
            
            params['featurizer'] = getattr(dc.feat,params['featurizer']['type'])(params['featurizer']['kwargs'])

            print("params:\n %s",params)

            smiles = [Chem.MolToSmiles(mol)]
            featurizer = params['featurizer']
            features = featurizer.featurize(smiles)
            d = dc.data.NumpyDataset(X=features,ids=smiles)
            
            print('preprocessed_data:', d)
            
            y_pred, y_std = model.predict_uncertainty(d)
            
            df = pd.DataFrame(list(zip(d.ids, y_pred)), columns=['smiles', 'value'])
            print('prediction_result:', df)

            return y_pred[0]
        return [PropValue]

    def calc_reward_from_objective_values(values, conf):
        #return np.tanh(values[0]/10) if None not in values else -1
        return values[0] if None not in values else -999

    
    
