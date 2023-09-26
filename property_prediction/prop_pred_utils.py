import deepchem as dc

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
        # datasets = dict(train=train_dataset, test=test_dataset)
        
        return folds