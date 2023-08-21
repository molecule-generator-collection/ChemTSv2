import os,sys
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import argparse
import pickle
import torch
from tqdm.auto import tqdm

from models.epsnet import *
from utils.datasets import *
from utils.transforms import *
from utils.misc import *

from confgf import dataset

def Embed3D_Geodiff(mol=None,ckpt_path=None,tag=None,
            device='cuda',
            clip=1000.0,
            n_steps=5000,
            global_start_sigma=0.5,
            w_global=1.0,
            sampling_type='ld',
            eta=1.0,
            smi=None,
            infile=None,
            edge_order=3,
            save_data=False,
            log_dir='./result/geodiff/',
            seed=12345):


    #def num_confs(num:str):
    #    if num.endswith('x'):
    #       return lambda x:x*int(num[:-1])
    #    elif int(num) > 0: 
    #        return lambda x:int(num)
    #    else:
    #        raise ValueError()

    # Logging
    output_dir = get_new_log_dir(log_dir, 'sample', tag=tag)
    logger = get_logger('test', output_dir)
    #logger.info(args)

            
            
    #num_confs = num_confs('2x')
    #num_samples = num_confs(2)
    #num_samples = args.num_confs
    num_samples = 1

    # Load checkpoint
    ckpt = torch.load(ckpt_path)
    seed_all(seed)
    

    # Datasets and loaders
    logger.info('Loading datasets...')

    transforms = Compose([
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=edge_order), # Offline edge augmentation
    ])
    
    # Model
    logger.info('Loading model...')
    model = get_model(ckpt['config'].model).to(device)
    model.load_state_dict(ckpt['model'])
   
    logger.info('Loading testset...')
    test_set = []

    if infile and smi:
        logger.error('Error. Mol or SMILES is required.')
        sys.exit()

    elif infile:
        with open(infile, 'r') as f:
            for s in f:
                s = f.readline().rstrip()
                test_set.append(s)

    elif smi:
        test_set.append(smi)
    
    elif mol != None and smi == None and infile == None:
        try:
            smi = Chem.MolToSmiles(mol)
            test_set.append(smi)
        except Exception as e:
            logger.error(e)
            sys.exit()

    print(test_set)

    test_set = map(dataset.dataset.smiles_to_data, test_set)

    done_smiles = set()
    results = []
    
    # Predict
    logger.info('Begin prediction ...')
    
    print(test_set)
    for i, data in enumerate(tqdm(test_set)):
        print(data)

        logger.info('%i, %s', i, data.smiles)

        data_input = data.clone()
        data_input['pos_ref'] = None
        batch = repeat_data(data_input, num_samples).to(device)

        clip_local = None
        for _ in range(2):  # Maximum number of retry
            try:
                pos_init = torch.randn(batch.num_nodes, 3).to(device)
                pos_gen, pos_gen_traj = model.langevin_dynamics_sample(
                    atom_type=batch.atom_type,
                    pos_init=pos_init,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    extend_order=False, # Done in transforms.
                    n_steps=n_steps,
                    step_lr=1e-6,
                    w_global=w_global,
                    global_start_sigma=global_start_sigma,
                    clip=clip,
                    clip_local=clip_local,
                    sampling_type=sampling_type,
                    eta=eta
                )
                pos_gen = pos_gen.cpu()

                data.pos_gen = pos_gen
                results.append(data)
                done_smiles.add(data.smiles)

                if save_data:
                    save_path = os.path.join(output_dir, 'samples_%d.pkl' % i)
                    logger.info('Saving samples to: %s' % save_path)
                    with open(save_path, 'wb') as f:
                        pickle.dump(results, f)


                mol = Chem.AddHs(data.rdmol)
                
                try:
                    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                    conf = mol.GetConformer()
                    logger.info("A temporal conformer is embedded by ETKDGv3.")
                    
                    
                except:
                    logger.info("Try ETDG...")
                    try:
                        AllChem.EmbedMolecule(mol, AllChem.ETDG())
                        conf = mol.GetConformer()
                        logger.info("A temporal conformer is embedded by ETDG.")           
                    except ValueError as e:
                        logger.error('%i, %s', i, e)
                        continue
                
                for a in range(mol.GetNumAtoms()):
                    x,y,z = data.pos_gen.tolist()[a]
                    conf.SetAtomPosition(i,Point3D(x,y,z))
                    
                AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
                
                mol = Chem.RemoveHs(mol)

                if save_data:
                    save_path_sdf = os.path.join(output_dir, 'samples_%d.sdf' % i)
                    logger.info('Saving samples to: %s' % save_path)
                    w = Chem.SDWriter(save_path_sdf)
                    w.write(mol)
    
                break   # No errors occured, break the retry loop
            except FloatingPointError:
                clip_local = 20
                logger.warning('Retrying with local clipping.')

    #save_path = os.path.join(output_dir, 'samples_all.pkl')
    #logger.info('Saving samples to: %s' % save_path)

#    def get_mol_key(data):
#        for i, d in enumerate(test_set):
#            if d.smiles == data.smiles:
#                return i
#        return -1
#    results.sort(key=get_mol_key)
#
#    print(results)
#    
#    if save_data:
#        with open(save_path, 'wb') as f:
#            pickle.dump(results, f)

    return mol



