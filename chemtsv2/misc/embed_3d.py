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
            log_dir='./result/geodiff',
            seed=12345,
            gid='',
            debug=True):

    # Logging
    output_dir = get_new_log_dir(log_dir, 'geodiff_'+str(gid), tag=tag)
    num_samples = 1

    # Load checkpoint
    ckpt = torch.load(ckpt_path)
    seed_all(seed)

    # Datasets and loaders
    print('Loading datasets...')

    transforms = Compose([
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=edge_order), # Offline edge augmentation
    ])
    
    # Model
    print('Loading model...')
    
    model = get_model(ckpt['config'].model).to(device)
    model.load_state_dict(ckpt['model'])
   
    print('Loading testset...')
    test_set = []

    if infile and smi:
        print('Error. Mol or SMILES is required.')
        sys.exit()

    elif infile:
        with open(infile, 'r') as f:
            for s in f:
                s = s.rstrip()
                test_set.append(s)

    elif smi:
        test_set.append(smi)
    
    elif mol != None and smi == None and infile == None:
        try:
            smi = Chem.MolToSmiles(mol)
            test_set.append(smi)
        except Exception as e:
            print(e)
            return

    if debug:
        print(test_set)

    test_set = map(dataset.dataset.smiles_to_data, test_set)

    done_smiles = set()
    results = []
    
    # Predict
    print('Begin prediction ...')
    
    if debug:
        print(test_set)

    for i, data in enumerate(tqdm(test_set)):

        if debug:
            print(data)

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

                print("results=", results)
                results.append(data)
                done_smiles.add(data.smiles)

                if debug:
                    print("## GeoDiff predict positions ")
                    print(data.pos_gen)

                if save_data:
                    save_path = os.path.join(output_dir, 'samples_%s.pkl' % str(i))
                    print(f'Saving the conformer to: {save_path}')
                    with open(save_path, 'wb') as f:
                        pickle.dump(results, f)


                #mol = Chem.AddHs(data.rdmol)
                mol = data.rdmol

                try:
                    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                    conf = mol.GetConformer()
                    if debug:
                        print("A temporal conformer is embedded by ETKDGv3.")
                    
                    
                except:
                    if debug:
                        print("Try ETDG...")
                    try:
                        AllChem.EmbedMolecule(mol, AllChem.ETDG())
                        conf = mol.GetConformer()
                        if debug:
                            print("A temporal conformer is embedded by ETDG.")           
                    except ValueError as e:
                        print('Error.', i, e)
                        continue

                conf = mol.GetConformer(-1)

                for a in range(mol.GetNumAtoms()):
                    x,y,z = data.pos_gen.tolist()[a]
                    conf.SetAtomPosition(i,Point3D(x,y,z))
                    
                if debug:
                    
                    print("### 3D Molecule (RDKit EmbedMolecule)")
                    print(Chem.MolToMolBlock(mol))
    
                    print("### 3D Molecule (GeoDiff)")
                    print(Chem.MolToMolBlock(mol))
    
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
                    
                    print("### 3D Molecule (MMFF optimized)")
                    print(Chem.MolToMolBlock(mol))
    
                    mol = Chem.RemoveHs(mol)
    
                if save_data:
                    save_path_sdf = os.path.join(output_dir, 'samples_%s.sdf' % str(i))
                    print(f'Saving the conformer to: {save_path}')
                    w = Chem.SDWriter(save_path_sdf)
                    w.write(mol)
    
                break   # No errors occured, break the retry loop
            except FloatingPointError:
                clip_local = 20
                print('[Warning] Retrying with local clipping.')

    return mol

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='logs/drugs_default/checkpoints/drugs_default.pt', help='path for loading the checkpoint')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=5000,
                    help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--global_start_sigma', type=float, default=0.5,
                    help='enable global gradients only when noise is low')
    parser.add_argument('--w_global', type=float, default=1.0,
                    help='weight for global gradients')
    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='ld',
                    help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                    help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')

    parser.add_argument('--smi', type=str, default=None)
    parser.add_argument('--infile', type=str, default=None,
                    help = 'input smi file for multiple monomers')
    parser.add_argument('--edge_order', type=int, default=3)
    parser.add_argument('--log_dir', type=str, default='./result/geodiff/')
    parser.add_argument('--save_data', action='store_true', default=False)
    
    args = parser.parse_args()
    
    n_steps = args.n_steps
    w_global = args.w_global
    global_start_sigma = args.global_start_sigma
    clip = args.clip
    sampling_type = args.sampling_type
    eta = args.eta
    tag = args.tag
    edge_order = args.edge_order
    device = args.device

    infile = args.infile
    smi = args.smi
    ckpt_path = args.ckpt
    log_dir = args.log_dir

    mol = Embed3D_Geodiff(smi = args.smi,
                    n_steps = args.n_steps,
                    w_global = args.w_global,
                    global_start_sigma = args.global_start_sigma,
                    clip = args.clip,
                    sampling_type = args.sampling_type,
                    eta = args.eta,
                    tag = args.tag,
                    edge_order = args.edge_order,
                    device = args.device,
                    infile = args.infile,
                    ckpt_path = args.ckpt,
                    seed = 12345,
                    save_data = args.save_data,
                    log_dir = args.log_dir)

    print(Chem.MolToMolBlock(mol))
    print(Chem.MolToSmiles(mol))

