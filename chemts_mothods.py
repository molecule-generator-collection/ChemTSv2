import os , sys, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, PandasTools, rdMolDescriptors, Draw
from rdkit.ML.Cluster import Butina
from openbabel import pybel
from IPython.core.debugger import Pdb

import time

plot_cols = ['reward', 'Add_Substituent_MW', 'Add_Substituent_LogP']

def select_weight_model(smiles, estimate_mw):
    peak_values = [n for n in range(0, 800, 50)]

    init_mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
    total_mw = init_mw + estimate_mw
    near_peak = 0
    min_distance = 100
    for peak in peak_values:
        distance = abs(peak - total_mw)
        if distance <= min_distance:
            min_distance = distance
            near_peak = peak
    
    model_dir = str(near_peak-50) + str(near_peak+50)

    # with open('tsukuba_work/model_dir.txt', mode='w') as w:
    #     w.write(str(model_dir))

    model_dir = 'model/weight/' + model_dir
    return model_dir

def make_config_file(configs, weight_model_dir):
    chemts_config = configs['ChemTS']

    # MWごとにモデル切り替え機能(現在は使用していない)
    # chemts_config.setdefault('model_setting', {})
    # chemts_config['model_setting']['model_json'] = os.path.join(weight_model_dir, 'model.tf25.json')
    # chemts_config['model_setting']['model_weight'] = os.path.join(weight_model_dir, 'model.tf25.best.ckpt.h5')
    # chemts_config['token'] = os.path.join(weight_model_dir, 'tokens.pkl')

    # chemts_config['model_setting']['model_json'] = os.path.join('model', 'riken', 'model_chembl220K_r_last.tf25.json')
    # chemts_config['model_setting']['model_weight'] = os.path.join('model', 'riken',  'model_chembl220K_r_last.tf25.best.ckpt.h5')
    # chemts_config['token'] = os.path.join('model', 'riken', 'tokens_chembl220K_r_last.pkl')
    
    # 評価関数の設定
    mw_center = configs['mw']
    logp_center = configs['logp']

    chemts_config['Dscore_parameters']['MW']['top_max'] = mw_center + chemts_config['Dscore_parameters']['MW']['top_range_left']
    chemts_config['Dscore_parameters']['MW']['top_min'] = mw_center - chemts_config['Dscore_parameters']['MW']['top_range_right']
    chemts_config['Dscore_parameters']['MW']['bottom_max'] = mw_center + chemts_config['Dscore_parameters']['MW']['bottom_range_left']
    chemts_config['Dscore_parameters']['MW']['bottom_min'] = mw_center - chemts_config['Dscore_parameters']['MW']['bottom_range_right']
        
    chemts_config['Dscore_parameters']['LogP']['top_max'] = logp_center + chemts_config['Dscore_parameters']['LogP']['top_range_left']
    chemts_config['Dscore_parameters']['LogP']['top_min'] = logp_center - chemts_config['Dscore_parameters']['LogP']['top_range_right']
    chemts_config['Dscore_parameters']['LogP']['bottom_max'] = logp_center + chemts_config['Dscore_parameters']['LogP']['bottom_range_left']
    chemts_config['Dscore_parameters']['LogP']['bottom_min'] = logp_center - chemts_config['Dscore_parameters']['LogP']['bottom_range_right']

    chemts_config['Dscore_parameters'].setdefault('acceptor', {})
    chemts_config['Dscore_parameters']['acceptor']['max'] = configs['acceptor']['max']
    chemts_config['Dscore_parameters']['acceptor']['min'] = configs['acceptor']['min']
    
    chemts_config['Dscore_parameters'].setdefault('donor', {})
    chemts_config['Dscore_parameters']['donor']['max'] = configs['donor']['max']
    chemts_config['Dscore_parameters']['donor']['min'] = configs['donor']['min']

    with open(os.path.join('ChemTSv2', 'work', '_setting.yaml'), 'w') as f:
        yaml.dump(chemts_config, f, default_flow_style=False, sort_keys=False)  

def rearrange_smiles(smi, atom_idx):
    pbmol = pybel.readstring('smi', smi)
    conv = pybel.ob.OBConversion()
    conv.SetOutFormat("smi")
    conv.SetOptions('l"%d"'%(atom_idx), conv.OUTOPTIONS)     # 1始まりなので+1
    rearranged_smiles = conv.WriteString(pbmol.OBMol).split()[0]
    # print(f"({atom_idx) " + rearranged_smiles)   # 出力文字列の最後に"\t\n"が付いていたのでsplitで切り離し
    return rearranged_smiles

def read_mol(pdb_path):
    mol1 = Chem.MolFromPDBFile(pdb_path, sanitize=False)
    # smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    smi = Chem.MolToSmiles(Chem.rdmolops.RemoveHs(mol1), isomericSmiles=True)
    mol2 = Chem.MolFromSmiles(smi)
    smi = Chem.MolToSmiles(Chem.rdmolops.RemoveHs(mol2), isomericSmiles=True)
    for mol in [mol1, mol2]:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx()+1)
    mol1=Chem.rdmolops.RemoveHs(mol1)

    map_num = []
    for atom in mol1.GetAtoms():
        map_num.append(atom.GetAtomMapNum())

    mat = list(mol2.GetSubstructMatch(mol1))
    mat = [m+1 for m in mat]
    match = pd.DataFrame(mat).reset_index(drop=False)
    match['index'] = match['index'] +1
    match['H_num_index'] = map_num
    match.columns = ['no_H_num_index', 'obabel_num', 'H_num_index']
    match = match[['obabel_num', 'no_H_num_index', 'H_num_index']]

    return match, smi

def get_obabel_num(match, extend_idx):
    extend_idx = int(extend_idx)
    return int(match[match['H_num_index']==extend_idx]['obabel_num'].iloc[0]) 

def set_rearrange_smiles(pdb_path, atom_num):
    mol_dir = os.path.dirname(pdb_path)
    match, smi = read_mol(pdb_path)
    extend_idx = int(atom_num.split('.')[1].split('_')[1])
    obabel_num = get_obabel_num(match, extend_idx)
    rearrange_smi = rearrange_smiles(smi, obabel_num)

    return rearrange_smi 

def choise_mol(df, outpath, cutoff=0.3, nsamples=10):
    clusters = mol_clustering_butina(df['mols'], cutoff=cutoff)
    df = df.reset_index(drop=True)
    for cluster_num, idx in clusters.items():
        df.loc[idx, 'clusters'] = cluster_num
    df['clusters'] = df['clusters'].astype(int)
    plot_clustering_tsne(df, outpath, n_components=2, nsamples=nsamples)
    df_choise = choise_mol_from_clustering(df, nsamples)
    
    return df_choise
    
def mol_clustering_butina(mols, cutoff):
    morgan_fp = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048) for x in mols]
    dis_matrix = []
    for i in range(1, len(morgan_fp)):
        similarities = DataStructs.BulkTanimotoSimilarity(morgan_fp[i], morgan_fp[:i], returnDistance = True)
        dis_matrix.extend(similarities)
    clusters = Butina.ClusterData(dis_matrix, len(mols), cutoff, isDistData = True)
    clusters = sorted(clusters, key=len, reverse=True)
    clusters_dict = {index: list(tuple_) for index, tuple_ in enumerate(clusters)}

    return clusters_dict

def plot_clustering_tsne(df, outpath, n_components=2, nsamples=10):
    df_tsne = df.copy()
    df_tsne = df_tsne[df_tsne['clusters']<=nsamples]
    dis_array = calc_distance_array(df_tsne['mols'])
    
    tsne = TSNE(n_components=n_components)
    embedded_points = tsne.fit_transform(dis_array)
    
    # t-SNEの埋め込み結果をdfに追加
    df_tsne['tsne_x'] = embedded_points[:, 0]
    df_tsne['tsne_y'] = embedded_points[:, 1]
    
    # t-SNEのプロット
    plt.figure(figsize=(8, 6))
    
    # クラスタごとに色分け
    clusters = sorted(df_tsne['clusters'].unique())
    for cluster in clusters:
        # クラスタごとにフィルタリング
        cluster_df = df_tsne[df_tsne['clusters'] == cluster]
        plt.scatter(
            cluster_df['tsne_x'],
            cluster_df['tsne_y'],
            label=f"Cluster {cluster}",
            alpha=0.7
        )
    
    # プロットの設定
    plt.title('t-SNE visualization of clustered data')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(outpath, 'clustering.png'), bbox_inches='tight') 

def calc_distance_array(mols):
    morgan_fp = [AllChem.GetMorganFingerprintAsBitVect(x,2,2048) for x in mols]
    dis_matrix = [DataStructs.BulkTanimotoSimilarity(morgan_fp[i], morgan_fp[:len(mols)],returnDistance=True) for i in range(len(mols))]
    dis_array = np.array(dis_matrix)
    return dis_array

def choise_mol_from_clustering(df, nsamples):
    dis_matrix_tri = calc_distance_array(df['mols'])
    
    choise_rows = []
    choise_mols = []

    # 要検討
    # if len(set(df['clusters'])) < nsamples:

    for cluster in range(nsamples):
        indices = df[df['clusters']==cluster].index
        n = len(indices)
        result_array = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                row_index = indices[i]
                col_index = indices[j]
                result_array[i, j] = dis_matrix_tri[row_index, col_index]
        choise_rows.append(indices[np.argmin(np.mean(result_array,axis=0))])
    df_choise = df.iloc[choise_rows]
    return df_choise

def csv_to_mol2(csv, prefix, cutoff, nsamples, ligand_pdb):

    # 立ち上げの基準となるligand
    lig = Chem.MolFromPDBFile(ligand_pdb)
    lig_morgan_fp = AllChem.GetMorganFingerprintAsBitVect(lig, 2, 1024)

    # read csv
    df = pd.read_csv(csv)
    if len(df)==0:
        return

    df = df.drop_duplicates(['smiles'])
    # set unique id
    df['chemts_id'] = range(len(df))

    # add canonical smiles
    df['mols'] = [Chem.MolFromSmiles(smi) for smi in df['smiles']]
    df['canonical_smiles'] = [Chem.MolToSmiles(m) for m in df['mols']]

    # Add mw logp donner acceptor
    df['MW'] = [Descriptors.ExactMolWt(m) for m in df['mols']] 
    df['LogP'] = [Descriptors.MolLogP(m) for m in df['mols']] 
    df['donor'] = [rdMolDescriptors.CalcNumLipinskiHBD(m) for m in df['mols']] 
    df['acceptor'] = [rdMolDescriptors.CalcNumLipinskiHBA(m) for m in df['mols']] 
    
    morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in df['mols']]
    df['Tanimoto_sim'] = DataStructs.BulkTanimotoSimilarity(lig_morgan_fp, morgan_fps)
    # df.drop('mols', axis=1).to_csv(csv)
    
    # 1.0が多かったらTanimotoで遠いものを選ぶ
    df_reward_equal_1 = df[df['reward'] == 1.0]
    if len(df_reward_equal_1) > nsamples:
        df_choise = choise_mol(df_reward_equal_1, os.path.dirname(csv), cutoff, nsamples)
    else:
        df_choise = df.sort_values('reward', ascending=False)[:nsamples]
    df_choise.drop('mols', axis=1).reset_index(drop=False).to_csv(os.path.join(os.path.dirname(csv), 'choise_to_docking.csv'))

    # create ID
    df_choise['ChemTS_idx'] = ['ChemTS_%06d' % i for i in df_choise['chemts_id']]

    # Add hidrogen
    df_choise['mhs'] = [Chem.AddHs(m) for m in df_choise['mols']] 

    # calc 3D structures 
    for mh in df_choise['mhs']:
        try:
            AllChem.ConstrainedEmbed(mh, lig)
        except:
            # AllChem.EmbedMolecule(mh, randomSeed=0)
            continue

    # write structure
    out_cols = list(df_choise.columns)
    out_cols.remove('mhs')
    PandasTools.SaveXlsxFromFrame(df_choise[out_cols], prefix + '.xlsx', molCol='mols', size=(150, 150))

    # write to args.sdf
    sdf_name = prefix + '.sdf' 
    PandasTools.WriteSDF(df_choise, sdf_name, molColName='mhs', 
            properties=['generated_id', 'smiles', 'MW', 'LogP', 'donor', 'acceptor'], idName='ChemTS_idx')

    for i, pbmol in enumerate(pybel.readfile('sdf', sdf_name)):
        mol2_name = '%s_%03d.mol2' % (prefix, (i))
        pbmol.write('mol2', mol2_name, overwrite=True)
    
    if all(col in df.columns for col in plot_cols):
        legends = ['reward=' + str(round(rw, 3)) + '\n,add MW' + str(round(mw, 3)) + ' ,add LogP' + str(round(lp, 2)) + ',' + '\ntanimoto sim=' + str(round(ts, 3))\
                for rw, mw, lp, ts in zip(df_choise['reward'], df_choise['Add_Substituent_MW'], df_choise['Add_Substituent_LogP'], df_choise['Tanimoto_sim'])] 
    else:
        legends = ['reward=' + str(round(rw, 3)) for rw in df_choise['reward']] 
    
    mols = list(df_choise['mols'])
    AllChem.Compute2DCoords(lig)
    for mol in mols:
        AllChem.GenerateDepictionMatching2DStructure(mol, lig)
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(400, 400), legends=legends)
    img.save(os.path.join(os.path.dirname(csv), 'mols.png'))

def plot_reward(result_csv_path):
    window = 50
    fig0, axs0 = plt.subplots(1, 3, figsize=(15, 5))
    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
    df = pd.read_csv(result_csv_path)

    if not all(col in df.columns for col in plot_cols):
        return

    for trial_num, split_df in split_df_on_decrease(df, 'trial'):
        split_df = split_df.reset_index(drop=True)
        for ax, col in zip(axs0, plot_cols):
            ax.set_title(col)
            ax.plot(split_df.index, split_df[col], label=str(trial_num))
        for ax, col in zip(axs1, plot_cols):
            ax.set_title(col)
            smoothing_val = split_df[col].rolling(window=window).mean()
            ax.plot(range(len(smoothing_val)), smoothing_val, label=str(trial_num))

    # Axesにプロットされたデータがあるか確認し、データがある場合は凡例を表示
    if axs0[0].has_data():
        plt.legend()
        fig0.savefig(os.path.join(os.path.dirname(result_csv_path), 'reward.png'))
        fig1.savefig(os.path.join(os.path.dirname(result_csv_path), 'reward_smoothing.png'))

# ひとかたまりのdfを分割する
def split_df_on_decrease(df, column_name='trial'):
    trial_num_set = set(df[column_name])
    for trial_num in trial_num_set:
        df_one = df[df[column_name]==trial_num]
        yield trial_num, df_one

if __name__ == '__main__':
    smiles = sys.argv[1]
    estimate_mw = float(sys.argv[2])
    model_dir = select_weight_model(smiles, estimate_mw)
    print(model_dir)


