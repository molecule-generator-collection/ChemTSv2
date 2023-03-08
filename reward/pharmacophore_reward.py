import collections
import itertools
import os

import numpy as np
from rdkit import RDConfig, Chem, Geometry
from rdkit.Chem import ChemicalFeatures, rdDistGeom, rdMolTransforms
from rdkit.Chem.Pharm3D import EmbedLib, Pharmacophore
from rdkit.Numerics import rdAlignment

from reward.reward import Reward


def create_pharmacophore():
    fdef = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    cluster_centers_sel = {
        'donors': [
            [-14.494, 18.226, -23.777749999999997],
            [-13.9495, 12.32225, -27.62225]],
        'acceptors': [
            [-11.54175, 12.9275, -29.19325],
            [-16.076, 17.244500000000002, -22.4005]],
        'hydrophobics': [
            [-14.0335, 18.17425, -21.44525]],
        'aromatics': [
            [-12.645999999999999, 14.753833333333333, -28.08808333333333]],
    }
    type_dict = {
        'donors': 'Donor',
        'acceptors': 'Acceptor',
        'hydrophobics': 'Hydrophobe',
        'aromatics': 'Aromatic',
    }
    radi_dict = {
        'Donor': 0.5,
        'Acceptor': 0.5,
        'Hydrophobe': 1.0,
        'Aromatic': 1.0,
    }
    
    ph4Feats = []
    for ptype, coords in cluster_centers_sel.items():
        for coord in coords:
            feat = ChemicalFeatures.FreeChemicalFeature(
                type_dict[ptype],
                Geometry.Point3D(coord[0], coord[1], coord[2])
            )
            ph4Feats.append(feat)

    pcophore = Pharmacophore.Pharmacophore(ph4Feats)
    #print(pcophore)
    bmatrix = pcophore._boundsMat
    for i, j in zip(*np.where(np.triu(bmatrix) > 0)):
        i_type = pcophore.getFeature(i).GetFamily()
        j_type = pcophore.getFeature(j).GetFamily()
        pcophore.setUpperBound(i, j, bmatrix[i, j] + (radi_dict[i_type]+radi_dict[j_type]))
    for i, j in zip(*np.where(np.tril(bmatrix) > 0)):
        i_type = pcophore.getFeature(i).GetFamily()
        j_type = pcophore.getFeature(j).GetFamily()
        pcophore.setLowerBound(i, j, max(bmatrix[i, j] - (radi_dict[i_type]+radi_dict[j_type]), 0))
    #print(pcophore)
    return pcophore, feature_factory

PCOPHORE, FEATURE_FACTORY = create_pharmacophore()

def get_transform_matrix(alignRef, confEmbed, atomMatch):
    """
    Author: Greg Landrum, Date: Nov 4, 2016
    URL: https://github.com/rdkit/UGM_2016/blob/master/Notebooks/Stiefl_RDKitPh4FullPublication.ipynb
    """
    alignProbe = []
    for matchIds in atomMatch:
        dummyPoint = Geometry.Point3D(0.0, 0.0, 0.0)
        for mid in matchIds:
            dummyPoint += confEmbed.GetAtomPosition(mid)
        dummyPoint /= len(matchIds)
        alignProbe.append(dummyPoint)
    ssd, transformMatrix = rdAlignment.GetAlignmentTransform(alignRef, alignProbe)
    return ssd, transformMatrix

def transform_embeddings(pcophore, embeddings, atomMatch):
    """
    Author: Greg Landrum, Date: Nov 4, 2016
    URL: https://github.com/rdkit/UGM_2016/blob/master/Notebooks/Stiefl_RDKitPh4FullPublication.ipynb
    """
    alignRef = [f.GetPos() for f in pcophore.getFeatures()]
    ssds = []
    for embedding in embeddings:
        conformer = embedding.GetConformer()
        ssd, transformMatrix = get_transform_matrix(alignRef, conformer, atomMatch)
        rdMolTransforms.TransformConformer(conformer, transformMatrix)
        ssds.append(ssd)
    return ssds

class Pharmacophore_reward(Reward):
    def get_objective_functions(conf):
        def PharmacophoreSearch(mol):
            can_match, all_matches = EmbedLib.MatchPharmacophoreToMol(mol, FEATURE_FACTORY, PCOPHORE)
            ph4_feat_counter = collections.Counter([f.GetFamily() for f in PCOPHORE.getFeatures()])
            if can_match:
                lig_feat_counter = collections.Counter([f.GetFamily() for f in itertools.chain.from_iterable(all_matches)])
                match_list = [f"{k}:{lig_feat_counter[k]}:{ph4_feat_counter[k]}" for k in ph4_feat_counter.keys()]
            else:
                lig_feat_counter = collections.Counter([f.GetFamily() for f in FEATURE_FACTORY.GetFeaturesForMol(mol)])
                match_list = [f"{k}:{lig_feat_counter[k]}:{ph4_feat_counter[k]}" for k in ph4_feat_counter.keys()]
                return [match_list, None]

            bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
            failed, _, matched, _  = EmbedLib.MatchPharmacophore(all_matches, bm, PCOPHORE, useDownsampling=False)
            if failed:
                return [match_list, None]
            atom_match = [list(x.GetAtomIds()) for x in matched]
            molH = Chem.AddHs(mol)
            Chem.AssignStereochemistry(molH, force=True, cleanIt=True)
            try:
                _, embeddings, _ = EmbedLib.EmbedPharmacophore(molH, atom_match, PCOPHORE, count=20, randomSeed=42, silent=True)
            except ValueError as e:
                return [match_list, None]

            ssds = transform_embeddings(PCOPHORE, embeddings, atom_match)
            if len(ssds) == 0:
                return [match_list, None]
            best_fit_idx = min(enumerate(ssds), key=lambda x:x[1])[0]
            
            embeddings[best_fit_idx].SetDoubleProp('ssd', ssds[best_fit_idx])
            output_conf_dir = os.path.join(conf['output_dir'], 'conformer')
            os.makedirs(output_conf_dir, exist_ok=True)
            writer = Chem.SDWriter(os.path.join(output_conf_dir, f"mol_{conf['gid']}.sdf"))
            writer.write(embeddings[best_fit_idx])
            writer.close()
            
            return [match_list, ssds[best_fit_idx]]
        return [PharmacophoreSearch]

    def calc_reward_from_objective_values(values, conf):
        match_list, ssd = values[0]
        ligf_total = 0
        ph4f_total = 0
        for m in match_list:
            _, ligf_cnt, ph4f_cnt = m.split(":")
            ph4f_total += int(ph4f_cnt)
            ligf_total += min(int(ligf_cnt), int(ph4f_cnt))
        feat_match_ratio = ligf_total / ph4f_total
        ssd_score = 0 if ssd is None else 1 / (1 + ssd)

        return (1*feat_match_ratio + 4*ssd_score) / 5

