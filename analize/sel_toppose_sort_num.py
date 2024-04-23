import os
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, SDWriter, PandasTools

def main():
    # 1. sdfを読み込み、molファイルのリストを作成する
    dir_path = "/home/toita/Labnote/v161_analogue_by_ChemTSv2_gnina/result/gnina_strain_test/run3/3D_pose/"
    files = os.listdir(dir_path)
    toppose_mol_list = []
    for i in range(0, 7000):
        filename = f"mol_{i}_out.sdf"
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path):
            suppl = SDMolSupplier(path)
            toppose_mol = suppl[0]
            toppose_mol_list.append(toppose_mol)
        else:
           print("None")
    # 3. 残したmolファイルを新しいsdfに書き出す
    output_sdf_file = '/home/toita/Labnote/v161_analogue_by_ChemTSv2_gnina/result/strain_test.sdf'  # 出力SDFファイルのパス
    writer = SDWriter(output_sdf_file)
    for mol in toppose_mol_list:
        writer.write(mol)
    writer.close()

if __name__ == "__main__":
    main()
