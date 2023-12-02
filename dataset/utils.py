import json
from rdkit.Chem import AllChem
import numpy as np

def my_collate_fn(batch):
    all_smiles  = [_[0] for _ in batch]
    all_y =  [_[1] for _ in batch]
    # all_y = np.concatenate(all_y, axis=0 )
    return all_smiles, all_y

def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    # input_df = pd.read_csv(input_path, sep=',')
    # smiles_list = input_df['smiles']
    # rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    #    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    # labels = input_df[tasks]
    # # convert 0 to -1
    # labels = labels.replace(0, -1)
    # # convert nan to 0
    # labels = labels.fillna(0)
    # assert len(smiles_list) == len(rdkit_mol_objs_list)
    # assert len(smiles_list) == len(labels)
    with open(input_path) as json_file:
        binary_list = json.load(json_file)

    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 

    return smiles_list, rdkit_mol_objs_list, labels


def _load_sider_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """

    with open(input_path) as json_file:
        binary_list = json.load(json_file)

    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 
    # print(smiles_list)
    # print(labels)
    # raise TypeError

    # input_df = pd.read_csv(input_path, sep=',')
    # smiles_list = input_df['smiles']
    # rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # tasks = ['Hepatobiliary disorders',
    #    'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
    #    'Investigations', 'Musculoskeletal and connective tissue disorders',
    #    'Gastrointestinal disorders', 'Social circumstances',
    #    'Immune system disorders', 'Reproductive system and breast disorders',
    #    'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
    #    'General disorders and administration site conditions',
    #    'Endocrine disorders', 'Surgical and medical procedures',
    #    'Vascular disorders', 'Blood and lymphatic system disorders',
    #    'Skin and subcutaneous tissue disorders',
    #    'Congenital, familial and genetic disorders',
    #    'Infections and infestations',
    #    'Respiratory, thoracic and mediastinal disorders',
    #    'Psychiatric disorders', 'Renal and urinary disorders',
    #    'Pregnancy, puerperium and perinatal conditions',
    #    'Ear and labyrinth disorders', 'Cardiac disorders',
    #    'Nervous system disorders',
    #    'Injury, poisoning and procedural complications']
    # labels = input_df[tasks]
    # # convert 0 to -1
    # labels = labels.replace(0, -1)
    # assert len(smiles_list) == len(rdkit_mol_objs_list)
    # assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels

if __name__ == "__main__":
    input_path = "/home/richard/projects/fsadmet/data/tox21/new/12/raw/tox21.json"
    _load_tox21_dataset(input_path)


