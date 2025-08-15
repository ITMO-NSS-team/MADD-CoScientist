import os
import pandas as pd
from rdkit.Avalon import pyAvalonTools
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import numpy as np
import warnings
from rdkit import RDLogger
import joblib
RDLogger.DisableLog('rdApp.*')
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.filterwarnings('ignore')

#a function that predicts the standard value for the transmitted smiles list
def predict(smiles_list):
    model = joblib.load("utils/ki_models/tyrosine_regression_inference/model_tyrosine_regr.pkl")
    predictions = model.predict(create_features_for_smiles(smiles_list))
    return predictions

#a function that brings the transmitted smiles to the canonical form
def safe_canon_smiles(smiles):
    try:
        return Chem.CanonSmiles(smiles)
    except Exception as e:
        print(f"Bad Smiles: {smiles}")
        return None

#a function that generates descriptors to describe the smiles molecule
def get_all_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    mol_descriptors = []
    for mol in mols:
        mol = Chem.AddHs(mol)
        descriptors = calc.CalcDescriptors(mol)
        mol_descriptors.append(descriptors)
    return mol_descriptors, desc_names

#a function that generates fingerprints to describe the structure of a molecule
def generate_AVfpts(data):
    Avalon_fpts = []
    mols = [Chem.MolFromSmiles(x) for x in data if x is not None]
    for mol in mols:
        avfpts = pyAvalonTools.GetAvalonFP(mol, nBits=512)
        Avalon_fpts.append(avfpts)
    return pd.DataFrame(np.array(Avalon_fpts))

#a function that creates a dataframe with all the features for the transmitted smiles
def create_features_for_smiles(smiles_names):
    df = pd.DataFrame(columns=["Smiles"])
    for i in smiles_names:
        df = df._append({"Smiles": i}, ignore_index=True)
    df['Canonical Smiles'] = df.Smiles.apply(safe_canon_smiles)
    df.drop(['Smiles'], axis=1, inplace=True)
    df.drop_duplicates(inplace=True, keep='first', subset='Canonical Smiles')
    mol_descriptors, descriptors_names = get_all_descriptors(df['Canonical Smiles'].tolist())
    descriptors_df = pd.DataFrame(mol_descriptors, columns=descriptors_names)
    AVfpts = generate_AVfpts(df['Canonical Smiles'])
    AVfpts.columns = AVfpts.columns.astype(str)
    df.drop(["Canonical Smiles"], axis=1, inplace=True)
    X_test = pd.concat([descriptors_df, AVfpts], axis=1)
    scaler= joblib.load("utils/ki_models/tyrosine_regression_inference/scaler_tyrosine_regr.pkl")
    X_new = scaler.transform(get_only_scale_features(df = X_test))
    return X_new

#a function that selects only columns with a pre-applied scaler for the dataframe
def get_only_scale_features(df):
    path = 'utils/ki_models/tyrosine_regression_inference/selected_features_tyrosine_regr.csv'#os.path.join("..", "regression_csvs", "selected_features_tyrosine_regr.csv")
    with open(path, "r") as inf:
        arr = [i for i in inf.readline().split(",")]
    return df[arr]

if __name__=='__main__':
    predictions = predict(["Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nc(-c2cccnc2)cs1", "Cc1cc(C)c(/C=C2\C(=O)Nc3ccc(-c4cccnc4)cc32)[nH]1", "CNC(=O)c1nn(C)c2c1CCc1cnc(NC3CCN(C(=O)C4CCN(S(C)(=O)=O)CC4)CC3)nc1-2"])
    print(predictions)