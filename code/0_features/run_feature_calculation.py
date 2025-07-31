""" Calculate feature data for training UAM and standard ML models

This script allows the user to calculate the different types of molecular descriptors
for the training chemicals used to train the UAM and standard ML models based on harmonized SMILES strings: 
RDKIT descriptors, MACCS fingerprints, CDDD embeddings and Morgan fingerprints. 
It also calculates the molecular weight for all training chemicals to allow unit conversion of the PODs. 
For every molecular descriptor, it provides three subsets of training data: the expanded data set, 
a standardized subset (STD) and non-standardized subset (EXT).
Note that the CDDD descriptors need to be prepared in a different virtual environment (see readme.txt in cddd subfolder)

"""

import argparse
from modules.features import *
from modules.structures import *

# ---- SELECT DATA SET ----
effect = 'rd'   # options: rd, nc

# ---- ARGUMENT PARSER (for command-line execution) ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, default=effect, help='options: rd, nc')
    return parser.parse_args()

# ---- EXECUTION LOGIC ----
if __name__ == "__main__":

    # - load POD data
    df = pd.read_csv('../PODUAM/Data/data_pod_' + effect + '.csv')

    # Calculate molecular weight to transform y label to molar unit and log-scale
    mw = calculate_descriptors_rdkit_df(df, 'canonical order SMILES', rdkit_desc=['MolWt'])['MolWt']
    mw.name = 'MW'
    y = np.log10(df['POD'].divide(mw * 1000))  # POD data in mg/kg-d
    y.name = 'POD_logmol'

    # Calculate features
    # -- RDKIT
    X_rdkit = calculate_descriptors_rdkit_df(df, 'canonical order SMILES')
    X_rdkit.replace([np.inf, -np.inf], np.nan, inplace=True)

    # -- MORGAN fingerprints
    # 512 bits for model training, 1024 bits for Jaccard distance calculation
    X_morgan_512 = calculate_descriptors_morgan_df(df, 'canonical order SMILES', radius=2, nBits=512)
    X_morgan_1024 = calculate_descriptors_morgan_df(df, 'canonical order SMILES', radius=2, nBits=1024)

    # -- MACCS fingerprints
    X_maccs = calculate_descriptors_maccs_df(df, 'canonical order SMILES')

    # -- CDDD
    X_cddd = pd.read_csv('../PODUAM/Features/cddd/X_cddd_pod_' + effect + '.csv')

    # Create feature data files
    # - merge feature matrix X with target variable y
    Xy_rdkit = pd.concat([df['ID'], mw, y, X_rdkit], axis=1)
    Xy_morgan_512 = pd.concat([df['ID'], mw, y, X_morgan_512], axis=1)
    Xy_morgan_1024 = pd.concat([df['ID'], mw, y, X_morgan_1024], axis=1)
    Xy_maccs = pd.concat([df['ID'], mw, y, X_maccs], axis=1)
    Xy_cddd = pd.concat([df['ID'], mw, y, X_cddd], axis=1)

    # - keep records with min 4 underlying data points
    count_flt = ~(df['count'] < 4)
    Xy_rdkit[count_flt].to_csv('../PODUAM/Features/xy_pod_' + effect + '_rdkit.csv', index=False)
    Xy_morgan_512[count_flt].to_csv('../PODUAM/Features/xy_pod_' + effect + '_morgan-512.csv', index=False)
    Xy_morgan_1024[count_flt].to_csv('../PODUAM/Features/xy_pod_' + effect + '_morgan-1024.csv', index=False)
    Xy_maccs[count_flt].to_csv('../PODUAM/Features/xy_pod_' + effect + '_maccs.csv', index=False)
    Xy_cddd[count_flt].to_csv('../PODUAM/Features/xy_pod_' + effect + '_cddd.csv', index=False)


    # Create features for standardized subset (QSAR ready)
    # - Handle QSAR-ready duplicates
    qsarr = pd.read_csv('../PODUAM/data/OPERA_standardized/smi_' + effect + '_Summary_file.csv')
    df_std_all = df.iloc[qsarr['RowID'] - 1].reset_index(drop=True)
    df_std_all['Canonical_QSARr'] = qsarr['Canonical_QSARr']
    df_std_all['INCHIKEY_QSARr'] = qsarr['InChI Key_QSARr']
    df_std_all['INCHI_QSARr'] = qsarr['InChI_Code_QSARr']
    df_std_all['Salt_Solvent'] = qsarr['Salt_Solvent']
    df_std_all['Salt_Solvent_ID'] = qsarr['Salt_Solvent_ID']
    df_std_all['y'] = np.log10(df['POD'].divide(mw * 1000)).iloc[qsarr['RowID'] - 1].reset_index(drop=True)

    df_std_all = df_std_all[~(df_std_all['count'] < 4)]

    duplicates = df_std_all[df_std_all['Canonical_QSARr'].duplicated(keep=False)]
    aggregate = duplicates.groupby(['Canonical_QSARr', 'INCHIKEY_QSARr', 'INCHI_QSARr']).agg(
        y = ('y', 'mean'), y_std = ('y', 'std'), count_duplicates = ('y', 'count'), 
        count = ('count', 'sum'), casrn=('casrn', ' ,'.join), dtxsid=('dtxsid', ' ,'.join),
        name=('name', ' ,'.join)).reset_index()
    aggregate['ID'] = aggregate.index+20000

    df_std = pd.concat([df_std_all.drop_duplicates(subset='Canonical_QSARr', keep=False), aggregate], axis=0).reset_index(drop=True)
    df_std.to_csv('../PODUAM/data/data_pod_' + effect + '-std.csv', index=False)

    # - Calculate molecular weight of QSAR ready chemicals 
    mw_std = calculate_descriptors_rdkit_df(df_std, 'Canonical_QSARr', rdkit_desc=['MolWt'])['MolWt']
    mw_std.name = 'MW'
    y_std = df_std['y']
    y_std.name = 'POD_logmol'

    # - Calculate features
    # --- RDKIT
    X_rdkit_std = calculate_descriptors_rdkit_df(df_std, 'Canonical_QSARr')
    X_rdkit_std.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- MORGAN fingerprints
    X_morgan_512_std = calculate_descriptors_morgan_df(df_std, 'Canonical_QSARr', radius=2, nBits=512)
    X_morgan_1024_std = calculate_descriptors_morgan_df(df_std, 'Canonical_QSARr', radius=2, nBits=1024)

    # --- MACCS fingerprints
    X_maccs_std = calculate_descriptors_maccs_df(df_std, 'Canonical_QSARr')

    # --- CDDD
    X_cddd_std = pd.read_csv('../PODUAM/features/cddd/X_cddd_pod_' + effect + '-std.csv')

    # - merge feature matrix X with target variable y
    Xy_rdkit_std = pd.concat([df_std['ID'], mw_std, y_std, X_rdkit_std], axis=1)
    Xy_morgan_512_std = pd.concat([df_std['ID'], mw_std, y_std, X_morgan_512_std], axis=1)
    Xy_morgan_1024_std = pd.concat([df_std['ID'], mw_std, y_std, X_morgan_1024_std], axis=1)
    Xy_maccs_std = pd.concat([df_std['ID'], mw_std, y_std, X_maccs_std], axis=1)
    Xy_cddd_std = pd.concat([df_std['ID'], mw_std, y_std, X_cddd_std], axis=1)

    # -- save
    Xy_rdkit_std.to_csv('../PODUAM/features/xy_pod_' + effect + '-std_rdkit.csv', index=False)
    Xy_morgan_512_std.to_csv('../PODUAM/features/xy_pod_' + effect + '-std_morgan-512.csv', index=False)
    Xy_morgan_1024_std.to_csv('../PODUAM/features/xy_pod_' + effect + '-std_morgan-1024.csv', index=False)
    Xy_maccs_std.to_csv('../PODUAM/features/xy_pod_' + effect + '-std_maccs.csv', index=False)
    Xy_cddd_std.to_csv('../PODUAM/features/xy_pod_' + effect + '-std_cddd.csv', index=False)


    # Create features for challenging external test set (from discarded structures)
    discarded = pd.read_csv('../PODUAM/data/OPERA_standardized/smi_' + effect + '_DiscardedStructures.csv')

    df_ext_all = df.iloc[discarded['RowID'] - 1].reset_index(drop=True)
    df_ext_all['Error'] = discarded['Error']

    count_flt_ext = df['ID'].isin(df_ext_all['ID']) & count_flt

    Xy_rdkit[count_flt_ext].to_csv('../PODUAM/features/xy_pod_' + effect + '-ext_rdkit.csv', index=False)
    Xy_morgan_512[count_flt_ext].to_csv('../PODUAM/features/xy_pod_' + effect + '-ext_morgan-512.csv', index=False)
    Xy_morgan_1024[count_flt_ext].to_csv('../PODUAM/features/xy_pod_' + effect + '-ext_morgan-1024.csv', index=False)
    Xy_maccs[count_flt_ext].to_csv('../PODUAM/features/xy_pod_' + effect + '-ext_maccs.csv', index=False)
    Xy_cddd[count_flt_ext].to_csv('../PODUAM/features/xy_pod_' + effect + '-ext_cddd.csv', index=False)

    df_ext_all.to_csv('../PODUAM/data/data_pod_' + effect + '-ext.csv', index=False)
