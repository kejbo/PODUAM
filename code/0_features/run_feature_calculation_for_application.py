""" Calculate final model inputs for application chemicals 

This script allows the user to calculate the necessary inputs for running the final models:
RDKIT descriptors (for model training), Morgan fingerprints (for similarity assessment) and
molecular weight (for predicted POD unit conversions) for a given set of application chemicals 
(here: marketed chemicals) based on harmonized SMILES strings. It provides the necessary inputs
for run_final_models_for_application.py

"""

from modules.features import *
from modules.structures import *

# Load chemical structure data
df = pd.read_csv('../PODUAM/data/data_marketed_chemicals.csv')
df = df.dropna(subset='canonical order SMILES').reset_index(drop=True)

# Calculate features for standardized chemicals
qsarr = pd.read_csv('../PODUAM/data/OPERA_standardized/smi_marketed_Summary_file.csv')

df_std = df.iloc[qsarr['RowID'] - 1].reset_index(drop=True)
df_std['Canonical_QSARr'] = qsarr['Canonical_QSARr']
df_std['INCHIKEY_QSARr'] = qsarr['InChI Key_QSARr']
df_std['INCHI_QSARr'] = qsarr['InChI_Code_QSARr']
df_std['Salt_Solvent'] = qsarr['Salt_Solvent']
df_std['Salt_Solvent_ID'] = qsarr['Salt_Solvent_ID']

df_std.to_csv('../PODUAM/data/data_marketed_chemicals_std.csv', index=False)

# - Calculate molecular weight
mw_std = calculate_descriptors_rdkit_df(df_std, 'Canonical_QSARr', rdkit_desc=['MolWt'])['MolWt']
mw_std.name = 'MW'

# -- Morgan fingerprints (for similarity assessment)
X_morgan_1024_std = calculate_descriptors_morgan_df(df_std, 'Canonical_QSARr', radius=2, nBits=1024)
df_morgan_1024_std = pd.concat([df_std, mw_std, X_morgan_1024_std], axis=1)
df_morgan_1024_std.to_csv('../PODUAM/features/data_market_morgan-1024_std.csv', index=False)

# -- RdKit (for prediction model)
X_rdkit_std = calculate_descriptors_rdkit_df(df_std, 'Canonical_QSARr')
X_rdkit_std.replace([np.inf, -np.inf], np.nan, inplace=True)
df_rdkit_std = pd.concat([df_std, mw_std, X_rdkit_std], axis=1)
df_rdkit_std.to_csv('../PODUAM/features/data_market_rdkit_std.csv', index=False)

# Calculate features for diverse chemicals
discarded = pd.read_csv('../PODUAM/data/OPERA_standardized/smi_marketed_DiscardedStructures.csv')

df_div = df.iloc[discarded['RowID'] - 1].reset_index(drop=True)
df_div['standardization_failure'] = discarded['Error']

# df_div = df[df['ID'].isin(df_std['ID'])]
df_div.to_csv('../PODUAM/data/data_marketed_chemicals_div.csv', index=False)

# - Calculate molecular weight
mw_div = calculate_descriptors_rdkit_df(df_div, 'canonical order SMILES', rdkit_desc=['MolWt'])['MolWt']
mw_div.name = 'MW'

# -- Morgan fingerprints (for similarity assessment)
X_morgan_1024_div = calculate_descriptors_morgan_df(df_div, 'canonical order SMILES', radius=2, nBits=1024)
df_morgan_1024_div = pd.concat([df_div, mw_div, X_morgan_1024_div], axis=1)
df_morgan_1024_div.to_csv('../PODUAM/features/data_market_morgan-1024_div.csv', index=False)

# -- RdKit (for prediction model)
X_rdkit_div = calculate_descriptors_rdkit_df(df_div, 'canonical order SMILES')
X_rdkit_div.replace([np.inf, -np.inf], np.nan, inplace=True)
df_rdkit_div = pd.concat([df_div, mw_div, X_rdkit_div], axis=1)
df_rdkit_div.to_csv('../PODUAM/features/data_market_rdkit_div.csv', index=False)
