""" Module containing wrapper functions to calculate molecular descriptors from SMILES using rdkit.Chem

"""

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from modules.structures import *


def calculate_descriptors_rdkit(smiles, rdkit_desc):
    """ Wrapper function that calculates all RDKit molecular descriptors listed in rdkit_desc for a single SMILES

    Inputs
    ----------
    smiles : str, mandatory
        The SMILES string
    rdkit_desc: list, mandatory
        The list of RDKit descriptor names

    Outputs
    ----------
    array of calculated RDKit descriptors
    """

    mol = myMolFromSmiles(smiles)
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(rdkit_desc)

    return calculator.CalcDescriptors(mol)

def calculate_descriptors_rdkit_df(df, col_smiles, rdkit_desc='all'):
    """ Wrapper function that calculates all RDKit molecular descriptors listed in rdkit_desc for a series of SMILES

    Inputs
    ----------
    df : pandas dataframe, mandatory
        The dataframe containing the series of SMILES
    col_smiles: string, mandatory
        The column name containing the SMILES
    rdkit_desc: 'all' or list, default: 'all'
        The specified list of RDKit descriptor names. If 'all', all available RDKit descriptors are calculated

    Outputs
    ----------
    dataframe of calculated RDKit descriptors
    """

    if rdkit_desc == 'all':
        # list all molecular descriptors in RDKit
        rdkit_desc = [x[0] for x in Descriptors._descList]

    d = df[col_smiles].apply(calculate_descriptors_rdkit, rdkit_desc=rdkit_desc)

    return pd.DataFrame.from_records(d, columns=rdkit_desc)


def calculate_descriptors_morgan(smiles, **kwargs):
    """ Wrapper function that calculates Morgan fingerprints for a single SMILES

    Inputs
    ----------
    smiles : str, mandatory
        The SMILES string
    **kwargs: optional
        Pass in any arguments taken by rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect such as radius and nBits

    Outputs
    ----------
    array of calculated Morgan fingerprints
    """

    mol = myMolFromSmiles(smiles)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, **kwargs))


def calculate_descriptors_morgan_df(df, col_smiles, **kwargs):
    """ Wrapper function that calculates Morgan fingerprints for a series of SMILES

    Inputs
    ----------
    df : pandas dataframe, mandatory
        The dataframe containing the series of SMILES
    col_smiles: string, mandatory
        The column name containing the SMILES
    **kwargs: optional
        Pass in any arguments taken by rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect such as radius and nBits

    Outputs
    ----------
    dataframe of calculated Morgan fingerprints
    """

    d = df[col_smiles].apply(calculate_descriptors_morgan, **kwargs)
    return pd.DataFrame.from_records(d)


def calculate_descriptors_maccs(smiles, **kwargs):
    """ Wrapper function that calculates MACCS keys for a single SMILES

    Inputs
    ----------
    smiles : str, mandatory
        The SMILES string
     **kwargs: optional
        Pass in any arguments taken by rdkit.Chem.MACCSkeys.GenMACCSKeys

    Outputs
    ----------
    array of calculated MACCS keys
    """

    mol = myMolFromSmiles(smiles)
    return np.array(MACCSkeys.GenMACCSKeys(mol, **kwargs))


def calculate_descriptors_maccs_df(df, col_smiles, **kwargs):
    """ Wrapper function that calculates MACCS keys for a series of SMILES

    Inputs
    ----------
    df : pandas dataframe, mandatory
        The dataframe containing the series of SMILES
    col_smiles: string, mandatory
        The column name containing the SMILES
    **kwargs: optional
        Pass in any arguments taken by rdkit.Chem.MACCSkeys.GenMACCSKeys

    Outputs
    ----------
    dataframe of calculated MACCS keys
    """

    d = df[col_smiles].apply(calculate_descriptors_maccs, **kwargs)
    return pd.DataFrame.from_records(d)
