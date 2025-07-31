""" Module containing functions to curate and harmonize chemical structures

"""

import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def get_df_compounds_cas(df, col_cas, col_name):
    """ Fetches IUPAC name, canonical SMILES, InChI key and InChI strings from PubChem
    for a list of chemicals provided in dataframe df with their CAS numbers and chemical names

    Inputs
    ----------
    df : pandas dataframe, mandatory
        Dataframe containing a list of chemicals with columns for CAS numbers and chemical names
    col_cas: str, mandatory
        column name containing CAS numbers
    col_name: str, mandatory
        column name containing chemical names

    Outputs
    ----------
    df_out: pandas dataframe
        dataframe with CAS, chemical name, foundby (CAS or name), PubChem ID (CID), IUPAC name, 
        isomeric and caonical SMILES, InChI key and InChI strings
    """

    df_out = pd.DataFrame()

    for (cas, name) in zip(df.loc[:, col_cas], df.loc[:, col_name]):
        # Look-up chemical by CAS, else by name
        try:
            results = pcp.get_compounds(cas, 'name')
            foundby = 'CAS'
        except:
            results = pcp.get_compounds(name, 'name')
            foundby = 'name'

        for compound in results:
            df_out = pd.concat([df_out,
                                pd.DataFrame([[cas, name, foundby,
                                               compound.cid, compound.iupac_name,
                                               compound.isomeric_smiles, compound.canonical_smiles,
                                               compound.inchikey, compound.inchi]])
                                ])

    df_out.columns = [col_cas, col_name, 'Found by', 'CID', 'IUPAC',
                      'isomeric SMILES', 'canonical SMILES',
                      'InChIKey', 'InChI']

    return df_out


def mol_with_atom_index(mol):
    """ Support function numbering each atom in a mol object

    Inputs
    ----------
    mol : object, mandatory
        RDKit mol object created from SMILES or InChI

    Outputs
    ----------
    mol: object
        RDKit mol object with numbered atoms

    """

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return mol


def mol_without_atom_index(mol):
    """ Support function removing atom numbering in a mol object

    Inputs
    ----------
    mol : object, mandatory
        RDKit mol object with numbered atoms

    Outputs
    ----------
    mol: object
        RDKit mol object without numbering

    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def myMolFromSmiles(smiles):
    """ Function to create mol object from SMILES performing partial sanitization when necessary

    Inputs
    ----------
    smiles : str, mandatory
        SMILES string

    Outputs
    ----------
    mol: object
        RDKit mol object

    """
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # try partial sanitization
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol,
                             Chem.SanitizeFlags.SANITIZE_FINDRADICALS | Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                             Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                             Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                             catchErrors=True)
            print('Partial sanitization: ' + smiles)
        except:
            print('Partial sanitization failed - return none: ' + smiles)

    return mol


def create_ordered_smiles(smiles, remove_numbers=False):
    """ Function creates canonicalized ordered SMILES with optional atom numbering
    by creating mol files from ordered SMILES strings, applying tautomerization and 
    (optionally) adding atom numbering before converting back to SMILES

    Inputs
    ----------
    smiles : str, mandatory
        SMILES string
    remove_numbers: bool, default: False
        if True returns SMILES without explicit atom numbering

    Outputs
    ----------
    canonical_order_smiles : str
        canonicalized ordered SMILES string

    """

    # reorder SMILES
    if myMolFromSmiles(smiles) is None:
        new_mol = None
    else:
        mod_smi = Chem.MolToSmiles(myMolFromSmiles(smiles))
        new_mol = myMolFromSmiles(mod_smi)

    if new_mol is None:  # return empty string if still no mol
        print('No mol: ' + smiles)
        return ''
    else:
        # Tautomerize
        try:
            enumerator = rdMolStandardize.TautomerEnumerator()
            new_mol = enumerator.Canonicalize(new_mol)
        except:
            print('No tautomerization:' + smiles)

        # Add numbering
        new_mol_numbered = mol_with_atom_index(new_mol)

        if remove_numbers:
            new_mol_numbered = mol_without_atom_index(new_mol_numbered)

        canonical_order_smiles = Chem.MolToSmiles(new_mol_numbered)
        return canonical_order_smiles


def create_ordered_smiles_df(df, col_smiles, remove_numbers=False):
    """ Function to create canonicalized ordered SMILES for all SMILES in a given column of a dataframe

    Inputs
    ----------
    df : pandas dataframe, mandatory
        Dataframe containing a column with SMILES
    col_smiles: str, mandatory
        column name containing SMILES
    remove_numbers: bool, default: False
        if True returns SMILES without explicit atom numbering

    Outputs
    ----------
    Pandas series containing canonicalized ordered SMILES

    """

    return df[col_smiles].apply(create_ordered_smiles, remove_numbers=remove_numbers)


def remove_chiral_centers(smiles):
    """ Support function to remove chiral information from SMILES

    Inputs
    ----------
    smiles : str, mandatory
        SMILES string

    Outputs
    ----------
    res_smiles: str
        SMILES string without chiral information

    """
    m = myMolFromSmiles(smiles)
    if m is not None:
        all_chiral = Chem.FindMolChiralCenters(m)
        all_chiral_centers = [sublist[0] for sublist in all_chiral]
        if len(all_chiral_centers) > 0:
            for each in all_chiral_centers:
                m.GetAtomWithIdx(each).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        res_smiles = Chem.MolToSmiles(m)
    else:
        res_smiles = smiles

    return res_smiles


def remove_cis_trans(smiles):
    """ Support function to remove cis/trans information from SMILES

    Inputs
    ----------
    smiles : str, mandatory
        SMILES string

    Outputs
    ----------
    res_smiles: str
        SMILES string without cis/trans information

    """
    m = myMolFromSmiles(smiles)
    if m is not None:
        for b in m.GetBonds():
            if b.GetStereo() in {Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ,
                                 Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS,
                                 Chem.rdchem.BondStereo.STEREOANY}:
                b.SetStereo(Chem.rdchem.BondStereo.STEREONONE)

        res_smiles = Chem.MolToSmiles(m)
    else:
        res_smiles = smiles

    return res_smiles


def remove_canonical_duplicates(df, col_ids, col_canonical='canonical SMILES', col_isomeric='isomeric SMILES'):
    """ Function to remove canonical duplicates from a dataframe while giving priority to non-stereometric data entries,
    otherwise uses first duplicate. Returns dataframe without duplicates but retains all CIDs from original dataframe.
    
    Inputs
    ----------
    df : pandas dataframe, mandatory
        Dataframe containing columns with canonical and isomeric SMILES
    col_ids: list, mandatory
        list of column names that should be used to identify unique data entries
    col_canonical: str, default: 'canonical SMILES'
        column name containing canonical SMILES
    col_canonical: str, default: 'isomeric SMILES'
        column name containing isomeric SMILES

    Outputs
    ----------
    Pandas dataframe without canonical duplicates with list of duplicate CIDs in column all_CIDs

    """

    df['sort'] = df[col_canonical] == df[col_isomeric]
    df_sorted = df.sort_values(by=col_ids + ['sort'], ascending=False)
    df_dpl = df_sorted.drop_duplicates(col_ids, keep='first').drop(columns='sort')

    # Keep all identified CIDs
    cids = df_sorted.groupby(col_ids).aggregate(
        all_CIDs=('CID', lambda tdf: tdf.unique().tolist())).reset_index()

    return pd.merge(left=df_dpl, right=cids, on=col_ids)

