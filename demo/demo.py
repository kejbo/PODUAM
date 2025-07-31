""" Run final or diverse models for demo set of chemicals

This script allows the user to run the final or diverse models created with train_final_models.py for a given set of demo chemicals.
Demo chemicals are provided via an input file (input_path = path) - the only mandatory column is the SMILES column.
SMILES should be provided as standardized SMILES (using Mansouri et al. protocol) to use the final models, otherwise the diverse models should be used
(standardized = True or False). Results are saved to the demo_output.csv containing all information provided in demo_input.csv file alongside predictions
for reproductive/development (rd) and general non-cancer (nc) PODs with their 95% confidence intervals and the structural similarity with training chemicals
described by mean Jaccard distance. 

Computing time: 2 min on an hp Zbook Firefly G11 (Intel(R) Core(TM) Ultra 7 165H 1.40 GHz, 16 cores, 64 GB RAM) for a demo set of 10 chemicals

"""
import time
start_time = time.time()

import argparse
from modules.features import * 
from modules.training import * 

# ---- SELECT DATA SET ----
standardized = False  # options: True, False
input_file = '../PODUAM/demo/demo_input.csv'
output_file = '../PODUAM/demo/demo_output.csv'

# ---- ARGUMENT PARSER (for command-line execution) ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized", type=str, default=standardized, help='standardized or diverse SMILES - options: True, False')
    parser.add_argument("--input_file", type=str, default=input_file, help='path and name of input .csv file with column SMILES (+ optional other columns)')
    parser.add_argument("--output_file", type=str, default=output_file, help='path and name under which output .csv file should be saved')
    return parser.parse_args()

# ---- EXECUTION LOGIC ----
if __name__ == "__main__":

    # - load input data
    print("Loading data and models...preparing...")
    data_in = pd.read_csv(input_file)

    if standardized == True:
        col_smiles='SMILES'
        models_rd = load_models(endpoint='rd', standardized=True)
        models_nc = load_models(endpoint='nc', standardized=True)
    else:
        col_smiles='ordered SMILES'
        data_in['ordered SMILES'] = create_ordered_smiles_df(data_in, 'SMILES')
        models_rd = load_models(endpoint='rd', standardized=False)
        models_nc = load_models(endpoint='nc', standardized=False)


    # Calculate descriptors
    print("Calculating descriptors...")
    rdkit = calculate_descriptors_rdkit_df(data_in, col_smiles)
    mw = calculate_descriptors_rdkit_df(data_in, col_smiles, rdkit_desc=['MolWt'])
    morgan_1024 = calculate_descriptors_morgan_df(data_in, col_smiles, radius=2, nBits=1024)

    # Make predictions
    print("Making predictions...this may take a while...")
    prediction_rd = make_prediction(models_rd, rdkit, mw)
    prediction_nc = make_prediction(models_nc, rdkit, mw)

    prediction_df_rd = pd.DataFrame(prediction_rd.values, columns = ['PODrd [mg/kg-d]', 'PODrd (2.5%ile)', 'PODrd (97.5%ile)'])
    prediction_df_nc = pd.DataFrame(prediction_nc.values, columns = ['PODnc [mg/kg-d]', 'PODnc (2.5%ile)', 'PODnc (97.5%ile)'])

    # Structural similarity
    print("Assess structural similarity...")
    dist_rd, _ = models_rd['knn'].kneighbors(X=morgan_1024[morgan_1024.columns[-1024:]], n_neighbors=5, return_distance=True)
    dist_nc, _ = models_nc['knn'].kneighbors(X=morgan_1024[morgan_1024.columns[-1024:]], n_neighbors=5, return_distance=True)
    dJ_rd = pd.Series(dist_rd.mean(axis=1), name='Jaccard distance (rd)')
    dJ_nc = pd.Series(dist_rd.mean(axis=1), name='Jaccard distance (nc)')

    # Save results to output file
    data_out = pd.concat([data_in, prediction_df_rd, dJ_rd, prediction_df_nc, dJ_nc], axis=1)
    data_out.to_csv(output_file, index=False, float_format="%.4f")

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
