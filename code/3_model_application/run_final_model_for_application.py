""" Run final models for application chemicals 

This script allows the user to run the final models created with train_final_models.py for a given set of 
application chemicals (here: standardized marketed chemicals) based on input files created with 
run_feature_calculation_for_application.py. It creates a structured output file including meta-data from 
run_tsne.py which can be used to replicate Figures 3 and 4 from the manuscript by running visualize_application.py.
You can select the POD (effect = 'rd' or 'nc').

Computing time: 2h on an hp EliteBook 840 G8 (11th Gen Intel(R) Core(TM) i7-1185G7 , 3.00 GHz, 4 cores, 32 GB RAM)

"""

import argparse
import pickle
from modules.visuals import * 

# ---- SELECT DATA SET ----
effect = 'nc'   # options: rd, nc

# ---- ARGUMENT PARSER (for command-line execution) ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, default=effect, help='options: rd, nc')
    return parser.parse_args()

# ---- EXECUTION LOGIC ----
if __name__ == "__main__":

    # - load feature data
    # -- load rdkit features for running prediction models
    df = pd.read_csv('../PODUAM/features/data_market_rdkit_std.csv')
    col_feat = df.columns[14:]
    X_raw = df[col_feat]

    # -- load morgan fingerprints for distance evaluation
    morgan_1024 = pd.read_csv('../PODUAM/features/data_market_morgan-1024_std.csv')

    # - load final models
    # -- UAM
    CI=95
    filename = '../PODUAM/final_models/final_model_poduam_CI{CI}_rdkit_{effect}.pkl'.format(effect=effect, CI=CI)
    model = pickle.load(open(filename, 'rb'))

    # -- pre-processing
    filename = '../PODUAM/final_models/final_model_pipe_rdkit_{effect}.pkl'.format(effect=effect)
    pipe = pickle.load(open(filename, 'rb'))
    rdkit_desc = pd.read_csv('../PODUAM/final_models/final_set_desc_rdkit_{effect}.csv'.format(effect=effect)).iloc[:, 0].tolist()

    # -- KNN for distance evaluation
    k_knn = 5
    filename = '../PODUAM/final_models/final_model_knn_{effect}.pkl'.format(effect=effect)
    knn = pickle.load(open(filename, 'rb'))

    # Modeling
    # - pre-processing
    X = pd.DataFrame(pipe.transform(X_raw))
    X.columns = X_raw.columns
    X = X[rdkit_desc]
    
    # - predict marketed chemicals in batches
    batches = np.array_split(df.index, 10)

    y_pred = pd.Series()
    y_lower = pd.Series()
    y_upper = pd.Series()

    for bidx in batches:
        X_batch = X.iloc[bidx]
        model.predict(X_batch)
        y_pred = pd.concat([y_pred, pd.Series(model.test_y_median_base)])
        y_lower = pd.concat([y_lower, pd.Series(model.test_y_lower_uacqrs)])
        y_upper = pd.concat([y_upper, pd.Series(model.test_y_upper_uacqrs)])

    df['yhat'] = y_pred.reset_index(drop=True)
    df['yhat_lo'] = y_lower.reset_index(drop=True)
    df['yhat_up'] = y_upper.reset_index(drop=True)
    df['uhat'] = (y_upper - y_lower).reset_index(drop=True)
    df['yhat_mg'] = np.log10(10 ** (df['yhat']) * df['MW'] * 1e3)
    df['yhat_lo_mg'] = np.log10(10 ** (df['yhat_lo']) * df['MW'] * 1e3)
    df['yhat_up_mg'] = np.log10(10 ** (df['yhat_up']) * df['MW'] * 1e3)

    # - obtain Jaccard distances between application chemicals and their 5 nearest neighbors among training chemicals
    dist, _ = knn.kneighbors(X=morgan_1024[morgan_1024.columns[-1024:]], n_neighbors=k_knn, return_distance=True)
    morgan_1024['dJ'] = dist.mean(axis=1)

    # Create final output file
    # - merge prediction results with Jaccard distances
    df_all = pd.merge(df.drop(columns=col_feat), morgan_1024[['INCHIKEY', 'dJ']], on='INCHIKEY')

    # - merge prediction results with TSNE 
    tsne = pd.read_csv('../PODUAM/data/data_tsne_market.csv')
    df_all = pd.merge(df_all, tsne[['INCHIKEY', 'TSNE1', 'TSNE2']], on='INCHIKEY')

    # - merge prediction results with ClassyFire classes
    classyfire = pd.read_csv('../PODUAM/data/data_classyfire_market.csv')
    class_col = ['Kingdom', 'Superclass', 'Class', 'Subclass']
    df_all = pd.merge(df_all, classyfire[['INCHIKEY'] + class_col], on='INCHIKEY')
    top15 = df_all['Superclass'].value_counts().sort_values(ascending=False).index[:15]
    df_all['Superclass (top 15)'] = df_all['Superclass'].where(df_all['Superclass'].isin(top15), 'Other')

    # - merge prediction results with POD values
    pod = pd.read_csv('../PODUAM/data/data_pod_{effect}.csv'.format(effect=effect))
    pod_std = pd.read_csv('../PODUAM/data/data_pod_{effect}-std.csv'.format(effect=effect))

    pod['POD_mg_all'] = np.log10(pod['POD'])
    pod_std['POD_mg_training'] = np.log10(pod_std['POD'])

    df_all = pd.merge(df_all, pod[['canonical order SMILES', 'POD_mg_all']], on='canonical order SMILES', how='left')

    df_all = pd.merge(df_all, pod_std[['Canonical_QSARr', 'POD_mg_training']], on='Canonical_QSARr', how='left')


    # - save structured output file
    df_all.to_csv('../PODUAM/manuscript/results/application/out_final_model_market_{effect}.csv'.format(effect=effect), index=False)
