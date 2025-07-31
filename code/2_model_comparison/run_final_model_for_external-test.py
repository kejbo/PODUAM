""" Run final models for external test chemicals 

This script allows the user to run the final models created with train_final_models.py for a set of
test chemicals (here: non-standardized chemicals) based on input files created with 
run_feature_calculation.py. You can select the POD (effect = 'rd' or 'nc').

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
    df = pd.read_csv('../PODUAM/features/xy_pod_{effect}-ext_rdkit.csv'.format(effect=effect)).set_index('ID')
    ID = df.index

    col_y = 'POD_logmol'
    y = df[col_y]
    mw = df['MW'].reset_index(drop=True)

    X_raw = df.drop(columns=[col_y, 'MW'])

    # -- load morgan fingerprints for distance evaluation
    morgan_1024 = pd.read_csv('../PODUAM/features/xy_pod_' + effect + '-ext_morgan-1024.csv')
    morgan_1024 = morgan_1024[morgan_1024['ID'].isin(ID)].set_index('ID')
    col_fps = morgan_1024.columns[-1024:]

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
    
    # - predict marketed chemicals
    batches = np.array_split(df.index, 10)

    y_pred = pd.Series()
    y_lower = pd.Series()
    y_upper = pd.Series()

    model.predict(X)
    y_pred = pd.Series(model.test_y_median_base)
    y_upper = pd.Series(model.test_y_upper_uacqrs)
    y_lower = pd.Series(model.test_y_lower_uacqrs)

    out = pd.DataFrame()
    out['ID'] = ID
    out['y'] = pd.Series(y.values)
    out['yhat'] = y_pred.reset_index(drop=True)
    out['yhat_lo'] = y_lower.reset_index(drop=True)
    out['yhat_up'] = y_upper.reset_index(drop=True)
    out['uhat'] = (y_upper - y_lower).reset_index(drop=True)
    out['y_mg'] = np.log10(10 ** (out['y']) * mw * 1e3)
    out['yhat_mg'] = np.log10(10 ** (out['yhat']) * mw * 1e3)
    out['yhat_lo_mg'] = np.log10(10 ** (out['yhat_lo']) * mw * 1e3)
    out['yhat_up_mg'] = np.log10(10 ** (out['yhat_up']) * mw * 1e3)
    out['res'] = (out['y']-out['yhat'])
    out['uhat_std'] = out[['yhat_lo', 'yhat', 'yhat_up']].apply(lambda row: fit_skewnorm_std(row, CI), axis=1)

    # - obtain Jaccard distances between application chemicals and their 5 nearest neighbors among training chemicals
    dist, _ = knn.kneighbors(X=morgan_1024[morgan_1024.columns[-1024:]], n_neighbors=k_knn, return_distance=True)
    morgan_1024['dJ'] = dist.mean(axis=1)

    out = pd.merge(out, morgan_1024.reset_index()[['ID', 'dJ']], on='ID')

    # - merge meta data
    data_pod = pd.read_csv('../PODUAM/data/data_pod_{effect}-ext.csv'.format(effect=effect))
    data_class = pd.read_csv('../PODUAM/data/data_classyfire_{effect}.csv'.format(effect=effect))

    meta = pd.merge(data_pod[['ID', 'casrn', 'name', 'canonical order SMILES', 'Error']],
                    data_class[['ID', 'Kingdom', 'Superclass', 'Class', 'Subclass']], on='ID', how='left', copy=False)

    out = pd.merge(out, meta, on='ID', copy=False)

    # - save structured output file
    out.to_csv('../PODUAM/manuscript/results/external/out_final_model_ext_{effect}.csv'.format(effect=effect), index=False)
