"""  Train final models on all available data for deployment

This script retrains the models on all available data and saves them for deployment.

"""

import pickle
import argparse
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from modules.uacqr import uacqr
from modules.training import *


# ---- SELECT DATA SET ----
effect = 'nc'   # options: rd, nc
subset = '-std' # options: '', -std

# ---- ARGUMENT PARSER (for command-line execution) ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, default=effect, help='options: rd, nc')
    parser.add_argument("--subset", type=str, default='-std', help='options: '', -std')
    return parser.parse_args()

# ---- EXECUTION LOGIC ----
if __name__ == "__main__":
    args = parse_args()
    effect = args.effect  # override defaults
    subset = args.subset

    name = 'final' if subset=='-std' else 'diverse'

    # Settings
    model_params = dict(max_depth=15, max_features=0.5, min_impurity_decrease=0.001)

    CI = 95
    q_lower = (100-CI)/2
    q_upper = 100-(100-CI)/2

    k_knn = 5
    seed = 42

    # Load data, get label y and feature matrix X
    df = pd.read_csv('../PODUAM/features/xy_pod_{effect}_rdkit.csv'.format(effect=effect+subset)).set_index('ID')
    ID = df.index

    col_y = 'POD_logmol'
    y = df[col_y]
    mw = df['MW']
    X = df.drop(columns=[col_y, 'MW'])

    # load data for distance evaluation
    data_1024 = pd.read_csv('../PODUAM/features/xy_pod_{effect}_morgan-1024.csv'.format(effect=effect+subset))
    data_1024 = data_1024[data_1024['ID'].isin(ID)].set_index('ID')
    col_fps = data_1024.columns[-1024:]

    # Train models for deployment
    # - Preprocessing
    #  -- train rdkit pipe
    X, _, pipe = rdkit_pipe(X, X)
    X.index = ID

    filename = '../PODUAM/final_models/{name}_model_pipe_rdkit_{effect}.pkl'.format(name=name, effect=effect)
    pickle.dump(pipe, open(filename, 'wb'))

    # -- drop correlated features based on training data
    X = feature_selection(X)
    pd.Series(X.columns).to_csv('../PODUAM/final_models/{name}_set_desc_rdkit_{effect}.csv'.format(name=name, effect=effect), index=False)

    # - Model training
    # -- train on all data & save
    clip_lo = np.round(y.quantile(0.01))
    clip_up = np.round(y.quantile(0.99))

    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.25, random_state=seed,
                                                        stratify=y.round().clip(lower=clip_lo, upper=clip_up))

    model_all = uacqr(model_type='rfqr', B=1000, random_state=0, uacqrs_agg='iqr', q_lower=q_lower, q_upper=q_upper,
                    model_params=model_params)

    model_all.fit(X_train, y_train)
    model_all.calibrate(X_cal, y_cal)

    filename = '../PODUAM/final_models/{name}_model_poduam_CI{CI}_rdkit_{effect}.pkl'.format(name=name, effect=effect, CI=CI)
    pickle.dump(model_all, open(filename, 'wb'))

    # -- train KNN for Jaccard distance to training chemicals & save
    knn = NearestNeighbors(n_neighbors=k_knn, metric='jaccard').fit(data_1024[data_1024.columns[-1024:]], None)

    filename = '../PODUAM/final_models/{name}_model_knn_{effect}.pkl'.format(name=name, effect=effect)
    pickle.dump(knn, open(filename, 'wb'))


