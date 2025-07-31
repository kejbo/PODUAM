""" Nested cross-validation of KNN standard ML models

This script allows the user to perform a repeated nested cross-validation of
KNN models with four types of molecular descriptors, which allows to optimize
different hyperparameters and assess the external prediction performance.

Computing time: 3h 15 min on central DTU HPC cluster using 4 nodes
https://www.hpc.dtu.dk/?page_id=2520

"""

import argparse
from sklearn.neighbors import KNeighborsRegressor
from skopt.space import Integer
from modules.training import *
from pathlib import Path

# ---- SELECT DATA SET ----
effect = 'nc'   # options: rd, nc
subset = '-std'

# ---- ARGUMENT PARSER (for command-line execution) ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, default=effect, help='options: rd, nc')
    return parser.parse_args()

# ---- EXECUTION LOGIC ----
if __name__ == "__main__":

    alg = 'KNN'

    # Repeated nested cross-validation
    seed_list = np.arange(10)
    feature_list = ['rdkit', 'maccs', 'cddd', 'morgan-512']

    for seed in seed_list:
        for feat in feature_list:
            # - load training features X and target y 
            df = pd.read_csv('../PODUAM/features/xy_pod_' + effect + subset + '_' + feat + '.csv').set_index('ID')
            ID = df.index

            col_y = 'POD_logmol'
            y = df[col_y]
            mw = df['MW'].reset_index()
            X = df.drop(columns=[col_y, 'MW'])

            # - define model and hyperparameter search space
            model = KNeighborsRegressor()

            param_grid = dict(n_neighbors=Integer(3, 20),
                            weights=['uniform', 'distance'],
                            metric=['minkowski', 'jaccard'])

            # - stratify target y by order of magnitude
            clip_lo = np.round(y.quantile(0.01))
            clip_up = np.round(y.quantile(0.99))
            y_stratify = y.round().clip(lower=clip_lo, upper=clip_up)

            # - apply nested cross-validation
            config, out = nested_cv_sklearn(X, y, ID, model, feat, param_grid, 
                                            k_outer=10, k_inner=5, nrand_cv=30, stratify=y_stratify, seed=seed)
            out.columns = ['ID', 'y', 'yhat', 'fold', 'p']

            out = pd.merge(out, mw, on='ID')
            out['y_mg'] = np.log10(10 ** (out['y']) * out['MW'] * 1e3)
            out['yhat_mg'] = np.log10(10 ** (out['yhat']) * out['MW'] * 1e3)

            # - save optimized hyperparameter settings and test set prediction results from outer folds of nested cross-validation
            out_file = '{alg}_{feat}_out.csv'.format(alg=alg, feat=feat)
            config_file = '{alg}_{feat}_config.csv'.format(alg=alg, feat=feat)
            file_dir = Path('../PODUAM/manuscript/results/consensus/{effect_set}/Seed_{seed:02}/'.format(effect_set=effect+subset, seed=seed))
            file_dir.mkdir(parents=True, exist_ok=True)

            out.to_csv(file_dir / out_file, index=False)
            config.to_csv(file_dir / config_file, index=False)
