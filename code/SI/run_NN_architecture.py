""" Nested cross-validation of NN conventional ML models

This script is an adapted version of run_NN.py to crossvalidate NN performance for different architectures
by setting the number and size of hidden layers. The number of hidden layers can be specified as integer (layers=int).
The size of hidden layers can be specified as a  multiple of input size p  either  as single value (all hidden layers 
have equal size) or as a list of strings (length = number of hidden layers) (nodes=str or list of str, e.g. 'p//2', 'p', '2*p')

"""

import argparse
from pathlib import Path
from skopt.space import Real, Integer
from modules.training import *

# ---- SELECT DATA SET ----
effect = 'nc'   # options: rd, nc
feat = 'rdkit'  # options: 'rdkit', 'maccs', 'cddd', 'morgan-512'
layers = 1      # number of hidden layers (int or str)
nodes = 'p'  # size of hidden layers as multiples of input size p (e.g. p//2, p, 2*p), single (=equal size) or list

subset = '-std'
seed = 0

# ---- ARGUMENT PARSER (for command-line execution) ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, default=effect, help='options: rd, nc')
    parser.add_argument("--feat", type=str, default=feat, help='options: rdkit, maccs, cddd, morgan-512')
    parser.add_argument("--layers", type=int, default=layers, help='number of hidden layers')
    parser.add_argument("--nodes", type=str, default=nodes, help='size of hidden layers as multiples of input size p (e.g. p//2, p, 2*p), single (=equal size) or list')

    return parser.parse_args()

# ---- EXECUTION LOGIC ----
if __name__ == "__main__":

    alg = 'NN'

    # Nested cross-validation
    arch = 'l' + str(layers) + '-' + str(nodes)

    # - load training features X and target y 
    df = pd.read_csv('../PODUAM/features/xy_pod_{effect}_{feat}.csv'.format(effect=effect+subset, feat=feat)).set_index('ID')
    ID = df.index

    col_y = 'POD_logmol'
    y = df[col_y]
    mw = df['MW'].reset_index()
    X = df.drop(columns=[col_y, 'MW'])

    p=X.shape[1]

    # - evaluate strings / force to for layers and nodes 
    layers_int = int(layers)
    if isinstance(nodes, list):
        nodes_int = [eval(n) for n in nodes]
    else:
        nodes_int = int(eval(nodes))

    # - define hyperparameter search space
    param_grid = dict(batch_size=[32, 64, 128],
                    dropout_rate=Real(0, 0.5),
                    hidden_layers=[layers_int],
                    nodes_per_layer=[nodes_int],
                    optimizer__learning_rate=Real(1e-4, 0.01, prior='log-uniform'),
                    optimizer__weight_decay=Real(1e-5, 0.1, prior='log-uniform'))

    # - stratify target y by order of magnitude
    clip_lo = np.round(y.quantile(0.01))
    clip_up = np.round(y.quantile(0.99))
    y_stratify = y.round().clip(lower=clip_lo, upper=clip_up)

    # - apply nested cross-validation
    config, out = nested_cv_mlp(X, y, ID, feat, param_grid,
                                k_outer=10, k_inner=5, nrand_cv=40, stratify=y_stratify, seed=seed)
    out.columns = ['ID', 'y', 'yhat', 'fold', 'p']

    out = pd.merge(out, mw, on='ID')
    out['y_mg'] = np.log10(10 ** (out['y']) * out['MW'] * 1e3)
    out['yhat_mg'] = np.log10(10 ** (out['yhat']) * out['MW'] * 1e3)

    # - save optimized hyperparameter settings and test set prediction results from outer folds of nested cross-validation
    out_file = '{alg}_{feat}_out_{arch}.csv'.format(alg=alg, feat=feat, arch=arch)
    config_file = '{alg}_{feat}_config_{arch}.csv'.format(alg=alg, feat=feat, arch=arch)
    file_dir = Path('../PODUAM/manuscript/results/consensus/architecture/{effect}/'.format(effect=effect+subset))
    file_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(file_dir / out_file, index=False)
    config.to_csv(file_dir / config_file, index=False)
