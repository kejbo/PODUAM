""" Derive confidence-based calibration data for CP models

This script allows the user to derives the fraction of reported PODs 
covered within different confidence intervals ("coverage") used for 
assessing the confidence-based calibration by training CP models 
at different confidence levels. You can select the POD (effect = 'rd' or 'nc'), 
descriptors (feat = 'rdkit', 'maccs', 'cddd' or 'morgan-512') and whether 
the standardized or expanded dataset should be used (subset = '-std' or '').

"""

import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

from modules.uacqr import uacqr
from modules.training import *

# ---- SELECT DATA SET ----
effect = 'nc'   # options: rd, nc
subset = '-std'     # options: '', -std
feat = 'rdkit'  # options: rdkit, maccs, cddd, morgan-512

# ---- ARGUMENT PARSER (for command-line execution) ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, default=effect, help='options: rd, nc')
    parser.add_argument("--subset", type=str, default=subset, help='options: '', -std')
    parser.add_argument("--feat", type=str, default=feat, help='options: rdkit, maccs, cddd, morgan-512')
    return parser.parse_args()

# ---- EXECUTION LOGIC ----
if __name__ == "__main__":
    # -- load training features X and target y 
    df = pd.read_csv('../PODUAM/features/xy_pod_' + effect + subset + '_' + feat + '.csv').set_index('ID')
    ID = df.index

    col_y = 'POD_logmol'
    y = df[col_y]
    mw = df['MW']
    X = df.drop(columns=[col_y, 'MW'])

    # Settings
    alg = 'RF'
    model_params = dict(max_depth=15, max_features=0.5, min_impurity_decrease=0.001)

    CI = 95
    q_lower = (100-CI)/2
    q_upper = 100-(100-CI)/2

    seed = 42

    # Model training with simple train-test split (20%)
    X_train_cal, X_test, y_train_cal, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    train_cal_idx = X_train_cal.index
    test_idx = X_test.index

    # - Preprocessing
    # -- pre-process data based on training data
    if 'rdkit' in feat:
        X_train_cal, X_test, pipe = rdkit_pipe(X_train_cal, X_test)
        X_train_cal.index, X_test.index = train_cal_idx, test_idx

    # - drop correlated features based on training data
    if any(x in feat for x in ['rdkit', 'maccs']):
        X_train_cal = feature_selection(X_train_cal)
        X_test = X_test[X_train_cal.columns]


    # - Modeling
    # -- split training data for model calibration (20% for calibrating confidence intervals)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.25, random_state=seed)
    train_idx = X_train.index
    cal_idx = X_cal.index

    p = X_train.shape[1]

    # -- Train models at defined confidence levels (expected "coverage")
    coverage = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

    pred_coverage = list()
    for num, c in enumerate(coverage):
        cov_model = uacqr(model_type='rfqr', B=1000, random_state=0, uacqrs_agg='iqr',
                        q_lower=((1-c)/2)*100, q_upper=(1-(1-c)/2)*100,
                        model_params=model_params)

        cov_model.fit(X_train, y_train)
        cov_model.calibrate(X_cal, y_cal)

        # -- Assess achieved "coverage" on test set
        cov_model.evaluate(X_test, y_test)
        pred_coverage.append(cov_model.uacqrs_test_coverage)

    # Save results with expected and observed "coverage" at different confidence levels
    out_cov = pd.concat([pd.Series(coverage, name='coverage'), pd.Series(pred_coverage, name='pred_coverage')], axis=1)
    out_cov.to_csv('../PODUAM/manuscript/results/crossvalidation/out_coverage_cp-{alg}_{feat}_{effect}.csv'.format(
        alg=alg, effect=effect+subset, feat=feat))
