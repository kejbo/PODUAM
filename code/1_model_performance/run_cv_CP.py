""" Run crossvalidation of CP models

This script allows the user to run the crossvalidation for the CP models
and merge the results with relevant meta data. You can select the POD 
(effect = 'rd' or 'nc'), descriptors (feat = 'rdkit', 'maccs', 'cddd' or 'morgan-512') 
and whether the standardized or expanded dataset should be used (subset = '-std' or '').

"""

import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from modules.uacqr import uacqr
from modules.training import *
import pickle

# ---- SELECT DATA SET ----
effect = 'nc'   # options: rd, nc
subset = '-std'     # options: '', -std
feat = 'cddd'  # options: rdkit, maccs, cddd, morgan-512

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
    df = pd.read_csv('../PODUAM/features/xy_pod_{effect}_{feat}.csv'.format(effect=effect+subset, feat=feat)).set_index('ID')
    ID = df.index

    col_y = 'POD_logmol'
    y = df[col_y]
    mw = df['MW']
    X = df.drop(columns=[col_y, 'MW'])

    # -- load Morgan fingerprints for distance evaluation
    data_path = '../PODUAM/features/xy_pod_' + effect+subset + '_morgan-1024.csv'
    data_1024 = pd.read_csv(data_path)
    data_1024 = data_1024[data_1024['ID'].isin(ID)].set_index('ID')
    col_fps = data_1024.columns[-1024:]

    # Settings
    alg = 'RF'
    model_params = dict(max_depth=15, max_features=0.5, min_impurity_decrease=0.001)

    CI = 95
    q_lower = (100-CI)/2
    q_upper = 100-(100-CI)/2

    k_cv = 10
    k_knn = 5
    seed = 42

    # Crossvalidation
    clip_lo = np.round(y.quantile(0.01))
    clip_up = np.round(y.quantile(0.99))
    folds = kfold(X, k=k_cv, seed=seed, stratify=y.round().clip(lower=clip_lo, upper=clip_up))
    cv_results = pd.DataFrame()
    n_fold = 0

    for train_cal_ix, test_ix in folds:
        X_train_cal, X_test = X.iloc[train_cal_ix], X.iloc[test_ix]
        y_train_cal, y_test = y.iloc[train_cal_ix], y.iloc[test_ix]

        train_cal_idx = X_train_cal.index
        test_idx = X_test.index

        # - Preprocessing
        # -- transform features based on training data
        if 'rdkit' in feat:
            X_train_cal, X_test, pipe = rdkit_pipe(X_train_cal, X_test)
            X_train_cal.index, X_test.index = train_cal_idx, test_idx

        # -- drop correlated features based on training data
        if any(x in feat for x in ['rdkit', 'maccs']):
            X_train_cal = feature_selection(X_train_cal)
            X_test = X_test[X_train_cal.columns]
        
        # - Modeling
        # -- split training data for model calibration (20% for calibrating confidence intervals)
        X_train, X_cal, y_train, y_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.25, random_state=seed,
                                                        stratify=y_train_cal.round().clip(lower=clip_lo, upper=clip_up))
        train_idx = X_train.index
        cal_idx = X_cal.index

        p = X_train.shape[1]

        # -- obtain Jaccard distances between test chemicals and their 5 nearest neighbors among train&val chemicals
        knn = NearestNeighbors(n_neighbors=k_knn, metric='jaccard').fit(data_1024.loc[train_cal_idx, col_fps], None)
        dist, _ = knn.kneighbors(X=data_1024.loc[test_idx, col_fps], n_neighbors=k_knn, return_distance=True)
        dJ_test = pd.Series(dist.mean(axis=1)).set_axis(test_idx)

        
        # -- define uncertainty model and train
        model = uacqr(model_type='rfqr', B=1000, random_state=0, uacqrs_agg='iqr', q_lower=q_lower, q_upper=q_upper,
                    model_params=model_params)
        model.fit(X_train, y_train)
        model.calibrate(X_cal, y_cal)

        # - apply trained model to predict test set
        model.evaluate(X_test, y_test)

        y_pred_k = pd.Series(model.test_y_median_base).set_axis(y_test.index)

        upper_k = pd.Series(model.test_y_upper_uacqrs).set_axis(y_test.index)
        lower_k = pd.Series(model.test_y_lower_uacqrs).set_axis(y_test.index)
        widths_k = upper_k - lower_k
        coverage_k = model.uacqrs_test_coverage

        # - collect test set prediction results
        cv_results = pd.concat([cv_results,
                                pd.concat([pd.Series(y_test.index),
                                        pd.Series(y_test.values),
                                        pd.Series(y_pred_k.values),
                                        pd.Series(widths_k.values),
                                        pd.Series(lower_k.values),
                                        pd.Series(upper_k.values),
                                        pd.Series(dJ_test.values),
                                        pd.Series(mw[test_idx].values),
                                        pd.Series(n_fold, index=range(len(y_test))),
                                        pd.Series(p, index=range(len(y_test)))], axis=1)], axis=0)

        n_fold += 1

    out = cv_results.reset_index(drop=True)
    out.columns = ['ID', 'y', 'yhat', 'uhat', 'yhat_lo', 'yhat_up', 'dJ', 'MW', 'fold', 'p']
    out['y_mg'] = np.log10(10 ** (out['y']) * out['MW'] * 1e3)
    out['yhat_mg'] = np.log10(10 ** (out['yhat']) * out['MW'] * 1e3)
    out['yhat_lo_mg'] = np.log10(10 ** (out['yhat_lo']) * out['MW'] * 1e3)
    out['yhat_up_mg'] = np.log10(10 ** (out['yhat_up']) * out['MW'] * 1e3)
    out['res'] = (out['y']-out['yhat'])
    out['uhat_std'] = out[['yhat_lo', 'yhat', 'yhat_up']].apply(lambda row: fit_skewnorm_std(row, CI), axis=1)

    # Retrain model on all data and obtain internal model predictions
    # - Preprocessing
    #  -- train rdkit pipe
    if 'rdkit' in feat:
        X, _, pipe = rdkit_pipe(X, X)
        X.index = ID

        # - optional: save all models
        filename = '../PODUAM/all_models/model_pipe_rdkit_{effect}.pkl'.format(effect=effect)
        pickle.dump(pipe, open(filename, 'wb'))


    # -- drop correlated features based on training data
    if any(x in feat for x in ['rdkit', 'maccs']):
        X = feature_selection(X)

        # - optional: save all models
        X = feature_selection(X)
        pd.Series(X.columns).to_csv('../PODUAM/all_models/set_desc_{feat}_{effect}.csv'.format(feat=feat, effect=effect), index=False)


    # - Modeling
    # -- define uncertainty model and train on all data
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.25, random_state=seed,
                                                        stratify=y.round().clip(lower=clip_lo, upper=clip_up))

    model_all = uacqr(model_type='rfqr', B=1000, random_state=0, uacqrs_agg='iqr', q_lower=q_lower, q_upper=q_upper,
                    model_params=model_params)

    model_all.fit(X_train, y_train)
    model_all.calibrate(X_cal, y_cal)

    # -- apply trained model to predict training data (internal predictions)
    model_all.predict(X)
    y_pred = pd.Series(model_all.test_y_median_base)
    y_upper = pd.Series(model_all.test_y_upper_uacqrs)
    y_lower = pd.Series(model_all.test_y_lower_uacqrs)

    out_in = pd.DataFrame()
    out_in['ID'] = ID
    out_in['yhat_in'] = y_pred.reset_index(drop=True)
    out_in['uhat_in'] = (y_upper - y_lower).reset_index(drop=True)
    out_in['yhat_in_mg'] = np.log10(10 ** (out_in['yhat_in']) * mw.values * 1e3)

    # -- obtain Jaccard distances between training chemicals and their closest 5 neighbours among other training chemicals (themselves excluded)
    knn = NearestNeighbors(n_neighbors=k_knn + 1, metric='jaccard').fit(data_1024[data_1024.columns[-1024:]], None)
    dist_data, _ = knn.kneighbors(X=data_1024[data_1024.columns[-1024:]], n_neighbors=k_knn, return_distance=True)
    data_1024['dJ_in'] = dist_data.mean(axis=1)
    out_in = pd.merge(out_in, data_1024.reset_index()[['ID', 'dJ_in']], on='ID')

    # -- save model
    filename = '../PODUAM/all_models/model_poduam_CI{CI}_{feat}_{effect}.pkl'.format(feat=feat, effect=effect, CI=CI)
    pickle.dump(model_all, open(filename, 'wb'))

    # -- train KNN for Jaccard distance to training chemicals & save
    knn = NearestNeighbors(n_neighbors=k_knn, metric='jaccard').fit(data_1024[data_1024.columns[-1024:]], None)

    filename = '../PODUAM/all_models/model_knn_{feat}_{effect}.pkl'.format(feat=feat, effect=effect)
    pickle.dump(knn, open(filename, 'wb'))

    # Create final output file
    # - merge internal & crossvalidation prediction results
    out = pd.merge(out, out_in, on='ID', copy=False)

    # - merge meta data
    data_pod = pd.read_csv('../PODUAM/data/data_pod_{effect}.csv'.format(effect=effect+subset))
    data_class = pd.read_csv('../PODUAM/data/data_classyfire_{effect}.csv'.format(effect=effect))

    meta = pd.merge(data_pod[['ID', 'casrn', 'name', 'canonical order SMILES']],
                    data_class[['ID', 'Kingdom', 'Superclass', 'Class', 'Subclass']], on='ID', how='left', copy=False)

    out = pd.merge(out, meta, on='ID', copy=False)

    out.to_csv('../PODUAM/manuscript/results/crossvalidation/out_cv_cp-{alg}_{CI}_{feat}_{effect}.csv'.format(alg=alg, effect=effect+subset, feat=feat, CI=CI), index=False)
