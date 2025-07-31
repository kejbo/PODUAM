""" Run crossvalidation of BNN models and derive confidence-based calibration data

This script allows the user to run the crossvalidation for the BNN models
and merge the results with relevant meta data. It also derives the fraction 
of reported PODs covered within different credible intervals ("coverage") 
used for assessing the confidence-based calibration. You can select the POD 
(effect = 'rd' or 'nc'), descriptors (feat = 'rdkit', 'maccs', 'cddd' or 'morgan-512') 
and whether the standardized or expanded dataset should be used (subset = '-std' or '').

"""
import argparse
import tqdm
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from modules.nn_model import *
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
    batch_size = 32
    epochs = 1000
    validation_split = 0.2
    learning_rate = 0.002

    callbacks = [tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=1e-5,  # this is absolute
                patience=20,
                restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=10,
                min_delta=1e-5)]

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
    y_preds = pd.DataFrame()
    n_fold = 0

    for train_val_ix, test_ix in folds:
        X_train_val, X_test = X.iloc[train_val_ix], X.iloc[test_ix]
        y_train_val, y_test = y.iloc[train_val_ix], y.iloc[test_ix]

        train_val_idx = X_train_val.index
        test_idx = X_test.index

        # - Preprocessing
        # -- transform features based on training data
        if 'rdkit' in feat:
            X_train_val, X_test, pipe = rdkit_pipe(X_train_val, X_test)
            X_train_val.index, X_test.index = train_val_idx, test_idx

        # -- drop correlated features based on training data
        if any(x in feat for x in ['rdkit', 'maccs']):
            X_train_val = feature_selection(X_train_val)
            X_test = X_test[X_train_val.columns]
        
        # - Modeling
        # -- split training data for model calibration (20% validation data used for callbacks)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=seed,
                                                        stratify=y_train_val.round().clip(lower=clip_lo, upper=clip_up))
        train_idx = X_train.index
        val_idx = X_val.index

        p = X_train.shape[1]

        # -- obtain Jaccard distances between test chemicals and their 5 nearest neighbors among train&val chemicals
        knn = NearestNeighbors(n_neighbors=k_knn, metric='jaccard').fit(data_1024.loc[train_val_idx, col_fps], None)
        dist, _ = knn.kneighbors(X=data_1024.loc[test_idx, col_fps], n_neighbors=k_knn, return_distance=True)
        dJ_test = pd.Series(dist.mean(axis=1)).set_axis(test_idx)

        # -- define uncertainty model and train
        kl_loss_weight = 1 / X_train.shape[0]

        model = MLP_TFP(p, kl_loss_weight, feat)
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                    loss=negloglik, metrics=['mse'])
        history = model.fit(X_train, y_train, epochs=epochs, verbose=True, validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            batch_size=batch_size)
        model.summary()

        # - apply trained model to predict test set
        y_pred_list = []
        for i in tqdm.tqdm(range(500)):
            y_pred = model.predict(X_test)
            y_pred_list.append(y_pred)

        y_preds_k = np.concatenate(y_pred_list, axis=1)
        y_mean_k = pd.Series(np.mean(y_preds_k, axis=1)).set_axis(y_test.index)
        y_sigma_k = pd.Series(np.std(y_preds_k, axis=1)).set_axis(y_test.index)
        y_low_k = pd.Series(np.percentile(y_preds_k, q_lower, axis=1)).set_axis(y_test.index)
        y_up_k = pd.Series(np.percentile(y_preds_k, q_upper, axis=1)).set_axis(y_test.index)
        uhat_k = y_up_k-y_low_k

        # - collect test set prediction results
        cv_results = pd.concat([cv_results,
                                pd.concat([pd.Series(ID[test_ix]),
                                        pd.Series(y_test.values),
                                        pd.Series(y_mean_k.values),
                                        pd.Series(y_sigma_k.values),
                                        pd.Series(uhat_k.values),
                                        pd.Series(y_low_k.values),
                                        pd.Series(y_up_k.values),
                                        pd.Series(dJ_test.values),
                                        pd.Series(mw[test_idx].values),
                                        pd.Series(n_fold, index=range(len(y_test))),
                                        pd.Series(p, index=range(len(y_test)))], axis=1)], axis=0)

        y_preds = pd.concat([y_preds, pd.DataFrame(y_preds_k)], axis=0)

        n_fold += 1

    y_preds.reset_index(inplace=True, drop=True)
    out = cv_results.reset_index(drop=True)
    out.columns = ['ID', 'y', 'yhat', 'uhat_std', 'uhat', 'yhat_lo', 'yhat_up', 'dJ', 'MW', 'fold', 'p']
    out['y_mg'] = np.log10(10 ** (out['y']) * out['MW'] * 1e3)
    out['yhat_mg'] = np.log10(10 ** (out['yhat']) * out['MW'] * 1e3)
    out['yhat_lo_mg'] = np.log10(10 ** (out['yhat_lo']) * out['MW'] * 1e3)
    out['yhat_up_mg'] = np.log10(10 ** (out['yhat_up']) * out['MW'] * 1e3)
    out['res'] = (out['y']-out['yhat'])

    # Retrain model on all data and obtain internal model predictions
    # - Preprocessing
    #  -- train rdkit pipe
    if 'rdkit' in feat:
        X, _, pipe = rdkit_pipe(X, X)
        X.index = ID

    # -- drop correlated features based on training data
    if any(x in feat for x in ['rdkit', 'maccs']):
        X = feature_selection(X)

    # - Modeling
    # -- define uncertainty model and train on all data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed,
                                                        stratify=y.round().clip(lower=clip_lo, upper=clip_up))

    p = X_train.shape[1]
    kl_loss_weight = 1 / X_train.shape[0]

    model_all = MLP_TFP(p, kl_loss_weight, feat)
    model_all.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                loss=negloglik, metrics=['mse'])
    history = model_all.fit(X_train, y_train, epochs=epochs, verbose=True, validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        batch_size=batch_size)
    model_all.summary()

    # -- apply trained model to predict training data (internal predictions)
    y_pred_list_in = []
    for i in tqdm.tqdm(range(500)):
        y_pred_data_in = model_all.predict(X)
        y_pred_list_in.append(y_pred_data_in)

    y_preds_in = np.concatenate(y_pred_list_in, axis=1)
    y_mean_in = np.mean(y_preds_in, axis=1)
    y_sigma_in = np.std(y_preds_in, axis=1)
    y_low = pd.Series(np.percentile(y_preds_in, q_lower, axis=1))
    y_up = pd.Series(np.percentile(y_preds_in, q_upper, axis=1))

    out_in = pd.DataFrame()
    out_in['ID'] = ID
    out_in['yhat_in'] = y_mean_in
    out_in['uhat_std_in'] = y_sigma_in
    out_in['uhat_in'] = y_up-y_low
    out_in['yhat_in_mg'] = np.log10(10 ** (out_in['yhat_in']) * mw.values * 1e3)

    # -- obtain Jaccard distances between training chemicals and their closest 5 neighbours among other training chemicals (themselves excluded)
    knn = NearestNeighbors(n_neighbors=k_knn + 1, metric='jaccard').fit(data_1024[data_1024.columns[-1024:]], None)
    dist_data, _ = knn.kneighbors(X=data_1024[data_1024.columns[-1024:]], n_neighbors=k_knn, return_distance=True)
    data_1024['dJ_in'] = dist_data.mean(axis=1)
    out_in = pd.merge(out_in, data_1024.reset_index()[['ID', 'dJ_in']], on='ID')

    # Create final output file
    # - merge internal & crossvalidation prediction results
    out = pd.merge(out, out_in, on='ID', copy=False)

    # - merge meta data
    data_pod = pd.read_csv('../PODUAM/data/data_pod_{effect}.csv'.format(effect=effect+subset))
    data_class = pd.read_csv('../PODUAM/data/data_classyfire_{effect}.csv'.format(effect=effect))

    meta = pd.merge(data_pod[['ID', 'casrn', 'name', 'canonical order SMILES']],
                    data_class[['ID', 'Kingdom', 'Superclass', 'Class', 'Subclass']], on='ID', how='left', copy=False)

    out = pd.merge(out, meta, on='ID', copy=False)

    out.to_csv('../PODUAM/manuscript/results/crossvalidation/out_cv_bnn_{CI}_{feat}_{effect}.csv'.format(effect=effect+subset, feat=feat, CI=CI), index=False)

    # Derive fraction of reported PODs within credible intervals ("coverage") for confidence-based calibration assessment
    coverage = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    pred_coverage = calc_coverage_tfp(out, y_preds, coverage)

    out_cov = pd.concat([pd.Series(coverage, name='coverage'), pd.Series(pred_coverage, name='pred_coverage')], axis=1)
    out_cov.to_csv('../PODUAM/manuscript/results/crossvalidation/out_coverage_bnn_{feat}_{effect}.csv'.format(effect=effect+subset, feat=feat))
