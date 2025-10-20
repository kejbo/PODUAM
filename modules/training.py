""" Module containing functions to train and assess UAM and standard ML models

"""

import pickle
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from itertools import combinations
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import BayesSearchCV

from sklearn.metrics import mean_squared_error as sk_mse, mean_absolute_error as sk_mae, r2_score as sk_r2, \
    median_absolute_error as sk_mdae

import sklearn.compose as skcomp
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer

from tensorflow import keras
from scikeras.wrappers import KerasRegressor
from modules.nn_model import MLP_TF

# Error metrics
def mean_absolute_error(yhat, y):
    """ Returns mean absolute error while skipping entries without predicted values

    Inputs
    ----------
    yhat : pandas series, mandatory
        The predicted target values
    y : pandas series, mandatory
        The measured target values

    Outputs
    ----------
    mae: float
        mean absolute error between y and yhat
    """

    flt_nan = ~yhat.isna()
    return sk_mae(y[flt_nan], yhat[flt_nan])


def median_absolute_error(yhat, y):
    """ Returns median absolute error while skipping entries without predicted values

    Inputs
    ----------
    yhat : pandas series, mandatory
        The predicted target values
    y : pandas series, mandatory
        The measured target values

    Outputs
    ----------
    mdae: float
        median absolute error between y and yhat
    """
    flt_nan = ~yhat.isna()
    return sk_mdae(y[flt_nan], yhat[flt_nan])


def mean_squared_error(yhat, y):
    """ Returns mean squared error while skipping entries without predicted values

    Inputs
    ----------
    yhat : pandas series, mandatory
        The predicted target values
    y : pandas series, mandatory
        The measured target values

    Outputs
    ----------
    mse: float
        mean squared error between y and yhat
    """
    flt_nan = ~yhat.isna()
    return sk_mse(y[flt_nan], yhat[flt_nan])


def root_mean_squared_error(yhat, y):
    """ Returns root mean squared error while skipping entries without predicted values

    Inputs
    ----------
    yhat : pandas series, mandatory
        The predicted target values
    y : pandas series, mandatory
        The measured target values

    Outputs
    ----------
    rmse: float
        root mean squared error between y and yhat
    """

    flt_nan = ~yhat.isna()
    return sk_mse(y[flt_nan], yhat[flt_nan], squared=False)


def r2_score(yhat, y):
    """ Returns the cofficient of determination while skipping entries without predicted values
    Note: this metric is sometimes also called cofficient of efficiency

    Inputs
    ----------
    yhat : pandas series, mandatory
        The predicted target values
    y : pandas series, mandatory
        The measured target values

    Outputs
    ----------
    r2: float
        cofficient of determination between y and yhat
    """

    flt_nan = ~yhat.isna()
    return sk_r2(y[flt_nan], yhat[flt_nan])


def r2adj_score(yhat, y, p):
    """ Returns the adjusted cofficient of determination while skipping entries without predicted values
    with a global p

    Inputs
    ----------
    yhat : pandas series, mandatory
        The predicted target values
    y : pandas series, mandatory
        The measured target values
    p: int, mandatory
        The number of feature variables used to train a given model

    Outputs
    ----------
    r2adj: float
        adjusted cofficient of determination between y and yhat
    """

    n = len(y)
    r2v = r2_score(yhat, y)
    return 1 - (1 - r2v) * (n - 1) / (n - p - 1)


def r2adj_score_cv(df, col_yhat='yhat', col_y='y', col_p='p'):
    """ Returns the adjusted cofficient of determination while skipping entries without predicted values
    adjusted for cross-validation setting where p can vary across folds depending on feature pre-processing

    Inputs
    ----------
    df: dataframe, mandatory
        The dataframe containing columns for y, yhat and p
    col_yhat : str, mandatory
        The column name containing the predicted target values
    col_y : str, mandatory
        The column name containing the predicted target values
    col_p: str, mandatory
        The column name containing the number of feature variables used to train the model in a given fold

    Outputs
    ----------
    r2adj: float
        adjusted cofficient of determination between y and yhat
    """

    p = df[col_p].mean()
    n = len(df)
    r2v = r2_score(df[col_yhat], df[col_y])
    return 1 - (1 - r2v) * (n - 1) / (n - p - 1)


# Feature pre-processing
# https://stackoverflow.com/questions/44889508/remove-highly-correlated-columns-from-a-pandas-dataframe/44892279#44892279
def findCorrelation(corr, cutoff=0.9, exact=None):
    """
    This function is the Python implementation of the R function
    `findCorrelation()`.

    Relies on numpy and pandas, so must have them pre-installed.

    It searches through a correlation matrix and returns a list of column names
    to remove to reduce pairwise correlations.

    For the documentation of the R function, see
    https://www.rdocumentation.org/packages/caret/topics/findCorrelation
    and for the source code of `findCorrelation()`, see
    https://github.com/topepo/caret/blob/master/pkg/caret/R/findCorrelation.R

    Inputs:
    -----------
    corr: pandas dataframe.
        A correlation matrix as a pandas dataframe.
    cutoff: float, default: 0.9.
        A numeric value for the pairwise absolute correlation cutoff
    exact: bool, default: None
        A boolean value that determines whether the average correlations be
        recomputed at each step
    
    Outputs:
    --------
    list of column names

    """

    def _findCorrelation_fast(corr, avg, cutoff):

        combsAboveCutoff = corr.where(lambda x: (np.tril(x) == 0) & (x > cutoff)).stack().index

        rowsToCheck = combsAboveCutoff.get_level_values(0)
        colsToCheck = combsAboveCutoff.get_level_values(1)

        msk = avg[colsToCheck] > avg[rowsToCheck].values
        deletecol = pd.unique(np.r_[colsToCheck[msk], rowsToCheck[~msk]]).tolist()

        return deletecol

    def _findCorrelation_exact(corr, avg, cutoff):

        x = corr.loc[(*[avg.sort_values(ascending=False).index] * 2,)]

        if (x.dtypes.values[:, None] == ['int64', 'int32', 'int16', 'int8']).any():
            x = x.astype(float)

        x.values[(*[np.arange(len(x))] * 2,)] = np.nan

        deletecol = []
        for ix, i in enumerate(x.columns[:-1]):
            for j in x.columns[ix + 1:]:
                if x.loc[i, j] > cutoff:
                    if x[i].mean() > x[j].mean():
                        deletecol.append(i)
                        x.loc[i] = x[i] = np.nan
                    else:
                        deletecol.append(j)
                        x.loc[j] = x[j] = np.nan
        return deletecol

    if not np.allclose(corr, corr.T) or any(corr.columns != corr.index):
        raise ValueError("correlation matrix is not symmetric.")

    acorr = corr.abs()
    avg = acorr.mean()

    if exact or exact is None and corr.shape[1] < 100:
        return _findCorrelation_exact(acorr, avg, cutoff)
    else:
        return _findCorrelation_fast(acorr, avg, cutoff)


def feature_selection(X):
    """ This function allows to drop constant and highly correlated features
   from the feature matrix X
   
    Inputs
    ----------
    X: dataframe, mandatory
        The feature matrix 

    Outputs
    ----------
    X_selected: dataframe
        The feature matrix without constant or highly correlated columns
    """

    # - drop constant and highly correlated features
    X_selected = X.drop(columns=X.columns[X.nunique() <= 1])
    correlated = findCorrelation(X_selected.corr(), cutoff=0.9, exact=True)
    X_selected = X_selected.drop(columns=correlated)

    return X_selected


def rdkit_pipe(X_train, X_test):
    """ This function trains a sklearn pipeline to pre-process RDKit features
   by applying Yeo-Johnson transformation to create a more gaussian like distribution,
   standardize features to zero mean and unit variance, and impute missing values from its 
   five nearest neighbors. It returns the pipeline with fitted steps alongside the 
   transformed training and test feature matrices.
   
    Inputs
    ----------
    X_train: dataframe, mandatory
        The feature matrix of the training set
    X_test: dataframe, mandatory
        The feature matrix of the test set

    Outputs
    ----------
    X_train_tf: dataframe
        The pre-processed feature matrix of the training set
    X_test_tf: dataframe
        The pre-processed  feature matrix of the test set
    pipe: class
        The pipeline with fitted steps
    """

    non_fragment_cols = np.nonzero(~X_train.columns.str.contains('fr_'))[0].tolist()
    column_transformer = skcomp.make_column_transformer(
        (PowerTransformer(), non_fragment_cols), remainder='passthrough')
    column_scaler = skcomp.make_column_transformer(
        (StandardScaler(), np.arange(X_train.shape[1])), remainder='passthrough')

    pipe = Pipeline([
        ('transform', column_transformer),
        ('scale', column_scaler),
        ('impute', KNNImputer(n_neighbors=5, weights='distance'))])

    X_train_tf = X_train.replace([np.inf, -np.inf], np.nan)
    X_train_tf = pd.DataFrame(pipe.fit_transform(X_train_tf))
    X_train_tf.columns = X_train.columns

    X_test_tf = X_test.replace([np.inf, -np.inf], np.nan)
    X_test_tf = pd.DataFrame(pipe.transform(X_test_tf))
    X_test_tf.columns = X_test.columns

    return X_train_tf, X_test_tf, pipe


# Cross-validation
def kfold(X, k=5, stratify=None, seed=1):
    """ This function creates stratified or unstratified folds using sklearns KFold and StratifiedKFold classes
   
    Inputs
    ----------
    X: dataframe, mandatory
        The feature matrix
    k: int, default: 5
        The number of folds
    stratify: series, optional
        The variable by which to stratify. If None, no stratification is applied.
    seed: int, default: 0
        random state for creating reproducible folds

    Outputs
    ----------
    folds: list of arrays
        The training and test set indices for each split

    """


    if stratify is None:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds = kf.split(X)

    else:
        stratified_kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = stratified_kf.split(X, stratify)

    return folds

# - Standard ML training
def nested_cv_sklearn(X, y, ID, model, feat, param_grid, k_outer=10, k_inner=5, nrand_cv=10, stratify=None, seed=1):
    """ This function performs nested cross-validation for all sklearn models with a defined number of inner and outer 
    (stratified) folds using sklearns BayesianSearchCV to optimize within a given hyperparamter space.
   
    Inputs
    ----------
    X: dataframe, mandatory
        The feature matrix
    y: series, mandatory
        The target variable
    ID: series, mandatory
        The unique ID for every training/test record
    feat: string, mandatory
        The name of features to tailor pre-processing. options: 'rdkit', 'maccs', 'cddd', 'morgan-512'
    param_grid: dict, mandatory
        dictionary of hyperparameter names and search spaces compatible with BayesSearchCV's search_spaces
    k_outer: int, default: 10
        The number of outer folds for assessing the external prediction performance
    k_inner: int, default: 5
        The number of inner folds for optimizing hyperparameters
    nrand_cv: int, default: 10
        The number of hyperparameter settings that are sampled by BayesSearchCV
    stratify: series, optional
        The variable by which to stratify. If None, no stratification is applied.
    seed: int, default: 0
        random state for creating reproducible results

    Outputs
    ----------
    best_config: pandas dataframe
        Dataframe containing the ID, test set measured and predicted targte values, the number of the outer fold
        and the number of training features in this fold
    outer_results: pandas dataframe
        Dataframe containing the RMSE on the training and test sets for every outer fold, and the number of 
        training features in this fold as well as the optimal hyperparameter settings identified.

    """

    # Outer loop
    cv_outer = kfold(X, k=k_outer, stratify=stratify, seed=seed)
    outer_results = pd.DataFrame()
    best_config = list()
    n_outer = 0

    for train_ix, test_ix in cv_outer:
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        # - Preprocessing
        # -- transform features based on training data
        if 'rdkit' in feat:
            X_train, X_test, _ = rdkit_pipe(X_train, X_test)

        # -- drop correlated features based on training data
        if any(x in feat for x in ['rdkit', 'maccs']):
            X_train = feature_selection(X_train)
            X_test = X_test[X_train.columns]

        p = X_train.shape[1]
    
        # Inner loop
        if stratify is None:
            stratify_inner = None
        else:
            stratify_inner = stratify.iloc[train_ix]

        cv_inner = kfold(X_train, k=k_inner, stratify=stratify_inner, seed=seed)

        # - perform hyperparameter search
        search = BayesSearchCV(model, param_grid, cv=cv_inner, n_iter=nrand_cv, random_state=seed,
                                    scoring='neg_root_mean_squared_error', refit=True)
        result = search.fit(X_train, y_train)
        
        # Assess best model from inner loop in outer loop
        best_model = result.best_estimator_
        
        yhat = pd.Series(best_model.predict(X_test))
        yhat_train = pd.Series(best_model.predict(X_train))

        rmse = root_mean_squared_error(yhat.reset_index(drop=True), y_test.reset_index(drop=True))
        rmse_train = root_mean_squared_error(yhat_train.reset_index(drop=True), y_train.reset_index(drop=True))
        print('>rmse=%.3f, est=%.3f, cfg=%s' % (rmse, result.best_score_, result.best_params_))
        
        # - collect external model predictions and optimized model configuration results
        outer_results = pd.concat([outer_results,
                                   pd.concat([pd.Series(ID[test_ix]),
                                              y_test.reset_index(drop=True),
                                              yhat.reset_index(drop=True),
                                              pd.Series(n_outer, index=range(len(y_test))),
                                              pd.Series(p, index=range(len(y_test)))], axis=1)],axis=0)

        best_config.append([rmse_train.item(), rmse.item(), p, result.best_params_])

        n_outer += 1

    # print the estimated external prediction performance of the model from all outer folds
    outer_rmse = root_mean_squared_error(outer_results.iloc[:, 2], outer_results.iloc[:, 1])
    print('RMSE: %.3f' % outer_rmse)
    
    # return outputs
    best_config = pd.DataFrame(best_config, columns=['RMSE (train)', 'RMSE (test)', 'p', 'best_params'])

    return best_config, outer_results


def nested_cv_mlp(X, y, ID, feat, param_grid, k_outer=10, k_inner=5, nrand_cv=10, seed=1, stratify=None):
    """ This function performs nested cross-validation for the neural network with a defined number of inner 
    and outer (stratified) folds using sklearns BayesianSearchCV to optimize within a given hyperparamter space.
   
    Inputs
    ----------
    X: dataframe, mandatory
        The feature matrix
    y: series, mandatory
        The target variable
    ID: series, mandatory
        The unique ID for every training/test record
    feat: string, mandatory
        The name of features to tailor pre-processing. options: 'rdkit', 'maccs', 'cddd', 'morgan-512'
    param_grid: dict, mandatory
        dictionary of hyperparameter names and search spaces compatible with BayesSearchCV's search_spaces
    k_outer: int, default: 10
        The number of outer folds for assessing the external prediction performance
    k_inner: int, default: 5
        The number of inner folds for optimizing hyperparameters
    nrand_cv: int, default: 10
        The number of hyperparameter settings that are sampled by BayesSearchCV
    stratify: series, optional
        The variable by which to stratify. If None, no stratification is applied.
    seed: int, default: 0
        random state for creating reproducible results

    Outputs
    ----------
    best_config: pandas dataframe
        Dataframe containing the ID, test set measured and predicted targte values, the number of the outer fold
        and the number of training features in this fold
    outer_results: pandas dataframe
        Dataframe containing the RMSE on the training and test sets for every outer fold, and the number of 
        training features in this fold as well as the optimal hyperparameter settings identified.

    """
    # Outer loop
    cv_outer = kfold(X, k=k_outer, stratify=stratify, seed=seed)
    outer_results = pd.DataFrame()
    best_config = list()
    n_outer = 0

    for train_ix, test_ix in cv_outer:
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        # - Preprocessing
        # -- transform features based on training data
        if 'rdkit' in feat:
            X_train, X_test, _ = rdkit_pipe(X_train, X_test)

        # -- drop correlated features based on training data
        if any(x in feat for x in ['rdkit', 'maccs']):
            X_train = feature_selection(X_train)
            X_test = X_test[X_train.columns]

        p = X_train.shape[1]

        # Define MLP model
        callbacks = [keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-5,  # this is absolute
            patience=20,
            restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=10,
                min_delta=1e-5)]

        model = KerasRegressor(MLP_TF, p=p, dropout_rate=0.2, validation_split=0.2, epochs=500, hidden_layers=1, nodes_per_layer=p,
                              # optimizer='adam', loss='mse', metrics=[keras.metrics.RootMeanSquaredError()],
                               callbacks=callbacks, random_state=seed, )

        # Inner loop
        if stratify is None:
            stratify_inner = None
        else:
            stratify_inner = stratify.iloc[train_ix]

        cv_inner = kfold(X_train, k=k_inner, stratify=stratify_inner, seed=seed)

        # - perform hyperparameter search
        search = BayesSearchCV(model, param_grid, cv=cv_inner, n_iter=nrand_cv, random_state=seed,
                               scoring='neg_root_mean_squared_error', refit=True)
        
        result = search.fit(X_train, y_train)
        
        # Assess best model from inner loop in outer loop
        best_model = result.best_estimator_

        yhat = pd.Series(best_model.predict(X_test))
        yhat_train = pd.Series(best_model.predict(X_train))

        rmse = root_mean_squared_error(yhat.reset_index(drop=True), y_test.reset_index(drop=True))
        rmse_train = root_mean_squared_error(yhat_train.reset_index(drop=True), y_train.reset_index(drop=True))
        print('>rmse=%.3f, est=%.3f, cfg=%s' % (rmse, result.best_score_, result.best_params_))

        # - collect external model predictions and optimized model configuration results
        outer_results = pd.concat([outer_results,
                                   pd.concat([pd.Series(ID[test_ix]),
                                              y_test.reset_index(drop=True),
                                              yhat.reset_index(drop=True),
                                              pd.Series(n_outer, index=range(len(y_test))),
                                              pd.Series(p, index=range(len(y_test)))], axis=1)], axis=0)

        best_config.append([rmse_train.item(), rmse.item(), p, result.best_params_])

        n_outer += 1

    # print the estimated external prediction performance of the model from all outer folds
    outer_rmse = root_mean_squared_error(outer_results.iloc[:, 2], outer_results.iloc[:, 1])
    print('RMSE: %.3f' % outer_rmse)

    # return outputs
    best_config = pd.DataFrame(best_config, columns=['RMSE (train)', 'RMSE (test)', 'p', 'best_params'])

    return best_config, outer_results


# - UAM training
def skewnorm_ppf(q, a, loc=0, scale=1):
    """ Wrapper for scipy.stats skewnorm.ppf function
   
    Inputs
    ----------
    q: array_like, mandatory
        lower tail probability
    a: real, mandatory
        skewness parameter with a=0 representing a normal distribution
    loc: array_like, default=0
        location parameter
    scale: array_like, default: 0
        scale parameter

    Outputs
    ----------
    Quantile corresponding to lower tail probability

    """
    return stats.skewnorm.ppf(q, a, loc=loc, scale=scale)


def fit_skewnorm_std(x_values, CI):
    """ Function to convert a confidence interval to a standard deviation by estimating the a, loc and scale parameters 
    of a skewed normal percent point function based on three points given by the median and lower and upper bounds of 
    the confidence interval.
   
    Inputs
    ----------
    x_values: array, mandatory
        array containing the lower confidence bound, median and upper confidence bound
    CI: int, mandatory
        confidence level given in percent, for example: 95 for a 95% confidence interval

    Outputs
    ----------
    skewnorm_std: float
        standard deviation of a skewed normal distribution fitted to the three points of the confidence interval

    """
    y_values = np.array([(1 - CI / 100) / 2, 0.5, 1 - (1 - CI / 100) / 2])
    params, _ = curve_fit(skewnorm_ppf, y_values, x_values, bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]))
    a_estimated, loc_estimated, scale_estimated = params

    skewnorm_dist = stats.skewnorm(a_estimated, loc_estimated, scale_estimated)
    skewnorm_std = skewnorm_dist.std()

    return skewnorm_std

def calc_coverage_tfp(out, y_preds, coverage):
    """ Function to calculate the fraction of measured values within predicted credible interval of the BNN models
    by taking percentiles on the sampled output distribution.

    Inputs
    ----------
    out : pandas dataframe, mandatory
        The structured output from the crossvalidation
    y_preds: array, mandatory
        The samples of the output distributions
    coverage: list, mandatory
        The credible intervals to assess ("expected coverage")

    Outputs
    ----------
    pred_coverage: array
        The observed measured values for each value in the list of credible intervals ("predicted coverage")
    """

    y_covered = pd.DataFrame()
    for num, c in enumerate(coverage):
        q_lower = ((1 - c) / 2) * 100
        q_upper = (1 - (1 - c) / 2) * 100

        y_lower = np.percentile(y_preds, q_lower, axis=1)
        y_upper = np.percentile(y_preds, q_upper, axis=1)
        y_covered[num] = out['y'].between(y_lower, y_upper)

    pred_coverage = y_covered.mean()
    return pred_coverage

# Statistical tests
# - for n-fold crossvalidation
def generalization_error_ci_for_cv(df, loss='MSE', alpha=0.05):
    """ Function to obtain the confidence interval around the MAE or MSE from n-fold crossvalidation 
   
    Inputs
    ----------
    df: dataframe, mandatory
        dataframe containing the measured ('y_mg') and predicted target value ('yhat_mg')
    loss: str, optional
        absolute ('MAE') or squared error ('MSE')

    Outputs
    ----------
    zhat: float
        mean generalization error
    rhat_lo: float
        lower bound on the generalization error for the 95% confidence interval
    rhat_up:
        upper bound on the generalization error for the 95% confidence interval

    """

    n = len(df)

    if loss=='MSE':
        z = (df['yhat_mg']-df['y_mg'])**2  
    elif loss=='MAE':
        z = (df['yhat_mg']-df['y_mg']).abs()
    else:
        raise ValueError("Choose absolute or squared error.")

    zhat = z.mean()
    sigma = ((z-zhat)**2/(n*(n-1))).sum()**0.5
    
    rhat_lo = stats.t.ppf(alpha / 2, n-1, loc=zhat, scale=sigma)
    rhat_up = stats.t.ppf(1 - alpha / 2, n-1, loc=zhat, scale=sigma)

    return zhat, rhat_lo, rhat_up


def correlated_t_test_for_cv_pairwise_I(df_model_A, df_model_B, loss='MSE'):
    """ Function to perform pairwise student t-tests between models A and B to determine whether
    their difference in performance is statistically significant based on n-fold crossvalidation
   
    Inputs
    ----------
    df_model_A: dataframe, mandatory
        dataframe containing the residual ('res') between measured and predicted target value for model A
        from nfold cross-validation
    df_model_B: dataframe, mandatory
        dataframe containing the residual ('res') between measured and predicted target value for model A
        from nfold cross-validation
    loss: str, optional
        loss metric used to assess the prediction performance, can be absolute ('MAE') or squared error ('MSE')

    Outputs
    ----------
    that: float
        t-statistic of the paired student's t-test
    degf: int
        degrees of freedom of the paired student's t-test
    p: float
        p-value of the paired student's t-test
    d: float
        cohen's d effect size  
    zhat: float
        mean difference in generalization error between model A and B
    zhat_lo: float
        lower bound on the difference in generalization error for the 95% confidence interval
    zhat_up: float
        upper bound on the difference in generalization error for the 95% confidence interval

    """

    if len(df_model_A) != len(df_model_B):
        raise ValueError('The two dataframes need to have the same length.')
    else:
        n = len(df_model_A)

    if loss == 'MSE':
        ziA = df_model_A['res'] ** 2
        ziB = df_model_B['res'] ** 2
    elif loss=='MAE':
        ziA = df_model_A['res'].abs()
        ziB = df_model_B['res'].abs()
    else:
        raise ValueError("Choose absolute or squared error.")

    zi = ziA - ziB
    zhat = zi.mean()

    shat2 = 1 / (n * (n - 1)) * sum((zi - zhat) ** 2)
    sigma = shat2 ** 0.5

    that = zhat / sigma     # t-statistic
    degf = n - 1            # degrees of freedom
    p = 2 * stats.t.cdf(-abs(zhat), n - 1, loc=0, scale=sigma) # p-value
    d = (ziA.mean() - ziB.mean()) / ((np.std(ziA, ddof=1)**2 + np.std(ziB, ddof=1)**2) / 2) ** 0.5 # cohen's d

    # confidence interval
    zhat_lo = stats.t.ppf(0.025, degf, loc=zhat, scale=sigma)
    zhat_up = stats.t.ppf(0.975, degf, loc=zhat, scale=sigma)

    print('t=%.3f, degf=%.3f,p=%.3f, d=%.3f, z=%.3f (95%% CI: %.3f - %.3f)' % (that, degf, p, d, zhat, zhat_lo, zhat_up))

    return that, degf, p, d, zhat, zhat_lo, zhat_up

# - for repeated n-fold crossvalidation
def correlated_t_test_for_cv_pairwise_II(df, model_name_A, model_name_B, folds=10, reps=10, loss='MSE'):
    """ Function to perform pairwise student t-tests between models A and B to determine whether
    their difference in performance is statistically significant based on repeated n-fold crossvalidation.
   
    Inputs
    ----------
    df: dataframe, mandatory
        long-shaped dataframe containing the MSE for different folds from repeated n-fold crossvalidation
        with a separate column 'Model' indicating the model name
    df_model_A: str, mandatory
        name of model A
    df_model_B: str, mandatory
        name of model B
    loss: str, default: 'MSE'
        name of the column containing calculated MSE values per fold

    Outputs
    ----------
    p: float
        p-value of the paired student's t-test

    """

    K = folds
    J = reps * folds
    rho = 1 / K

    fltA = (df['Model']==model_name_A)
    fltB = (df['Model']==model_name_B)

    rjA = df.loc[fltA, [loss]].values
    rjB = df.loc[fltB, [loss]].values

    rj = rjA-rjB
    rhat = rj.mean()

    shat2 = 1/(J-1) * sum((rj-rhat)**2)
    sigma = ((1/J + rho/(1-rho)) *shat2)**0.5

    that = rhat/sigma
    p = 2*stats.t.cdf(-abs(that), J-1, loc=0, scale=1)[0]

    return p


def find_high_performing_models(df, p_value=0.05, loss='MSE'):
    """ Function to identify the highest performing models with prediction performance that is not statistically
    significantly different based on pairwise student's t-tests.
   
    Inputs
    ----------
    df: dataframe, mandatory
        long-shaped dataframe containing the MSE for different folds from repeated n-fold crossvalidation
        with a separate column 'Model' indicating the model name
    p_value: float, default: 0.05
        significance level
    loss: str, default: 'MSE'
        name of the column containing calculated MSE values per fold

    Outputs
    ----------
    model_list: list
        list of the best models

    """

    # perform pairwise student's t-test for all possible model combinations
    model_pairs = list(combinations(df['Model'].unique(), 2))

    p_list = list()
    for model_name_A, model_name_B in model_pairs:
        p = correlated_t_test_for_cv_pairwise_II(df, model_name_A, model_name_B, loss=loss)
        p_list.append(p)

    df_p = pd.DataFrame.from_records(model_pairs, columns=['A', 'B'])
    df_p['p_value'] = p_list
    df_p_flt = df_p[df_p['p_value'] > p_value]

    # keep all models that are statistically not significantly different from the best model or 
    # any other model that is not statistically different from the best model
    best_model = df.loc[df['MSE'].idxmin()]['Model']
    model_list = [best_model]
    new_model_list = [best_model]

    #while len(new_model_list) > 0:
    paired_models_list = []
    for model in new_model_list:
        # Find pairs where model appears in column A
        pairs_a = df_p_flt[df_p_flt['B'] == model]['A']
        # Find pairs where model appears in column B
        pairs_b = df_p_flt[df_p_flt['A'] == model]['B']
        # Combine pairs and remove duplicates
        paired_models = list(pd.concat([pairs_a, pairs_b]).drop_duplicates())
        paired_models_list += paired_models

    new_model_list = list(set(paired_models_list).difference(model_list))
    model_list += new_model_list

    return model_list


# Functions to load models and make predictions
def load_models(endpoint='nc', standardized=True):
    """ Function to load the final and diverse models inclduding the pre-processing pipeline
   
    Inputs
    ----------
    endpoint: string, mandatory, default: nc
        select the endpoint for which to load the models, options: nc, rd
    standardize: boolean, default: True
        select whether to load the models trained on standardized (final models) or expanded data sets (diverse models) models

    Outputs
    ----------
    models: dictionary
        dictionary of model objects

    """

    if standardized:
        pipe_name = 'final_model_pipe_rdkit_{endpoint}.pkl'.format(endpoint=endpoint)
        desc_name = 'final_set_desc_rdkit_{endpoint}.csv'.format(endpoint=endpoint)
        model_name = 'final_model_poduam_CI95_rdkit_{endpoint}.pkl'.format(endpoint=endpoint)
        knn_name = 'final_model_knn_{endpoint}.pkl'.format(endpoint=endpoint)
    else:
        pipe_name = 'diverse_model_pipe_rdkit_{endpoint}.pkl'.format(endpoint=endpoint)
        desc_name = 'diverse_set_desc_rdkit_{endpoint}.csv'.format(endpoint=endpoint)
        model_name = 'diverse_model_poduam_CI95_rdkit_{endpoint}.pkl'.format(endpoint=endpoint)
        knn_name = 'diverse_model_knn_{endpoint}.pkl'.format(endpoint=endpoint)

    models=dict()
    models['pipe'] = pickle.load(open('../PODUAM/final_models/' + pipe_name, 'rb'))
    models['desc_list'] = pd.read_csv('../PODUAM/final_models/' + desc_name).iloc[:, 0].tolist()
    models['main'] = pickle.load(open('../PODUAM/final_models/' + model_name, 'rb'))
    models['knn'] = pickle.load(open('../PODUAM/final_models/' + knn_name, 'rb'))

    return models


def make_prediction(models, descriptors, mw):
    """ Function to make predictions for new chemicals with the final or diverse models based on provided descriptors and molecular weights.
   
    Inputs
    ----------
    models: dictionary, mandatory
        dictionary of models loaded with the load_models function
    descriptors: pandas dataframe
        precomputed RDKit descriptors for the chemicals to be predicted
    mw: pandas series
        molecular weights of the chemicals to be predicted

    Outputs
    ----------
    df: pandas dataframe
        dataframe containing the predicted values and their 2.5% and 97.5% confidence bounds

    """

    # Unpack models
    pipe = models['pipe']
    model = models['main']

    # Use the loaded model to make predictions
    X = pd.DataFrame(pipe.transform(descriptors))
    X.columns = descriptors.columns
    X = X[models['desc_list']]

    # - predict chemicals in batches
    model.predict(X)
    y_pred = np.log10(10 ** pd.Series(model.test_y_median_base) * mw['MolWt'] * 1e3) 
    y_lower = np.log10(10 ** pd.Series(model.test_y_lower_uacqrs) * mw['MolWt'] * 1e3) 
    y_upper = np.log10(10 ** pd.Series(model.test_y_upper_uacqrs) * mw['MolWt'] * 1e3)

    df = pd.concat([y_pred, y_lower, y_upper], axis=1)

    return df
