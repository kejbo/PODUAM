""" Create consensus predictions from standard ML models

This script allows the user to derive consensus predictions from the nested
cross-validation of standard ML models across combinations of six algorithms
(MLR, KNN, NN, SVM, RF, XGB) with four molecular descriptors (RDKIT, MACCS, CDDD, MORGAN).
It also recreates several figures visualizing the standard ML prediction performance found
in the supplemental information (SI) of the manuscript.

"""

import argparse
import glob
from pathlib import Path
from matplotlib import gridspec
from modules.training import *
from modules.visuals import *

# ---- SELECT DATA SET ----
effect = 'nc'   # options: rd, nc
subset = '-std'     # options: '', -std

# ---- ARGUMENT PARSER (for command-line execution) ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, default=effect, help='options: rd, nc')
    parser.add_argument("--subset", type=str, default=subset, help='options: '', -std')
    return parser.parse_args()

# ---- EXECUTION LOGIC ----
if __name__ == "__main__":

    # Merge prediction results from nested cross-validation for all ten repetitions
    seed_list = np.arange(10)
    out_all = pd.DataFrame()

    for seed in seed_list:
        dir_path = '../PODUAM/manuscript/results/consensus/{effect}/Seed_{seed:02}/*_out.csv'.format(effect=effect+subset, seed=seed)
        results_list = glob.glob(dir_path)
        results_names = [x.split('\\')[1].split('_out')[0] for x in results_list]

        
        for results_path, name in zip(results_list, results_names):
            out = pd.read_csv(results_path)

            out['Seed'] = [seed]*len(out)
            out['Model'] = [name]*len(out)
            out_all = pd.concat([out_all, out], axis=0)

    # Create consensus predictions
    # - calculate performance metrics for every fold from the repeated nested cross-validation for all models
    out_all.reset_index(inplace=True, drop=True)
    out_all = out_all.replace([np.inf, -np.inf], np.nan)
    df_metrics_fold = pd.DataFrame()
    df_metrics_fold['MAE'] = out_all.groupby(['Seed', 'Model', 'fold']).apply(lambda x: mean_absolute_error(x['yhat'], x['y']))
    df_metrics_fold['MdAE'] = out_all.groupby(['Seed', 'Model', 'fold']).apply(lambda x: median_absolute_error(x['yhat'], x['y']))
    df_metrics_fold['RMSE'] = out_all.groupby(['Seed', 'Model', 'fold']).apply(lambda x: root_mean_squared_error(x['yhat'], x['y']))
    df_metrics_fold['MSE'] = df_metrics_fold['RMSE']**2
    df_metrics_fold['R2'] = out_all.groupby(['Seed', 'Model', 'fold']).apply(lambda x: r2_score(x['yhat'], x['y']))
    df_metrics_fold.reset_index(inplace=True)

    # - find best performing models with no statistical difference in their prediction power
    best_models = find_high_performing_models(df_metrics_fold)

    # - create consensus predictions as average across best models
    out_cons = out_all[out_all['Model'].isin(best_models)]
    out_cons = out_cons.groupby(['Seed', 'fold', 'ID'])[['y', 'yhat', 'y_mg', 'yhat_mg', 'p']].mean().reset_index()
    out_cons['Model'] = 'consensus_consensus'

    out_all_cons = pd.concat([out_all, out_cons]).reset_index(drop=True)

    # -- calculate performance metrics for consensus predictions for every fold from the repeated nested cross-validation
    df_metrics_fold_cons = pd.DataFrame()
    df_metrics_fold_cons['MAE'] = out_all_cons.groupby(['Seed', 'Model', 'fold']).apply(lambda x: mean_absolute_error(x['yhat'], x['y']))
    df_metrics_fold_cons['MdAE'] = out_all_cons.groupby(['Seed', 'Model', 'fold']).apply(lambda x: median_absolute_error(x['yhat'], x['y']))
    df_metrics_fold_cons['RMSE'] = out_all_cons.groupby(['Seed', 'Model', 'fold']).apply(lambda x: root_mean_squared_error(x['yhat'], x['y']))
    df_metrics_fold_cons['R2'] = out_all_cons.groupby(['Seed', 'Model', 'fold']).apply(lambda x: r2_score(x['yhat'], x['y']))
    df_metrics_fold_cons.reset_index(inplace=True)
    df_metrics_fold_cons[['Algorithm', 'Features']] = df_metrics_fold_cons['Model'].str.split(pat='_', n=1, expand=True)
    df_metrics_fold_cons['Features'].replace(['maccs', 'rdkit', 'cddd', 'morgan-512'],
                                        ['MACCS fingerprint', 'RDKIT descriptors', 'CDDD embedding', 'MORGAN fingerprint'],
                                        inplace=True)
    df_metrics_fold_cons = df_metrics_fold_cons.infer_objects()

    # Create structured output file for cross-validated consensus predictions as mean across repetitions
    cv_cons = out_cons.groupby('ID')[['y', 'yhat', 'y_mg', 'yhat_mg', 'p']].mean()

    cv_cons['res'] = cv_cons['y'] - cv_cons['yhat']
    cv_cons.reset_index(inplace=True)

    data_pod = pd.read_csv('../PODUAM/data/data_pod_{effect}.csv'.format(effect=effect+subset))
    data_class = pd.read_csv('../PODUAM/data/data_classyfire_{effect}.csv'.format(effect=effect))

    meta = pd.merge(data_pod[['ID', 'casrn', 'name', 'canonical order SMILES']],
                    data_class[['ID', 'Kingdom', 'Superclass', 'Class', 'Subclass']], on='ID', how='left', copy=False)

    cv_cons = pd.merge(cv_cons, meta, on='ID', copy=False)

    cv_cons.to_csv('../PODUAM/manuscript/results/consensus/out_cv_consensus_{effect}.csv'.format(effect=effect+subset), index=False)

    # SI visualisations
    fontsize = 14
    letters = 'abcdefghijklmnopqrstuvwxyz'
    fig_dir = Path('../PODUAM/manuscript/figures/')

    # - Consensus prediction performance
    fig_name = 'SI_consensus_performance_{effect}.png'.format(effect=effect + subset)
    fig, ax = plt.subplots()
    plot_prediction_performance(cv_cons, mg=True, internal=False, ax=ax, fontsize=fontsize)
    plt.savefig(fig_dir / fig_name, dpi=300)
    plt.close()

    # - Performance metric comparison for different standard ML models
    fig_name = 'SI_standard-ML_metrics_comparison_{effect}.png'.format(effect=effect + subset)

    metrics_list =['MAE', 'MdAE', 'RMSE', 'R2']
    bench = pd.Series(index=metrics_list, dtype=float)
    y_mean = pd.Series(np.resize(out_cons['y'].mean(), len(out)))
    bench['MAE'] = mean_absolute_error(y_mean, out['y'])
    bench['MdAE'] = median_absolute_error(y_mean, out['y'])
    bench['RMSE'] = root_mean_squared_error(y_mean, out['y'])
    bench['R2'] = r2_score(y_mean, out['y'])

    limits = dict(MAE=((df_metrics_fold_cons['MAE'].min()-0.1).round(1), (bench['MAE']+0.1).round(1)),
                MdAE=((df_metrics_fold_cons['MdAE'].min()-0.1).round(1), (bench['MdAE']+0.1).round(1)),
                RMSE=((df_metrics_fold_cons['RMSE'].min()-0.1).round(1), (bench['RMSE']+0.1).round(1)),
                R2=((bench['R2']-0.1).round(1), (df_metrics_fold_cons['R2'].max()+0.1).round(1)))

    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(4, 1)

    for num, metric in enumerate(metrics_list):
        ax = fig.add_subplot(gs[num])
        nested_cv_performance_comparison(df_metrics_fold_cons, metric, bench[metric], limits[metric], ax=ax, fontsize=fontsize)

    for n,ax in enumerate(fig.axes):
        ax.text(0.02, 0.95, letters[n], transform=ax.transAxes, fontsize=fontsize + 6, fontweight='bold', va='top',
                ha='left')    
    plt.savefig(fig_dir / fig_name, dpi=300)
    plt.close()
