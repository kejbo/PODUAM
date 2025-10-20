""" Visualize CP models' prediction performance and calibration for diverse subset

This script allows the user to recreate Figure 2 from the manuscript visualizing the 
prediction performance and uncertainty calibration of the CP models trained with the 
expanded dataset for the non-standardized subset based on the outputs from run_cv_CP.py.
You can select the POD (effect = 'rd' or 'nc').

"""

import argparse
from pathlib import Path
from matplotlib import gridspec
from modules.visuals import *

# ---- SELECT DATA SET ----
effect = 'rd'   # options: rd, nc

# ---- ARGUMENT PARSER (for command-line execution) ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, default=effect, help='options: rd, nc')
    return parser.parse_args()

# ---- EXECUTION LOGIC ----
if __name__ == "__main__":
    # Model settings
    alg = 'RF'
    CI = 95
    feat='rdkit'

    # Load internal and crossvalidation predictions incl. meta data (for model trained on ALL)
    data = pd.read_csv('../PODUAM/manuscript/results/crossvalidation/out_cv_cp-{alg}_{CI}_{feat}_{effect}.csv'.format(
        alg=alg, effect=effect, feat='rdkit', CI=CI))

    # Load IDs for diverse and standardized subsets
    id_div = pd.read_csv('../PODUAM/data/data_pod_{effect}-ext.csv'.format(effect=effect))['ID']
    id_std = pd.read_csv('../PODUAM/data/data_pod_{effect}-std.csv'.format(effect=effect))['ID']

    conditions = [(data['ID'].isin(id_div)), # diverse
                (data['ID'].isin(id_std))]  # standardized
    choices = ['non-standardized', 'standardized']

    data['subset'] = np.select(conditions, choices, default=np.nan)

    data.to_csv('../PODUAM/manuscript/results/crossvalidation/out_cv_cp-{alg}_{CI}_{feat}_{effect}_subsets.csv'.format(
        alg=alg, effect=effect, feat='rdkit', CI=CI), index=False)

    data_div = data[data['subset'] == 'non-standardized']
    data_std = data[data['subset'] == 'standardized']

    # Visualization
    fontsize = 6
    letters = 'abcdefghijklmnopqrstuvwxyz'
    fig_dir = Path('../PODUAM/manuscript/figures/')
    palette = ['black', 'mediumblue']

    # - Figure 2
    fig_name = 'figure_2_diverse_models_non-standardized_{effect}.png'.format(alg=alg, feat=feat, CI=CI, effect=effect)

    fig = plt.figure(figsize=(16/2.54, 16/2.54))
    gs = gridspec.GridSpec(3, 2, hspace=0.01, wspace=0.01, height_ratios=[1.5, 1, 1])
    plt.rcParams.update({'font.size': fontsize, 'axes.linewidth': 0.3})
    
    # a: Overall comparison standardized vs. non-standardized - POD
    ax00 = fig.add_subplot(gs[0, 0])
    plot_non_vs_standardized_violins(data, ax=ax00, fontsize=fontsize, palette=palette)

    # b: Overall comparison standardized vs. diverse - RMU/RMSE
    ax01 = fig.add_subplot(gs[0, 1])
    plot_non_vs_standardized_rmse(data, groupby='subset', ax=ax01, fontsize=fontsize, palette=palette)

    # c: Prediction performance
    ax10 = fig.add_subplot(gs[1, 0])
    plot_prediction_performance(data_div, mg=True, internal=True, ax=ax10, fontsize=fontsize)

    # d: Uncertainty histogram
    ax11 = fig.add_subplot(gs[1, 1])
    plot_histogram_with_cumulative(data_div, data_std, 'uhat', x_label='prediction uncertainty (95% CI width)', sety=(True,True), 
                                   set_label=('non-standardized', 'standardized'),  colors=('mediumblue', 'cornflowerblue'), 
                                   text_pos='right', ax=ax11, fontsize=fontsize)

    # e: Error-based calibration
    ax20 = fig.add_subplot(gs[2, 0])
    plot_calibration_error(data_div, col_uhat='uhat_std', ax=ax20, fontsize=fontsize)

    # f: Distance-based calibration
    ax21 = fig.add_subplot(gs[2, 1])
    plot_calibration_distance(data_div, col_uhat='uhat_std', ax=ax21, fontsize=fontsize)

    title = 'reproductive/developmental toxicity' if effect=='rd' else 'general non-cancer toxicity'
    fig.suptitle(title, fontsize=fontsize, fontweight='bold', y=0.99)
    for n, ax in enumerate([ax00, ax01, ax10, ax11, ax20, ax21]):
        ax.text(-0.21, 1, letters[n], transform=ax.transAxes, fontsize=fontsize + 1, fontweight='bold', va='top', ha='right')
    gs.tight_layout(fig)
    gs.update(left=0.1)
    fig.savefig(fig_dir / fig_name, dpi=600)
    plt.close()
