""" Visualize CP models' prediction performance and calibration for external test set

This script allows the user to visualize the prediction performance and uncertainty calibration
of the CP models trained on the standardized dataset for a challenging test set of non-standardized chemnicals
based on the outputs from run_final_model_for_external-test.py. You can select the POD (effect = 'rd' or 'nc').

"""
import sys
sys.path.append('../PODUAM/')

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

    # Load CV and external test set predictions
    data = pd.read_csv('../PODUAM/manuscript/results/external/out_final_model_ext_{effect}.csv'.format(effect=effect))
    data['subset'] = np.where(data['Error'] == ' Inorganic', 'inorganic', 'non-standardized organic')
    data['set'] = 'non-standardized (ext)'

    data_cv = pd.read_csv('../PODUAM/manuscript/results/crossvalidation/out_cv_cp-RF_95_rdkit_{effect}-std.csv'.format( effect=effect))
    data_cv['set'] = 'standardized (cv)'

    col_list = ['ID', 'y', 'yhat', 'yhat_lo', 'yhat_up', 'uhat', 'y_mg', 'yhat_mg',
        'yhat_lo_mg', 'yhat_up_mg', 'res', 'uhat_std', 'dJ', 'casrn', 'name',
        'canonical order SMILES', 'Kingdom', 'Superclass', 'Class', 'Subclass', 'set']

    data_merged = pd.concat([data[col_list], data_cv[col_list]])

    data_merged.to_csv('../PODUAM/manuscript/results/external/out_final_models_ext_{effect}_merged.csv'.format(effect=effect), index=False)

    # Visualization
    fontsize = 6
    letters = 'abcdefghijklmnopqrstuvwxyz'
    fig_dir = Path('../PODUAM/manuscript/figures/')

    # - Figure 2
    fig_name = 'SI_final_models_external_{effect}.png'.format(effect=effect)

    fig = plt.figure(figsize=(16/2.54, 16/2.54))
    gs = gridspec.GridSpec(3, 2, hspace=0.01, wspace=0.01)
    plt.rcParams.update({'font.size': fontsize, 'axes.linewidth': 0.3})

    order = ['standardized (cv)', 'non-standardized (ext)']
    hue_order = ['reported POD', 'predicted POD']
    palette = ['black', 'mediumblue']

    # a: Prediction performance
    ax00 = fig.add_subplot(gs[0, 0])
    plot_prediction_performance(data, mg=True, lim=(-4,4), internal=False, test_label='Test data (ext)', ax=ax00, fontsize=fontsize)

    # b: Overall comparison standardized vs. diverse - RMU/RMSE
    ax01 = fig.add_subplot(gs[0, 1])
    plot_non_vs_standardized_rmse(data_merged, groupby='set', ax=ax01, fontsize=fontsize, palette=palette)

    # c: Uncertainty histogram
    ax10 = fig.add_subplot(gs[1, 0])
    plot_histogram_with_cumulative(data, data_cv, 'uhat', x_label='prediction uncertainty (95% CI width)', sety=(True,True), set_label=('non-standardized (ext)', 'standardized (cv)'), 
                                colors=('mediumblue', 'cornflowerblue'), text_pos='right', ax=ax10, fontsize=fontsize)

    # d: Distance histogram 
    ax11 = fig.add_subplot(gs[1, 1])
    plot_histogram_with_cumulative(data, data_cv, 'dJ', x_label='Jaccard distance', sety=(True,True), set_label=('non-standardized (ext)', 'standardized (cv)'), 
                                colors=('mediumblue', 'cornflowerblue'), text_pos='left',ax=ax11, fontsize=fontsize)

    # e: Error-based calibration
    ax20 = fig.add_subplot(gs[2, 0])
    plot_calibration_error(data, col_uhat='uhat_std', ax=ax20, fontsize=fontsize)

    # f: Distance-based calibration
    ax21 = fig.add_subplot(gs[2, 1])
    plot_calibration_distance(data, col_uhat='uhat_std', ax=ax21, fontsize=fontsize)

    title = 'reproductive/developmental toxicity' if effect=='rd' else 'general non-cancer toxicity'
    fig.suptitle(title, fontsize=fontsize, fontweight='bold', y=0.99)
    for n, ax in enumerate([ax00, ax01, ax10, ax11, ax20, ax21]):
        ax.text(-0.21, 1, letters[n], transform=ax.transAxes, fontsize=fontsize + 1, fontweight='bold', va='top', ha='right')
    gs.tight_layout(fig)
    gs.update(left=0.1)
    fig.savefig(fig_dir / fig_name, dpi=600)
