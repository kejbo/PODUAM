"""  Visualize BNN models' prediction performance and calibration

This script allows the user to create the BNN analogues to Figure 1 from the manuscript
visualizing the prediction performance and uncertainty calibration of the BNN models 
based on the outputs from run_cv_BNN.py. You can select the input file based on 
descriptors (feat = 'rdkit', 'maccs', 'cddd' or 'morgan-512') used for training.

"""

import argparse
from pathlib import Path
from matplotlib import gridspec
from modules.visuals import *

# ---- SELECT DATA SET ----
feat = 'rdkit' # options: rdkit, maccs, cddd, morgan-512

# ---- ARGUMENT PARSER (for command-line execution) ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat", type=str, default=feat, help='options: rdkit, maccs, cddd, morgan-512')
    return parser.parse_args()

# ---- EXECUTION LOGIC ----
if __name__ == "__main__":
    # Model settings
    CI = 95

    # Load internal and crossvalidation predictions incl. meta data
    data_rd = pd.read_csv('../PODUAM/manuscript/results/crossvalidation/out_cv_bnn_{CI}_{feat}_{effect}.csv'.format(
        effect='rd-std', feat=feat, CI=CI))
    data_nc = pd.read_csv('../PODUAM/manuscript/results/crossvalidation/out_cv_bnn_{CI}_{feat}_{effect}.csv'.format(
        effect='nc-std', feat=feat, CI=CI))

    # Load coverage data for confidence-based calibration/ECE
    cov_rd = pd.read_csv('../PODUAM/manuscript/results/crossvalidation/out_coverage_bnn_{feat}_{effect}.csv'.format(
        effect='rd-std', feat=feat))
    cov_nc = pd.read_csv('../PODUAM/manuscript/results/crossvalidation/out_coverage_bnn_{feat}_{effect}.csv'.format(
        effect='nc-std', feat=feat))


    # Visualisation
    # -- settings
    fontsize = 14
    letters = 'abcdefghijklmnopqrstuvwxyz'
    fig_dir = Path('../PODUAM/manuscript/figures/')

    # -- Figure 1
    fig_name = 'figure_1_performance_bnn_{CI}_{feat}_std.png'.format(feat=feat, CI=CI)

    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(5, 2, hspace=0.01, wspace=0.01)

    # A: Prediction performance
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1], sharey=ax00)
    plot_prediction_performance(data_rd, mg=True, internal=True, ax=ax00, fontsize=fontsize)
    plot_prediction_performance(data_nc, mg=True, internal=True, ax=ax01, fontsize=fontsize)

    # B: Uncertainty histogram
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1], sharex=ax10, sharey=ax10)
    plot_histogram_uncertainty(data_rd, ax10, fontsize)
    plot_histogram_uncertainty(data_nc, ax11, fontsize)

    # B: Confidence-based calibration
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1], sharey=ax20)
    plot_calibration_confidence(cov_rd['coverage'], cov_rd['pred_coverage'], ax=ax20, fontsize=fontsize)
    plot_calibration_confidence(cov_nc['coverage'], cov_nc['pred_coverage'], ax=ax21, fontsize=fontsize)

    # C: Error-based calibration
    ax30 = fig.add_subplot(gs[3, 0])
    ax31 = fig.add_subplot(gs[3, 1], sharey=ax30)
    plot_calibration_error(data_rd, ax=ax30, fontsize=fontsize)
    plot_calibration_error(data_nc, ax=ax31, fontsize=fontsize)

    # D: Distance-based calibration
    ax40 = fig.add_subplot(gs[4, 0])
    ax41 = fig.add_subplot(gs[4, 1], sharey=ax40)
    plot_calibration_distance(data_rd, ax=ax40, fontsize=fontsize)
    plot_calibration_distance(data_nc, ax=ax41, fontsize=fontsize)

    ax00.set_title('reproductive/developmental toxicity', fontsize=fontsize + 2, fontweight='bold', pad=20)
    ax01.set_title('general non-cancer toxicity', fontsize=fontsize + 2, fontweight='bold', pad=20)
    for ax in [ax01, ax11, ax21, ax31]:
        ax.tick_params(labelleft=False)
        ax.set_ylabel('')
    for n, ax in enumerate([ax00, ax10, ax20, ax30, ax40]):
        ax.text(-0.15, 1, letters[n], transform=ax.transAxes, fontsize=fontsize + 6, fontweight='bold', va='top',
                ha='right')
    gs.tight_layout(fig)
    gs.update(left=0.1)
    fig.savefig(fig_dir / fig_name, dpi=300)
    plt.close()


    # SI
    # - plot ENCE = f(B)
    batches = [5, 10, 20, 35, 50, 70, 100, 150, 200]

    fig_name = 'SI_ENCE_over_nbatches_bnn_{CI}_{feat}_std.png'.format(feat=feat, CI=CI)

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 2, left=0.05, top=0.95, bottom=0.1)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1], sharey=ax00)
    ax00.set_title('reproductive/developmental toxicity', fontsize=12, fontweight='bold')
    ax01.set_title('general non-cancer toxicity', fontsize=12, fontweight='bold')

    plot_ence(data_rd, batches, ax=ax00, fontsize=fontsize)
    plot_ence(data_rd, batches, ax=ax01, fontsize=fontsize)

    fig.savefig(fig_dir / fig_name, dpi=300)
    plt.close()
