""" Visualize correlations between UAM models' median predictions and uncertainty estimates

This script allows the user to perform a correlation analysis between the different UAM model's
median predictions and uncertainty estimates based on the outputs from run_cv_BNN.py run_cv_CP.py
with different molecular descriptors. It creates the heatmaps, faceted correlation plots and overall
correlation plot between mean prediction uncertainty and prediction variability across UAM models 
that are available in the SI.

"""

from pathlib import Path
from matplotlib import gridspec
from modules.visuals import *

# Load data
filepath = r"../PODUAM/manuscript/results/crossvalidation/"

yhat_matrix_rd, uhat_matrix_rd, ytrue_rd = load_uam_data(filepath, 'rd')
yhat_matrix_nc, uhat_matrix_nc, ytrue_nc = load_uam_data(filepath, 'nc')

# Visualisation
fontsize = 12
letters = 'abcdefghijklmnopqrstuvwxyz'
fig_dir = Path('../PODUAM/manuscript/figures/')

# - Correlation heatmaps
fig_name = 'SI_uam_models_corr_heatmap.png'
fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2])

fig.text(0.5, 0.95, "Correlation of model prediction medians", ha='center', fontsize=fontsize, fontstyle='italic')

ax00 = fig.add_subplot(gs[0, 0])
sns.heatmap(yhat_matrix_rd.corr(), annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, vmin=0, vmax=1, ax=ax00, cbar=False)

ax01 = fig.add_subplot(gs[0, 1])
sns.heatmap(yhat_matrix_nc.corr(), annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, vmin=0, vmax=1, ax=ax01)

fig.text(0.5, 0.48, "Correlation of model prediction uncertainty", ha='center', fontsize=fontsize, fontstyle='italic')

ax10 = fig.add_subplot(gs[1, 0])
sns.heatmap(uhat_matrix_rd.corr(), annot=True, fmt=".2f", cmap="Reds", linewidths=0.5, vmin=0, vmax=1, ax=ax10, cbar=False)

ax11 = fig.add_subplot(gs[1, 1])
sns.heatmap(uhat_matrix_nc.corr(), annot=True, fmt=".2f", cmap="Reds", linewidths=0.5, vmin=0, vmax=1, ax=ax11)

plt.subplots_adjust(top=0.94, left=0.1, right=0.99, hspace=0.3, wspace=0.25)
ax00.set_title('reproductive/developmental toxicity', fontsize=fontsize + 2, fontweight='bold', pad=30)
ax01.set_title('general non-cancer toxicity', fontsize=fontsize + 2, fontweight='bold', pad=30)
fig.savefig(fig_dir / fig_name, dpi=600)

# - Correlation scatter plot
fig_name = 'SI_uam_models_corr_scatter_prediction_rd.png'
fig = plot_correlation_scatter_pairgrid(yhat_matrix_rd, "Pairwise scatter plots of model prediction medians")
fig.fig.suptitle('reproductive/developmental toxicity', fontsize=fontsize+2, fontweight='bold')
fig.savefig(fig_dir / fig_name, dpi=600)

fig_name = 'SI_uam_models_corr_scatter_uncertainty_rd.png'
fig = plot_correlation_scatter_pairgrid(uhat_matrix_rd, "Pairwise scatter plots of model prediction uncertainty",lim=(0,10))
fig.fig.suptitle('reproductive/developmental toxicity', fontsize=fontsize+2, fontweight='bold')
fig.savefig(fig_dir / fig_name, dpi=600)

fig_name = 'SI_uam_models_corr_scatter_prediction_nc.png'
fig = plot_correlation_scatter_pairgrid(yhat_matrix_nc, "Pairwise scatter plots of model prediction medians")
fig.fig.suptitle('general non-cancer toxicity', fontsize=fontsize+2, fontweight='bold')
fig.savefig(fig_dir / fig_name, dpi=600)

fig_name = 'SI_uam_models_corr_scatter_uncertainty_nc.png'
fig = plot_correlation_scatter_pairgrid(uhat_matrix_nc, "Pairwise scatter plots of model prediction uncertainty",lim=(0,10))
fig.fig.suptitle('general non-cancer toxicity', fontsize=fontsize+2, fontweight='bold')
fig.savefig(fig_dir / fig_name, dpi=600)

# - Prediction variability vs mean predicted uncertainty
df_rd = pd.DataFrame({
    'uncertainty_mean': uhat_matrix_rd.mean(axis=1),
    'prediction_ci': yhat_matrix_rd.quantile(0.975, axis=1) - yhat_matrix_rd.quantile(0.025, axis=1),
    'residual_abs': (yhat_matrix_rd.mean(axis=1) - ytrue_rd).abs()
})

df_nc = pd.DataFrame({
    'uncertainty_mean': uhat_matrix_nc.mean(axis=1),
    'prediction_ci': yhat_matrix_nc.quantile(0.975, axis=1) - yhat_matrix_nc.quantile(0.025, axis=1),
    'residual_abs': (yhat_matrix_nc.mean(axis=1) - ytrue_nc).abs()
})


# Plotting - Variability (prediction_ci) vs Mean Uncertainty
fig_name = 'SI_uam_models_corr_uncertainty-variability.png'
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(1, 2)

ax00 = fig.add_subplot(gs[0, 0])
plot_uncertainty_variability_correlation(df_rd, ax=ax00, fontsize=fontsize)

ax01 = fig.add_subplot(gs[0, 1])
plot_uncertainty_variability_correlation(df_nc, ax=ax01, fontsize=fontsize)

ax00.set_title('reproductive/developmental toxicity', fontsize=fontsize + 2, fontweight='bold', pad=30)
ax01.set_title('general non-cancer toxicity', fontsize=fontsize + 2, fontweight='bold', pad=30)
fig.savefig(fig_dir / fig_name, dpi=600)
