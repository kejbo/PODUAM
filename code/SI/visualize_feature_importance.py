
""" Load final models and analyze feature importance

This script allows the user to load the final models created with train_final_models.py and analyze
the feature importance across the underlying quantile regression forests.

"""

import pickle
from pathlib import Path
from matplotlib import gridspec

from modules.visuals import *

# Load data
model_rd = pickle.load(open('../PODUAM/final_models/final_model_poduam_CI95_rdkit_rd.pkl', 'rb'))
model_nc = pickle.load(open('../PODUAM/final_models/final_model_poduam_CI95_rdkit_nc.pkl', 'rb'))
rdkit_desc_rd = pd.read_csv('../PODUAM/final_models/final_set_desc_rdkit_rd.csv').iloc[:, 0].tolist()
rdkit_desc_nc = pd.read_csv('../PODUAM/final_models/final_set_desc_rdkit_nc.csv').iloc[:, 0].tolist()

# Feature importance
importance_rd, sorted_features_rd = collect_feature_importance(model_rd, rdkit_desc_rd)
importance_nc, sorted_features_nc = collect_feature_importance(model_nc, rdkit_desc_nc)

importance_long_rd = importance_rd.melt(var_name='feature', value_name='importance')
importance_long_nc = importance_nc.melt(var_name='feature', value_name='importance')

# Visualisation
fontsize = 12
letters = 'abcdefghijklmnopqrstuvwxyz'
fig_dir = Path('../PODUAM/manuscript/figures/')

# - feature importance


fig_name = 'SI_final_models_feature-importance.png'
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.5)

ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])

plot_feature_importance(importance_long_rd, sorted_features_rd, ax=ax00, top_n=50, title="Top 50 features by mean importance")
plot_feature_importance(importance_long_nc, sorted_features_nc, ax=ax01, top_n=50, title="Top 50 features by mean importance")
ax00.set_title('reproductive/developmental toxicity', fontsize=fontsize + 2, fontweight='bold', pad=30)
ax01.set_title('general non-cancer toxicity', fontsize=fontsize + 2, fontweight='bold', pad=30)
fig.savefig(fig_dir / fig_name, dpi=600)
