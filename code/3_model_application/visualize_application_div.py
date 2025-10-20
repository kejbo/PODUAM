""" Visualize final models' predictions for the application chemicals

This script allows the user to recreate Figures 3 and 4 analogues from the manuscript
visualizing the predictions results when applying the diverse models to the non-standardized
subset of marketed chemicals based on the structured output file from 
run_diverse_model_for_application.py

"""

from pathlib import Path
from matplotlib import gridspec
from modules.visuals import *

# Load final model predictions of marketed chemicals
df_rd = pd.read_csv('../PODUAM/manuscript/results/application/out_diverse_model_market_{effect}.csv'.format(effect='rd'))
df_nc = pd.read_csv('../PODUAM/manuscript/results/application/out_diverse_model_market_{effect}.csv'.format(effect='nc'))

# Visualisation
# - settings
CI=95
fontsize = 6
letters = 'abcdefghijklmnopqrstuvwxyz'
fig_dir = Path('../PODUAM/manuscript/figures/')

top15 = df_rd['Superclass'].value_counts().sort_values(ascending=False).index[:15]
hue_order = top15.sort_values().to_list() + ['Other']
palette = ['darkslategrey', 'teal', 'aquamarine', 'darkred', 'orangered', 'mediumpurple', 'darkorchid',
           'mediumblue', 'royalblue', 'skyblue', 'darkgoldenrod', 'darkorange', 'gold', 'lightpink', 'hotpink',
           'lightgrey']

# - Figure 3 (chemical space)
fig3_name = 'figure_3_chemspace_diverse_model.png'

# -- define clusters for annotation

c1 = [('group_name', 'inorganic metal\ncompounds'),('chem_name', 'cadmium'),
      ('smiles', '[Cd]'), ('xy', (52, -5)), ('width', 55), ('height', 35), ('angle', 0)]

c2 = [('group_name', 'organotin\ncompounds'),('chem_name', 'tributyltin acetate'),
      ('smiles', 'CCCC[Sn](CCCC)(CCCC)OC(C)=O'),
      ('xy', (113, -128)), ('width', 11), ('height', 14), ('angle', 0)]

clusters = dict()
c_all = c1 + c2 
append_values_to_keys(clusters, c_all)

# -- figure plotting
fig = plt.figure(figsize=(18/2.54, 18/2.54))
gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 1.5, 1], top=0.97, bottom=0, left=0, right=0.93, hspace=0, wspace=0.25)
ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])
ax10 = fig.add_subplot(gs[1, 0])
ax11 = fig.add_subplot(gs[1, 1])
ax20 = fig.add_subplot(gs[2, :2])
ax00.set_title('reproductive/developmental toxicity', fontsize=fontsize, fontweight='bold')
ax01.set_title('general non-cancer toxicity', fontsize=fontsize, fontweight='bold')
plot_chemspace(df_rd, hue_col='yhat_mg', clusters=clusters, legend=False, fontsize=fontsize, ax=ax00)
plot_chemspace(df_nc, hue_col='yhat_mg', clusters=clusters, legend=True, fontsize=fontsize, ax=ax01)
plot_chemspace(df_rd, hue_col='uhat', clusters=clusters, CI=CI, legend=False, fontsize=fontsize, ax=ax10)
plot_chemspace(df_nc, hue_col='uhat', clusters=clusters, CI=CI, legend=True, fontsize=fontsize, ax=ax11)
plot_annotated_clusters(clusters, ax=ax20, fontsize=fontsize)

for n, ax in enumerate([ax00, ax01, ax10, ax11]):
    ax.text(0.03, 1, letters[n], transform=ax.transAxes, fontsize=fontsize + 1, fontweight='bold', va='top', ha='right')
ax20.text(0.02, 0.95, letters[n+1], transform=ax20.transAxes, fontsize=fontsize + 1, fontweight='bold', va='top', ha='right')

fig.savefig(fig_dir / fig3_name, dpi=600)
plt.close()


# - Figure 4 (chemical class ranking)
df_rd['my_class'] = df_rd['Subclass'].fillna(df_rd['Class'] + ' (C)').fillna(df_rd['Superclass'] + ' (S)').fillna('Unknown')
df_nc['my_class'] = df_nc['Subclass'].fillna(df_nc['Class'] + ' (C)').fillna(df_nc['Superclass'] + ' (S)').fillna('Unknown')

highlight_top = 25  # top n classes to highlight
min_chem = 10    # minimum nr of chemicals in class to be included in ranking plots

fig4_name = 'figure_4_class-ranking_diverse_model.png'
fig4 = plt.figure(figsize=(16/2.54, 18.5/2.54))
gs = gridspec.GridSpec(5, 2, height_ratios=(3, 1, 2, 4, 3), hspace=0.07, wspace=0.07, left=0.18, right=0.99, top=0.84, bottom=0.3)
plt.rcParams.update({'font.size': fontsize, 'axes.linewidth': 0.3})

# A: Scatter ranking (class-level)
ax00 = fig4.add_subplot(gs[0, 0])
ax01 = fig4.add_subplot(gs[0, 1], sharex=ax00, sharey=ax00)
plot_group_ranking_scatter(df_rd, col_uhat='uhat', CI=CI, legend=True, highlight_top=highlight_top, min_chem=min_chem,
                           fontsize=fontsize, ax=ax00, hue='Superclass (top 15)', hue_order=hue_order, palette=palette)
plot_group_ranking_scatter(df_nc, col_uhat='uhat', CI=CI, legend=False, highlight_top=highlight_top, min_chem=min_chem,
                           fontsize=fontsize, ax=ax01, hue='Superclass (top 15)', hue_order=hue_order, palette=palette)

# --- buffer axis
ax10 = fig4.add_subplot(gs[1, 0])
ax11 = fig4.add_subplot(gs[1, 1])
ax10.axis('off')
ax11.axis('off')

# B: Nr of chemicals per group (highest ranked classes)
ax20 = fig4.add_subplot(gs[2, 0])
ax21 = fig4.add_subplot(gs[2, 1], sharey=ax20)
plot_group_counts(df_rd, fontsize=fontsize, ax=ax20, hue='Superclass (top 15)', hue_order=hue_order, palette=palette, highlight_top=highlight_top, min_chem=min_chem,)
plot_group_counts(df_nc, fontsize=fontsize, ax=ax21, hue='Superclass (top 15)', hue_order=hue_order, palette=palette, highlight_top=highlight_top, min_chem=min_chem,)

# C: Distribution of PODs as violins and boxplots (highest ranked classes)
ax30 = fig4.add_subplot(gs[3, 0], sharex=ax20)
ax31 = fig4.add_subplot(gs[3, 1], sharex=ax21, sharey=ax30)
plot_group_ranking_violins(df_rd, fontsize=fontsize, ax=ax30, invert_y=True, highlight_top=highlight_top, min_chem=min_chem,
                           hue='Superclass (top 15)', hue_order=hue_order, palette=palette)
plot_group_ranking_violins(df_nc, fontsize=fontsize, ax=ax31, highlight_top=highlight_top, min_chem=min_chem,
                           hue='Superclass (top 15)', hue_order=hue_order, palette=palette)

# D: Distribution of uncertainty as violins and boxplots (highest ranked classes)
ax40 = fig4.add_subplot(gs[4, 0], sharex=ax20)
ax41 = fig4.add_subplot(gs[4, 1], sharex=ax21, sharey=ax40)
plot_group_ranking_violins(df_rd, y_col='uhat', y_label='uncertainty ({CI}% CI width)'.format(CI=CI), highlight_top=highlight_top, min_chem=min_chem,
                           fontsize=fontsize, ax=ax40, hue='Superclass (top 15)', hue_order=hue_order, palette=palette)
plot_group_ranking_violins(df_nc, y_col='uhat', y_label='uncertainty ({CI}% CI width)'.format(CI=CI), highlight_top=highlight_top, min_chem=min_chem,
                           fontsize=fontsize, ax=ax41, hue='Superclass (top 15)', hue_order=hue_order, palette=palette)
ax00.set_title('reproductive/developmental', fontsize=fontsize, fontweight='bold', pad=77)
ax01.set_title('general non-cancer', fontsize=fontsize, fontweight='bold', pad=77)

for ax in [ax20, ax21, ax30, ax31]:
    ax.tick_params(labelbottom=False)
    ax.set_xlabel('')
for ax in [ax01, ax21, ax31, ax41]:
    ax.tick_params(labelleft=False)
    ax.set_ylabel('')
for n, ax in enumerate([ax00, ax01, ax20, ax21, ax30, ax31, ax40, ax41]):
    if n%2==0:
        ax.text(-0.19, 1, letters[n], transform=ax.transAxes, fontsize=fontsize + 1, fontweight='bold', va='top',ha='right')
    else:
        ax.text(-0.02, 1, letters[n], transform=ax.transAxes, fontsize=fontsize + 1, fontweight='bold', va='top', ha='right')

fig4.savefig(fig_dir / fig4_name, dpi=600, transparent=True)
plt.close()
