""" Visualize final models' predictions for the application chemicals

This script allows the user to recreate Figures 3 and 4 from the manuscript (and SI analogues)
visualizing the predictions results when applying the final models to a large set of standardized 
marketed chemicals based on the structured output file from run_final_model_for_application.py

"""

from pathlib import Path
from matplotlib import gridspec
from modules.visuals import *

# Load final model predictions of marketed chemicals
df_rd = pd.read_csv('../PODUAM/manuscript/results/application/out_final_model_market_{effect}.csv'.format(effect='rd'))
df_nc = pd.read_csv('../PODUAM/manuscript/results/application/out_final_model_market_{effect}.csv'.format(effect='nc'))

# Visualisation
# - settings
CI=95
fontsize = 12
letters = 'abcdefghijklmnopqrstuvwxyz'
fig_dir = Path('../PODUAM/manuscript/figures/')

top15 = df_rd['Superclass'].value_counts().sort_values(ascending=False).index[:15]
hue_order = top15.sort_values().to_list() + ['Other']
palette = ['darkslategrey', 'teal', 'aquamarine', 'darkred', 'orangered', 'mediumpurple', 'darkorchid',
           'mediumblue', 'royalblue', 'skyblue', 'darkgoldenrod', 'darkorange', 'gold', 'lightpink', 'hotpink',
           'lightgrey']

# - Figure 3 (chemical space)
fig3_name = 'figure_3_chemspace_final_model.png'

# -- define clusters for annotation
c1 = [('group_name', 'poly- and perfluoro\nalkyl substances'), ('chem_name', 'PFOS'),
      ('smiles', 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F'),
      ('xy', (5, -145)), ('width', 60), ('height', 55), ('angle', 0)]

c2 = [('group_name', 'aliphatic organo\nthiophosphates'), ('chem_name', 'terbufos'),
      ('smiles', 'CCOP(=S)(OCC)SCSC(C)(C)C'),
      ('xy', (45, -126)), ('width', 18), ('height', 15), ('angle', 0)]

c3 = [('group_name', 'aromatic organo\nthiophosphates'), ('chem_name', 'chlorpyrifos'),
      ('smiles', 'CCOP(=S)(OCC)OC1=NC(Cl)=C(Cl)C=C1Cl'),
      ('xy', (-46, -96)), ('width', 16), ('height', 16), ('angle', 0)]

c4 = [('group_name', 'PBBs and PBDEs\n'),('chem_name', 'BDE-47'),
      ('smiles', 'BrC1=CC(Br)=C(OC2=CC=C(Br)C=C2Br)C=C1'),
      ('xy', (-125, -95)), ('width', 25), ('height', 25), ('angle', 0)]

c5 = [('group_name', 'PCBs and PCBEs\n'),('chem_name', 'PCB-77'),
      ('smiles', 'C1=CC(=C(C=C1C2=CC(=C(C=C2)Cl)Cl)Cl)Cl'),
      ('xy', (-95, 0)), ('width', 25), ('height', 20), ('angle', 0)]

c6 = [('group_name', 'benzodioxins and\nbenzofurans'),('chem_name', 'TCDD'),
      ('smiles', 'ClC1=C(Cl)C=C2OC3=CC(Cl)=C(Cl)C=C3OC2=C1'),
      ('xy', (-160, 51)), ('width', 18), ('height', 23), ('angle', 0)]

c7 = [('group_name', 'polychlorinated\ncycloaliphatics'),('chem_name', 'dienochlor'),
      ('smiles', 'ClC1=C(Cl)C(Cl)(C(Cl)=C1Cl)C1(Cl)C(Cl)=C(Cl)C(Cl)=C1Cl'),
      ('xy', (76, 48)), ('width', 12), ('height', 12), ('angle', 0)]

c8 = [('group_name', 'steroids\n'),('chem_name', 'betamethasone'),
      ('smiles', '[H][C@@]12C[C@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C'),
      ('xy', (112, 123)), ('width', 60), ('height', 40), ('angle', -30)]

c9 = [('group_name', 'natural\nproducts'), ('chem_name', 'aporphine'),  # alkaloids, phenylpropanoids & polyketides, lignans,
      ('smiles', 'CN1CCC2=C3C1CC4=CC=CC=C4C3=CC=C2'),
      ('xy', (-5, 148)), ('width', 40), ('height', 35), ('angle', 0)]

c10 = [('group_name', 'phenothiazines\n'),('chem_name', 'phenothiazine'),
      ('smiles', 'C1=CC=C2C(=C1)NC3=CC=CC=C3S2'),
      ('xy', (-37, 95)), ('width', 18), ('height', 20), ('angle', 0)]

clusters = dict()
c_all = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10
append_values_to_keys(clusters, c_all)

# -- figure plotting
fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 1.5, 1], top=0.97, bottom=0, left=0, right=0.93, hspace=0, wspace=0.25)
ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])
ax10 = fig.add_subplot(gs[1, 0])
ax11 = fig.add_subplot(gs[1, 1])
ax20 = fig.add_subplot(gs[2, :2])
ax00.set_title('reproductive/developmental toxicity', fontsize=fontsize + 2, fontweight='bold')
ax01.set_title('general non-cancer toxicity', fontsize=fontsize + 2, fontweight='bold')
plot_chemspace(df_rd, hue_col='yhat_mg', clusters=clusters, legend=False, fontsize=fontsize, ax=ax00)
plot_chemspace(df_nc, hue_col='yhat_mg', clusters=clusters, legend=True, fontsize=fontsize, ax=ax01)
plot_chemspace(df_rd, hue_col='uhat', clusters=clusters, CI=CI, legend=False, fontsize=fontsize, ax=ax10)
plot_chemspace(df_nc, hue_col='uhat', clusters=clusters, CI=CI, legend=True, fontsize=fontsize, ax=ax11)
plot_annotated_clusters(clusters, ax=ax20, fontsize=fontsize)

ax00.text(0.03, 1, letters[0], transform=ax00.transAxes, fontsize=fontsize + 6, fontweight='bold', va='top', ha='left')
ax10.text(0.03, 1, letters[1], transform=ax10.transAxes, fontsize=fontsize + 6, fontweight='bold', va='top', ha='left')
ax20.text(0.03, 0.9, letters[2], transform=ax20.transAxes, fontsize=fontsize + 6, fontweight='bold', va='top', ha='left')
ax20.text(0.5, 0.9, 'selected chemical clusters', transform=ax20.transAxes, fontsize=fontsize + 2, fontweight='bold', va='top', ha='center')

fig.savefig(fig_dir / fig3_name, dpi=600)
plt.close()


# - Figure 4 (chemical class ranking)
df_rd['my_class'] = df_rd['Subclass'].fillna(df_rd['Class'] + ' (C)').fillna(df_rd['Superclass'] + ' (S)').fillna('Unknown')
df_nc['my_class'] = df_nc['Subclass'].fillna(df_nc['Class'] + ' (C)').fillna(df_nc['Superclass'] + ' (S)').fillna('Unknown')

fig4_name = 'figure_4_class-ranking_final_model.png'
fig4 = plt.figure(figsize=(16, 17))
gs = gridspec.GridSpec(5, 2, height_ratios=(3, 1, 1, 4, 3), hspace=0.05, wspace=0.05,
                       left=0.07, right=0.98, top=0.88, bottom=0.27)

# A: Scatter ranking (class-level)
ax00 = fig4.add_subplot(gs[0, 0])
ax01 = fig4.add_subplot(gs[0, 1], sharex=ax00, sharey=ax00)
plot_group_ranking_scatter(df_rd, col_uhat='uhat', CI=CI, legend=True, highlight_top=50, min_chem=10,
                           fontsize=fontsize, ax=ax00, hue='Superclass (top 15)', hue_order=hue_order, palette=palette)
plot_group_ranking_scatter(df_nc, col_uhat='uhat', CI=CI, legend=False, highlight_top=50, min_chem=10,
                           fontsize=fontsize, ax=ax01, hue='Superclass (top 15)', hue_order=hue_order, palette=palette)

# --- buffer axis
ax10 = fig4.add_subplot(gs[1, 0])
ax11 = fig4.add_subplot(gs[1, 1])
ax10.axis('off')
ax11.axis('off')

# B: Nr of chemicals per group (highest ranked classes)
ax20 = fig4.add_subplot(gs[2, 0])
ax21 = fig4.add_subplot(gs[2, 1], sharey=ax20)
plot_group_counts(df_rd, fontsize=fontsize, ax=ax20, hue='Superclass (top 15)', hue_order=hue_order, palette=palette)
plot_group_counts(df_nc, fontsize=fontsize, ax=ax21, hue='Superclass (top 15)', hue_order=hue_order, palette=palette)

# C: Distribution of PODs as violins and boxplots (highest ranked classes)
ax30 = fig4.add_subplot(gs[3, 0], sharex=ax20)
ax31 = fig4.add_subplot(gs[3, 1], sharex=ax21, sharey=ax30)
plot_group_ranking_violins(df_rd, fontsize=fontsize, ax=ax30, invert_y=True,
                           hue='Superclass (top 15)', hue_order=hue_order, palette=palette)
plot_group_ranking_violins(df_nc, fontsize=fontsize, ax=ax31,
                           hue='Superclass (top 15)', hue_order=hue_order, palette=palette)

# D: Distribution of uncertainty as violins and boxplots (highest ranked classes)
ax40 = fig4.add_subplot(gs[4, 0], sharex=ax20)
ax41 = fig4.add_subplot(gs[4, 1], sharex=ax21, sharey=ax40)
plot_group_ranking_violins(df_rd, y_col='uhat', y_label='uncertainty ({CI}% CI width)'.format(CI=CI),
                           fontsize=fontsize, ax=ax40, hue='Superclass (top 15)', hue_order=hue_order, palette=palette)
plot_group_ranking_violins(df_nc, y_col='uhat', y_label='uncertainty ({CI}% CI width)'.format(CI=CI),
                           fontsize=fontsize, ax=ax41, hue='Superclass (top 15)', hue_order=hue_order, palette=palette)
ax00.set_title('reproductive/developmental', fontsize=fontsize + 2, fontweight='bold', pad=120)
ax01.set_title('general non-cancer', fontsize=fontsize + 2, fontweight='bold', pad=120)
for ax in [ax20, ax21, ax30, ax31]:
    ax.tick_params(labelbottom=False)
    ax.set_xlabel('')
for ax in [ax01, ax21, ax31, ax41]:
    ax.tick_params(labelleft=False)
    ax.set_ylabel('')
for n, ax in enumerate([ax00, ax20, ax30, ax40]):
    ax.text(-0.11, 1, letters[n], transform=ax.transAxes, fontsize=fontsize + 6, fontweight='bold', va='top',
            ha='right')

fig4.savefig(fig_dir / fig4_name, dpi=600, transparent=True)
plt.close()


# - Create table 
col_list = ['Superclass', 'my_class', 'total_count', 'measured_count', 'potent_frac',
            'CASRN', 'PREFERRED_NAME', 'IUPAC_NAME', 'log POD (95% CI)', 'uhat', 'dJ',
             'training log POD', 'all reported log POD']

most_potent_list_rd = df_rd.groupby('my_class_top')['yhat_mg'].median().sort_values().index  # [:10]
df_rd_most_potent = pd.DataFrame()
for c in most_potent_list_rd:
    rd_c = df_rd[df_rd['my_class'] == c].sort_values(by='yhat_mg')  # [:3]
    rd_c['log POD ({CI}% CI)'.format(CI=CI)] = rd_c[['yhat_mg', 'yhat_lo_mg', 'yhat_up_mg']].apply(
        lambda row: f"{row['yhat_mg']:.2f} [{row['yhat_lo_mg']:.2f}, {row['yhat_up_mg']:.2f}]", axis=1)
    rd_c['training log POD'] = rd_c['POD_mg_training'].round(2)
    rd_c['all reported log POD'] = rd_c['POD_mg_all'].round(2)
    rd_c['dJ'] = rd_c['dJ'].round(2)
    rd_c['total_count'] = (df_rd['my_class'] == c).sum()
    rd_c['measured_count'] = (~rd_c['POD_mg_training'].isna() & ~rd_c['POD_mg_all'].isna()).sum()
    rd_c['potent_count'] = ((df_rd['my_class'] == c) & (df_rd['yhat_mg'] <= df_rd['yhat_mg'].quantile(0.01))).sum()
    rd_c['potent_frac'] = rd_c['potent_count']/rd_c['total_count']
    df_rd_most_potent = pd.concat([df_rd_most_potent, rd_c[col_list]], axis=0)
df_rd_most_potent.to_csv('../PODUAM/manuscript/results/application/most_potent_final_model_market_rd.csv', index=False)

most_potent_list_nc = df_nc.groupby('my_class_top')['yhat_mg'].median().sort_values().index  # [:10]
df_nc_most_potent = pd.DataFrame()
for c in most_potent_list_nc:
    nc_c = df_nc[df_nc['my_class'] == c].sort_values(by='yhat_mg')  # [:3]
    nc_c['log POD (95% CI)'] = nc_c[['yhat_mg', 'yhat_lo_mg', 'yhat_up_mg']].apply(
        lambda row: f"{row['yhat_mg']:.2f} [{row['yhat_lo_mg']:.2f}, {row['yhat_up_mg']:.2f}]", axis=1)
    nc_c['training log POD'] = nc_c['POD_mg_training'].round(2)
    nc_c['all reported log POD'] = nc_c['POD_mg_all'].round(2)
    nc_c['dJ'] = nc_c['dJ'].round(2)
    nc_c['total_count'] = (df_nc['my_class'] == c).sum()
    nc_c['measured_count'] = (~nc_c['POD_mg_training'].isna() & ~nc_c['POD_mg_all'].isna()).sum()
    nc_c['potent_count'] = ((df_nc['my_class'] == c) & (df_nc['yhat_mg'] < df_nc['yhat_mg'].quantile(0.01))).sum()
    nc_c['potent_frac'] = nc_c['potent_count'] / nc_c['total_count']
    df_nc_most_potent = pd.concat([df_nc_most_potent, nc_c[col_list]], axis=0)
df_nc_most_potent.to_csv('../PODUAM/manuscript/results/application/most_potent_final_model_market_nc.csv', index=False)


# SI
# - Histograms of Jaccard distances and uncertainty
cv_rd = pd.read_csv('../PODUAM/manuscript/results/crossvalidation/out_cv_cp-RF_95_rdkit_rd-std.csv')
cv_nc = pd.read_csv('../PODUAM/manuscript/results/crossvalidation/out_cv_cp-RF_95_rdkit_nc-std.csv')

fig_name = 'SI_histogram_pod-U-dJ_final_model.png'

fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(3, 2, left=0.1, right=0.95, top=0.95, bottom=0.05)
ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])
ax10 = fig.add_subplot(gs[1, 0])
ax11 = fig.add_subplot(gs[1, 1])
ax20 = fig.add_subplot(gs[2, 0])
ax21 = fig.add_subplot(gs[2, 1])
ax00.set_title('reproductive/developmental toxicity', fontsize=fontsize+2, fontweight='bold')
ax01.set_title('general non-cancer toxicity', fontsize=fontsize+2, fontweight='bold')

plot_histogram_with_cumulative(cv_rd, df_rd, col_x='yhat_mg', x_label='predicted POD [log(mg/kg-d)]',
                               sety=(True, False), fontsize=fontsize, ax=ax00)
plot_histogram_with_cumulative(cv_nc, df_nc, col_x='yhat_mg', x_label='predicted POD [log(mg/kg-d)]',
                               sety=(False, True), fontsize=fontsize, ax=ax01)
plot_histogram_with_cumulative(cv_rd, df_rd, col_x='uhat', x_label='95% CI width',
                               sety=(True, False), fontsize=fontsize, ax=ax10)
plot_histogram_with_cumulative(cv_nc, df_nc, col_x='uhat', x_label='95% CI width',
                               sety=(False, True), fontsize=fontsize, ax=ax11)
plot_histogram_with_cumulative(cv_rd, df_rd, col_x='dJ', x_label='mean Jaccard distance (5NN)',
                               sety=(True, False), fontsize=fontsize, ax=ax20)
plot_histogram_with_cumulative(cv_nc, df_nc, col_x='dJ', x_label='mean Jaccard distance (5NN)',
                               sety=(False, True), fontsize=fontsize, ax=ax21)

for n, ax in enumerate([ax00, ax10, ax20]):
    ax.text(-0.11, 1, letters[n], transform=ax.transAxes, fontsize=fontsize + 6, fontweight='bold', va='top',
            ha='right')
fig.savefig(fig_dir / fig_name, dpi=600)
plt.close()
