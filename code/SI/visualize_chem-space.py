""" Run t-SNE to obtain two-dimensional embedding of marketed chemicals

This script allows the user to visualize the two-dimensional embedding of the marketed chemicals
obtained with the t-SNE algorithm applied to Morgan fingerprints. Each marketed chemical is
colored by the 15 most frequent chemical superclass obtained with ClassyFire following the work
presented in von Borries et al. (2023), doi: 10.1021/acs.est.3c05300

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from pathlib import Path

# Load marketed chemicals with their TSNE coordinate and ClassyFire classification
df_tsne = pd.read_csv('../PODUAM/data/data_tsne_market.csv')
df_classyfire = pd.read_csv('../PODUAM/data/data_classyfire_market.csv')

class_col = ['Kingdom', 'Superclass', 'Class', 'Subclass']
df = pd.merge(df_tsne, df_classyfire[['INCHIKEY'] + class_col], on='INCHIKEY')

top15 = df.groupby('Superclass').count()['TSNE1'].sort_values(ascending=False).index[:15]
df['Superclass_top15'] = df['Superclass'].where(df['Superclass'].isin(top15), 'Other')

# Plot figure
fig_dir = Path('../PODUAM/manuscript/figures/')
fig_name = 'SI_chemspace_superclasses.png'

c = 'Superclass_top15'

palette = ['darkslategrey', 'teal', 'aquamarine', 'darkred', 'orangered', 'mediumpurple', 'darkorchid',
           'mediumblue', 'royalblue', 'skyblue', 'darkgoldenrod', 'darkorange', 'gold',  'lightpink', 'hotpink',
           'lightgrey']

order = top15.sort_values().to_list() + ['Other']

fig, ax = plt.subplots(figsize=(18, 10))
plt.subplots_adjust(left=0.1, right=0.65)
sns.scatterplot(df, x='TSNE1', y='TSNE2', hue=c, hue_order=order, palette=palette, edgecolor=None, alpha=1, s=2, legend=None)
ax.axis('off')

# custom legend
occurrence = df[c].value_counts(normalize=True)
ax1 = inset_axes(ax,
                 width=1,  # inch
                 height=9,  # inch
                 bbox_transform=ax.transAxes,  # relative axes coordinates
                 bbox_to_anchor=(1.25, 1.1),  # relative axes coordinates
                 loc=2)  # loc=lower left corner

width, height, pad = 0.5, 20, 0.25
y = 0
ax1.axis('off')

for n, s in enumerate(order[::-1]):
    h = max(height * occurrence[s], 0.1)
    ax1.add_patch(patches.Rectangle((0, y), width, h, facecolor=palette[::-1][n]))
    ax1.text(-0.5, y + h / 2 - 0.1, "{:.1%}".format(occurrence[s]),
             fontdict={'size': 10})
    ax1.text(width + 0.1, y + h / 2 - 0.1, s,
             fontdict={'size': 10})
    y = y + h + pad

ax1.text(-0.5, y + h / 2 + 0.1, "$\\bf{Total\ number\ of\ chemicals}$: " + str(df.shape[0]),
             fontdict={'size': 10})

plt.xlim([0, 1])
plt.ylim([0, y])

fig.savefig(fig_dir / fig_name, dpi=600)

