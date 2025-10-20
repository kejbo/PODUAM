""" Visualize reported POD data

This script allows the user to recreate the SI Figure visualizing the 
distribution of reported PODs while highlighting the 3 most toxic chemicals.

"""

from pathlib import Path
from matplotlib import gridspec
from modules.visuals import *


# Load internal and CV prediction incl. meta data
data_rd = pd.read_csv('../PODUAM/data/data_pod_rd.csv')
data_nc = pd.read_csv('../PODUAM/data/data_pod_nc.csv')

data_rd['POD'] = np.log10(data_rd['POD'])
data_nc['POD'] = np.log10(data_nc['POD'])

# Visualisation
fontsize = 10
fig_dir = Path('../PODUAM/manuscript/figures/')

fig_name = 'SI_histogram_over_POD.png'

fig = plt.figure(figsize=(16, 5))
gs = gridspec.GridSpec(1, 2, left=0.05, right=0.95, top=0.95, bottom=0.1)
ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1], sharey=ax00)
ax00.set_title('reproductive/developmental toxicity', fontsize=12, fontweight='bold')
ax01.set_title('general non-cancer toxicity', fontsize=12, fontweight='bold')

plot_histogram_pod(data_rd, invert=False, x_bin=0.5, ax=ax00)
plot_histogram_pod(data_nc, invert=False, x_bin=0.5, ax=ax01)

fig.savefig(fig_dir / fig_name, dpi=600)
plt.close()
