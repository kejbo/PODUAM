""" Influence of NN architecture on conventional ML model performance

This script allows the user to visualize differences in crossvalidation performance of
NN models with different architectures in terms of number and size of hidden layers.


"""

import argparse
import re
import os
import glob
from matplotlib import gridspec
from pathlib import Path
from modules.visuals import *

# ---- SELECT DATA SET ----
effect = 'nc'   # options: rd, nc
feat = 'rdkit'  # options: 'rdkit', 'maccs', 'cddd', 'morgan-512'

subset = '-std'
seed = 0

# ---- ARGUMENT PARSER (for command-line execution) ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, default=effect, help='options: rd, nc')
    parser.add_argument("--feat", type=str, default=feat, help='options: rdkit, maccs, cddd, morgan-512')

    return parser.parse_args()

# ---- EXECUTION LOGIC ----
if __name__ == "__main__":

        filepath= '../PODUAM/manuscript/results/consensus/architecture/{effect}/'.format(effect=effect+subset)
        config_file = 'NN_{feat}_config_*.csv'.format(feat=feat)
        filepattern = os.path.join(filepath, config_file)
        pattern = r"l(\d+)-([\d.]+)p.csv"

        df_list = []
        for file in glob.glob(filepattern):
                # - find number of layers and nodes
                filename = os.path.basename(file)
                
                match = re.search(pattern, filename)
                if not match: 
                        continue
                layers = int(match.group(1))
                nodes = float(match.group(2))

                # - load data
                data = pd.read_csv(file)

                data['layers'] = layers
                data['nodes'] = nodes

                # Append to list
                df_list.append(data)

        df = pd.concat(df_list, ignore_index=True)
        df_long = df.melt(id_vars=['layers', 'nodes', 'best_params'], value_vars=['RMSE (train)', 'RMSE (test)'],var_name='set', value_name='RMSE')
        df_long['set'] = df_long['set'].str.replace('RMSE \\(|\\)', '', regex=True)

        # Visualisation
        fontsize = 12
        letters = 'abcdefghijklmnopqrstuvwxyz'
        fig_dir = Path('../PODUAM/manuscript/figures/')

        fig_name = 'SI_nn-architecture_{feat}_{effect}.png'.format(feat=feat, effect=effect+subset)
        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(1, 2, wspace=0.1)

        # - by layers
        ax00 = fig.add_subplot(gs[0, 0])
        plot_rmse_nn_architecture(df_long, x='layers', xlabel='Nr of hidden layers', ax=ax00, fontsize=fontsize)

        # - by nodes
        ax01 = fig.add_subplot(gs[0, 1]) #, sharey=ax00
        plot_rmse_nn_architecture(df_long, x='nodes', xlabel='Nr of nodes (multiple of input size p)', ax=ax01, fontsize=fontsize, legend=True)

        ax01.tick_params(labelleft=False)
        ax01.set_ylabel('')
        for n, ax in enumerate([ax00, ax01]):
                ax.text(-0.05, 1, letters[n], transform=ax.transAxes, fontsize=fontsize + 6, fontweight='bold', va='top',
                ha='right')

        plt.show()
        fig.savefig(fig_dir / fig_name, dpi=600)
