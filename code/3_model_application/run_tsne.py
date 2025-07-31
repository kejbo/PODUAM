""" Run t-SNE to obtain two-dimensional embedding of marketed chemicals

This script allows the user to obtain a two-dimensional embedding of the marketed chemicals
by applying the t-SNE algorithm to Morgan fingerprints provided through 
run_feature_calculation_for_application.py. The derived t-SNE coordinates are essential meta-data
which is combined with the model predictions in run_final_model_for_application.py and used to 
create Figure 3 of the manuscript with visualize_application_std.py.

"""

import numpy as np
import pandas as pd
from openTSNE.sklearn import TSNE

# Load morgan fingerprints as embedding features
nbits = 1024
ref_path = '../PODUAM/features/data_market_morgan-{nbits}.csv'.format(nbits=nbits)
df = pd.read_csv(ref_path)

fps = np.array(df.iloc[:, -nbits:].astype('bool'))

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=100, n_iter=2000, learning_rate='auto',
            initialization='pca', metric='jaccard', random_state=42, verbose=3)
df_tsne = pd.DataFrame(tsne.fit_transform(fps))
df_tsne.columns = ['TSNE1', 'TSNE2']
df_tsne.index = df['INCHIKEY']

# Save TSNE embedding of marketed chemicals
df_tsne.to_csv('../PODUAM/data/data_tsne_market.csv', index=True)
