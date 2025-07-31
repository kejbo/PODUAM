""" Calculate crossvalidated RMSEs and their confidence intervals

This script allows the user to perform statistical analysis to derive confidence intervals
around the cross-validated RMSEs and assess the statistical significance for the difference
in performance between consensus predictions from conventional ML models and CP models.

"""

import argparse
from modules.training import *

# ---- SELECT DATA SET ----
feat = 'rdkit' # options: rdkit, maccs, cddd, morgan-512

# ---- ARGUMENT PARSER (for command-line execution) ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat", type=str, default=feat, help='options: rdkit, maccs, cddd, morgan-512')
    return parser.parse_args()

# ---- EXECUTION LOGIC ----
if __name__ == "__main__":
        ## UAM
        #  - model settings
        alg = 'RF'
        CI = 95
        # - load crossvalidation prediction results for models trained on standardized subset
        data_rd_std = pd.read_csv('../PODUAM/manuscript/results/crossvalidation/out_cv_cp-{alg}_{CI}_{feat}_{effect}.csv'.format(
                alg=alg, effect='rd-std', feat=feat, CI=CI))
        data_nc_std = pd.read_csv('../PODUAM/manuscript/results/crossvalidation/out_cv_cp-{alg}_{CI}_{feat}_{effect}.csv'.format(
                alg=alg, effect='nc-std', feat=feat, CI=CI))

        CI_RMSE_rd_std = np.array(generalization_error_ci_for_cv(data_rd_std, loss='MSE', alpha=0.05))**0.5
        CI_RMSE_nc_std = np.array(generalization_error_ci_for_cv(data_nc_std, loss='MSE', alpha=0.05))**0.5

        print('STD CI RMSE rep/dev: {CI}'.format(CI=CI_RMSE_rd_std))
        print('STD CI RMSE general: {CI}'.format(CI=CI_RMSE_nc_std))

        ## CONSENSUS
        # - load crossvalidation prediction results for models trained on standardized subset
        consensus_rd_std = pd.read_csv('../PODUAM/manuscript/results/consensus/out_cv_consensus_{effect}.csv'.format(effect='rd-std'))
        consensus_nc_std = pd.read_csv('../PODUAM/manuscript/results/consensus/out_cv_consensus_{effect}.csv'.format(effect='nc-std'))

        CI_RMSE_rd_std_cons = np.array(generalization_error_ci_for_cv(consensus_rd_std, loss='MSE', alpha=0.05))**0.5
        CI_RMSE_nc_std_cons = np.array(generalization_error_ci_for_cv(consensus_nc_std, loss='MSE', alpha=0.05))**0.5

        print('Consensus STD CI RMSE rep/dev : {CI}'.format(CI=CI_RMSE_rd_std_cons))
        print('Consensus STD CI RMSE general: {CI}'.format(CI=CI_RMSE_nc_std_cons))

        ## Check if performance difference is statistically significant between UAM ALL and Consensus ALL
        correlated_t_test_for_cv_pairwise_I(data_rd_std, consensus_rd_std)
        correlated_t_test_for_cv_pairwise_I(data_nc_std, consensus_nc_std)
