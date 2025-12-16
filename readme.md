# PODUAM
PODUAM is a free and open-source uncertainty-aware prediction model (UAM) 
to predict reproductive/developmental (rd) and general non-cancer (nc) points of departure (POD) 
with quantified uncertainty in the form of 95% confidence intervals.

The models were published in [von Borries et al. (2025)](doi.org/10.1038/s41467-025-67374-4)

The models can also be run via the [PODUAM Shiny web app](https://dtu-quantitative-sustainability-assessment.shinyapps.io/poduam)


## System requirements
This project is built with Python 3.11 and requires the packages specified in requirements.txt (including their dependencies).
Detailed installation instructions are provided below.
The code has been primarily developed and tested on Windows 11, but compatibility with macOS and Linux systems has been considered. 
Some additional steps may be required for successful installation on non-Windows systems.

All scripts can, in principle, be run on standard hardware, with execution times ranging from a few minutes to several hours per script. 
Most of the modeling work was performed on an HP EliteBook 840 G8 (11th Gen Intel® Core™ i7-1185G7, 3.00 GHz, 4 cores, 32 GB RAM).
Some of the more time-consuming tasks were executed on DTU’s high-performance computing (HPC) clusters to take advantage of parallel computing capabilities.

## Installation guide

1. Create an environment with Python 3.11

```
conda create -n "poduam_env" python=3.11
conda activate poduam_env
```

3. Install all required packages and versions from requirements.txt

```
cd your/path/to/PODUAM
pip install -r requirements.txt
```

(Installation with pip is recommended as not all packages are available through conda)

2. Add the file location of PODUAM to the PYTHONPATH of your environment

```
conda env config vars set PYTHONPATH=your/path/to/PODUAM
conda deactivate
conda activate poduam_env
```

(Reactivating environment is necessary to make the change active)

**Additional Setup for macOS Users**
Some dependencies (e.g., dm-tree) require compiling C++ code during installation. 
To enable this, you may need to install build tools before running pip install -r requirements.txt.

Install Xcode Command Line Tools:
```
xcode-select --install
```

Install CMake:
```
brew install cmake # system-wide
# or
pip install cmake # inside poduam_env
```

**Typical installation time:** <1 hour

## Structure

### code 
contains all code required to replicate figures and results presented in the manuscript organized by analysis, 
including (0) training feature calculation, (1) model training and performance assessment, (2) comparisons with conventional ML and non-standardized chemicals, and (3) model application to predict marketed chemicals as well as (4) additional analysis results presented in the SI.

### demo
contains example code to run the final or diverse models for a set of chemicals based on provided standardized (using Mansouri et al. protocol) or other SMILES

### data
contains data for training and application chemicals
- data_pod_*.csv: training data sets derived from Aurisano et al. (2019) and de Vito et al. (2024) with curated and harmonized chemical structures (SMILES)
- data_marketed_chemicals.csv: set of 134,129 marketed chemicals used to demonstrate the application of the model derived by von Borries et al. (2023)
- data_tsne_market.csv: two-dimensional embedding of marketed chemicals (described with morgan fingerprints) created with t-SNE
- data_classyfire_*.csv: chemical classification derived with ClassyFire by Djoumbou et al. (2016)

### features
contains feature representations for training and application chemicals generated using run_feature_calculation.py (training chemicals) 
and run_feature_calculation_for_application.py (application chemicals) located in ../code/0_features

### final_models
contains the final models for data pre-processing, prediction modeling and similarity calculations
- final_model_*.pkl: Models trained on standardized chemicals for data pre-processing (pipe), similarity calculations (knn) and prediction modeling (poduam_CI95_rdkit)
- diverse_model_*.pkl: Models trained on non-standardized chemicals for data pre-processing (pipe), similarity calculations (knn) and prediction modeling (poduam_CI95_rdkit)

### manuscript
contains all modelling results and figures presented in the manuscript and the S

### modules
Contains reusable functions for different tasks:

- structures.py: Curation and harmonization of chemical structures.
- training.py: Model training and validation functions.
- visuals.py: Visualization utilities.

## Instructions for use

### Demo
To run the final or diverse models on a new set of chemicals provide an input file with standardized SMILES (using Mansouri et al. protocol) or other SMILES in the *demo* folder and run demo.py. Final models (trained on standardized dataset) are used with standardized SMILES, otherwise diverse models (trained on expanded dataset) are used. 

- **demo.py**: Provide an input file containing a set of chemicals described by their SMILES in a column named 'SMILES' (plus optional other columns) (input_file = path). Specify whether SMILES are standardized (using Mansouri et al. protocol) or not (standardized = True or False). The script then calculates the training features, loads the models and computes the predictions of reproductive/developmental (rd) and general non-cancer (nc) PODs with 95% confidence intervals as well as the structural similarity to training chemicals (described by mean Jaccard distances). Results are saved under the output path provided (output_file = path). Expected run time: <5 minutes for 10 chemicals.

### Reproduction
To reproduce the figures and results from the manuscript, run the scripts in the **code** folder  in the order described below (grouped by analysis steps). Since results from earlier steps are pre-computed and stored in the **features** and **manuscript/results**  folders, you may also skip ahead to later steps if desired. Recomputing all steps of the analysis will take several days on a standard computer.

#### 0_features
Scripts for generating training features.
- **run_feature_calculation.py**: Calculate RdKit, MACCS, and Morgan fingerprints for training chemicals a selected POD (effect = 'rd' or 'nc'). CDDD descriptors (Winther et al., 2019) are pre-computed due to their dependency on a different virtual environment.
- **run_feature_calculation_for_application.py**: Calculates RdKit, MACCS, and Morgan fingerprints for marketed chemicals. CDDD descriptors (Winther et al., 2019) are pre-computed due to their dependency on a different virtual environment.

#### 1_model_performance
Scripts for CP and BNN model training and validation.
- **run_cv_CP.py**: Train and validate CP models with 10-fold cross-validation. You can select the POD (effect = 'rd' or 'nc'), descriptors (feat = 'rdkit', 'maccs', 'cddd' or 'morgan-512') and whether the standardized or expanded dataset should be used (subset = '-std' or '').
- **run_cv_CP_ece.py**: Train CP models at different confidence levels to assess confidence-based calibration. You can select the POD (effect = 'rd' or 'nc'), descriptors (feat = 'rdkit', 'maccs', 'cddd' or 'morgan-512') and whether the standardized or expanded dataset should be used (subset = '-std' or '').
- **run_cv_BNN.py**: Train and validate BNN models with 10-fold cross-validation. You can select the POD (effect = 'rd' or 'nc'), descriptors (feat = 'rdkit', 'maccs', 'cddd' or 'morgan-512') and whether the standardized or expanded dataset should be used (subset = '-std' or ''). Includes outputs to derive confidence-based calibration.
- **train_final_models.py**: Re-train models using the RDKit descriptors on the standardized datasets (final models) or the expanded datasets (diverse models) (subset='std' or '') for a selected POD (effect = 'rd' or 'nc)
- **visualize_cv_*.py**: Reproduce Figure 1 from the manuscript (and SI analogues). You can select the input file based on descriptors (feat = 'rdkit', 'maccs', 'cddd' or 'morgan-512') used for training (* = CP or BNN)
- **visualize_model_variation.py**: Perform a correlation analysis between the different UAM model's median predictions and uncertainty estimates based on the outputs from run_cv_BNN.py and run_cv_CP.py, reproducting SI heatmaps and scatter plots.


#### 2_model_comparison
Scripts for comparing model performance for non-standardized chemicals
- **visualize_div_CP.py**: Reproduce Figure 2 from the manuscript (and SI analogues) visualizing the prediction performance and uncertainty calibration of CP models trained with the expanded dataset (diverse models) for the non-standardized subset.  You can select the POD (effect = 'rd' or 'nc').
- **run_final_models_for_external-test.py**: Apply CP models trained on the standardized dataset (final models) on the non-standardized chemicals as an external test set. You can select the POD (effect = 'rd' or 'nc').
- **visualize_ext_CP.py**: Visualize the prediction performance and uncertainty calibration of CP models trained with the standardized dataset (final models) for the challenging test set of non-standardized chemicals, reproducting SI figures. You can select the POD (effect = 'rd' or 'nc').

##### consensus
Scripts for training conventional ML models and creating consensus predictions based on nested 10-fold crossvalidation.
Due to iterative model training in the repeated nested crossvalidation, reproducing these results is very time consuming on a standard computer and has been done with an HPC cluster.
- **run_*.py**: Train conventional ML models (KNN, MLR, NN, RF, SVM, XGB) with nested 10-fold cross-validation for a selected POD (effect = 'rd' or 'nc) on the standardized datasets.
- **create_consensus.py**: Derive consensus predictions from conventional ML models for a selected POD (effect = 'rd' or 'nc).
- **consensus_comparison.py**: Perform statistical tests for performance differences between CP models and consensus predictions. You can select the descriptors used for training of CP models (feat = 'rdkit', 'maccs', 'cddd' or 'morgan-512', use 'rdkit' for final models).

#### 3_model_application:
Scripts for applying models to the marketed chemicals
- **run_tsne.py**: Create a two-dimensional embedding of the marketed chemicals by applying the t-SNE algorithm to Morgan fingerprints (n=1024) following von Borries et al. (2023). Note that t-SNE is sensitive to minor variations in the virtual environment, which can result in slight differences in the embedding. Consequently, the manual annotations in Figure 3 may not align with the t-SNE embedding created in your own environment.
- **run_*_models_for_application.py**: Apply the final or diverse models to the set of standardized or non-standardized marketed chemicals to obtain selected POD predictions (effect = 'rd' or 'nc') (\*=finale or diverse)
- **visualize_application_*.py**: Reproduce Figures 3 and 4 (and SI analogues) from the manuscript visualizing the prediction results when applying the final or diverse models to the set of standardized or non-standardized marketed chemicals.

#### SI:
Scripts for generating additional SI figures
- **visualize_POD.py**: Create histograms of reported PODs with annotations of top 3 most toxic reported chemicals.
- **visualize_feature_importance.py**: Analyze the feature importance across the quantile regression forests underlying the final models.
- **run_NN_architecture.py**: Adapted version of run_NN.py to test crossvalidation performance for different NN architectures in terms of number and size of hidden layers. You can select the POD (effect = 'rd' or 'nc'), descriptors (feat = 'rdkit', 'maccs', 'cddd' or 'morgan-512'), the number of hidden layers (layers=int) and the size of hidden layers as multiple of input size p (nodes=str or list of str, e.g. 'p//2', 'p', '2*p')
- **visualize_NN_architecture.py**: Visualize differences in crossvalidation performance of NN models with different architectures. You can select the POD (effect = 'rd' or 'nc') and descriptors (feat = 'rdkit', 'maccs', 'cddd' or 'morgan-512').
- **visualize_chem-space.py**: Visualize the two-dimensional t-SNE embedding of the marketed chemicals colored by ClassyFire Superclasses (Djoumbou et al., 2016) following von Borries et al. (2023).

## Acknowledgements
This project relies on numerous open-source software packages, and we gratefully acknowledge the open-source community for providing the tools that made this work possible. A complete list of dependencies is provided in the requirements.txt file.The conformal prediction models in this project are based on Uncertainty Aware Conformalized Quantile Regression ([UACQR](https://github.com/rrross/uacqr)) developed by Rossellini et al. 

## License
This project is covered under the **MIT License**.
