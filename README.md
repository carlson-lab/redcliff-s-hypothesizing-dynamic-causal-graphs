# redcliff-s-hypothesizing-dynamic-causal-graphs <br/><br/> :wavy_dash::curly_loop::cyclone::loop::bulb::pencil:
Code corresponding to ICML 2025 paper "Generating Hypotheses of Dynamic Causal Graphs in Neuroscience: Leveraging Generative Factor Models of Observed Time Series"

We ask that any derivative works which draw significantly from this project please cite our work as: 
```diff 
- ... (TO-DO: Update Information) ...
```

**_References_** (APA format): 
 - [1] O.G. REDCLIFF-S PAPER 
```diff 
- ... (TO-DO: Update Information) ...
```
 - [2] Weichwald, S., Jakobsen, M. E., Mogensen, P. B., Petersen, L., Thams, N., & Varando, G. (2020, August). Causal structure learning from time series: Large regression coefficients may predict causal links better in practice than small p-values. In NeurIPS 2019 Competition and Demonstration Track (pp. 27-36). PMLR.
 - [3] Mague, S. D., Talbot, A., Blount, C., Walder-Christensen, K. K., Duffney, L. J., Adamson, E., ... & Dzirasa, K. (2022). Brain-wide electrical dynamics encode individual appetitive social behavior. Neuron, 110(10), 1728-1741.

**_README Overview_**: 
 - Repository Overview: describes the contents and naming convention(s) of the repository
 - Environment Setup: describes how to install dependencies and set up a local system to use the repo
 - Dataset Curration and Preparation: describes how to prepare the data used for experiments in [1]
 - Model Training: describes how to train models used in [1]
 - Evaluation and Results Analysis: describes how to analyze results and reconstruct findings reported in [1]

---
## Repository Overview :sunrise_over_mountains:

**_Modules_**:
 - data: contains scripts, functions, and class definitions for handling data used in [1]
 - evaluate: contains scripts and functions used to obtain results reported in [1]
 - models: contains class definitions for each algorithm/method used in [1]
 - general_utils: contains general purpose functions used to support modules in the repo
 - tidybench: a locally adapted copy of the original tidybench repository [2]
 - train: contains scripts for training algorithms/baselines described in [1]


**_Naming Conventions_**:
 - Data Scripts: files which end in "\_datasets.py" contain class definitions for DataSet and DataLoader objects, while all other files are used for currating and/or organizing (esp. "clean\_\*" named scripts) data files
 - Training Scripts: generally formatted as "algName_datasetName_uniqueScriptIdentifier", along with either ".py" for the actual python script or "\_cached_args.txt" to denote a file that contains the hyperparameters for the script
 - Evlauation Scripts: generally formatted as "evaluationType_datasetName_uniqueEvaluationIdentifier", along with either ".py" for the actual python script or "\_cached_args.txt" to denote a file that contains the hyperparameters for the script
    * Note that scripts for evaluating grid searches over hyperparameters break this convention, instead being named as "eval_gs_" followed by "algorithmName_datasetNameUsedInGridSearch_uniqueSearchIdentifier"

---
## Environment Setup :wrench::hammer:

To set up your local system(s) for running _most_ of the code in this repository (e.g. training algorithms defined in the 'models' module and most evaluations), complete the following steps:
 1) Set up Anaconda on your system(s)
 2) Create a new conda environment via ```conda env create --yourRedcliffEnvironment --file=redcliffs-env.yml```
 3) Activate the new conda environment via ```conda activate yourRedcliffEnvironment```
 4) Set up your local file structure. The repository assumes you have the following folders in your system - you may change them as needed, but this will require editing numerous scripts (esp. any ending in "\_cached_args.txt" that you wish to run):
    * 
```diff 
- ... (TO-DO: Update Information) ...
```
If, for whatever reason, the redcliffs-env.yml config file does not work for your local environment, alternative config files have been provided in the alternative_environments folder. We recommend using the redcliff-s-icml2025.yml configuration for training algorithms with classes in the models module. Otherwise, if you intend on running comparisons between REDCLIFF-S and the supervised causal discovery algorithms from Table 2 [1], then run the preceding steps, but substitute the command in Step 2 with ```conda env create --yourRedcliffEnvironment --file=redcliff-s-icml2025-eval-env.yml```. 

---
## Dataset Curration and Preparation :open_file_folder:

In this section we lay out the steps for downloading/generating and preparing data files for use in this repository.

**_Synthetic Systems Datasets_**:
 1) currate the dataset(s) by running ```cd data | python3 currate_sVARwInnovativeContinuousGaussianNoise_data_etNL.py```
 2) 'clean'/organize the dataset(s) by running ```cd data | python3 clean_sVARwInnovativeContinuousGaussianNoise_data_etNL.py```
 3) re-organize data into format for supervised causal discovery task (see Table 2 of [1]) by running ```cd data | python3 aggregate_synthetic_systems_datasets.py```
 4) load/manage data with classes and functions from data/synthetic_datasets.py

**_D4IC Dataset(s)_**:
 1) Downlaod the original Dream4 Challenge Dataset from <https://www.synapse.org/Portal/filehandle?ownerId=syn3049712&ownerType=ENTITY&fileName=DREAM4_InSilico_Size10.zip&preview=false&wikiId=74630>, and extract contents into ```/datasets/dream4/preprocessed/size10_individual_noStateLabels```
 2) Run ```cd data | python3 dream4.py``` to preprocess original Dream4 Challenge data prior to D4IC data curration
 3) Run ```cd data | python3 dream4_insilicoCombo.py``` to currate the D4IC dataset(s)
 4) load/manage data with classes and functions from data/dream4_datasets.py

**_TST-100Hz Dataset(s)_**:
 1) Download the original Tail Suspension Test (TST) Dataset from <https://research.repository.duke.edu/concern/datasets/zc77sr31x?locale=en>, and extract contents into ```/public_TST_data/original_format```
 2) Run ```cd data | python3 tst_100HzLP.py``` to preprocess the TST dataset
 3) load/manage data with classes and functions from data/local_field_potential_datasets.py

**_Social Preference-100Hz Dataset(s)_**:
 1) As directed in the original paper [3], obtain the original Social Preference (SP) Dataset by reaching out to the lead contact, Kafui Dzirasa (kafui.dzirasa@duke.edu)
 2) Run ```cd data | python3 socialPreference_100HzLP.py``` to preprocess the SP dataset
 3) load/manage data with classes and functions from data/local_field_potential_datasets.py

---
## Model Training :steam_locomotive::railway_car::railway_car: 

The algorithms defined in the 'models' module can be trained via ```cd train | python3 algName_dataName_uniqueExperimentIdentifier.py 2>&1 | tee logs_algName_dataName_uniqueExperimentIdentifier.out``` (substituting in an actual script name from the 'train' module). Note that the log file output is necessary for running various evaluations outlined in the next section of the README - if you do not intend to run these evaluations, it may not be necessary. 

To tune model hyperparameters, follow the pattern used in train/REDCLIFF_S_CMLP_tst100hzRerun1024AvgReg_gsSmooth1.py; the pattern can best be seen by running ```ctrl+F gen_lr``` in train/REDCLIFF_S_CMLP_tst100hzRerun1024AvgReg_gsSmooth1.py and seeing each reference to this particular hyperparameter throughout the file (similar references would need to be implemented for each hyperparameter you wish to tune in the training script for any algorithm from the 'models' module).

Note that the supervised causal discovery algorithms from Table 2 [1] were implemented later and thus do not follow the same training conventions as the algorithms in 'models'; to train these algorithms, see the instructions for running Table 2 evaluations in the next section of this README.

---
## Evaluation and Results Analysis :bar_chart::chart_with_downwards_trend::chart_with_upwards_trend:

In this section, we outline the instructions for how to run analyses to evaluate models and perform experiments from original paper [1]. Most model .fit() methods include some form of training performance tracking, so we do not cover training analytics here.

**_Comparing 'models' Algorithms on Synthetic Systems Dataset Performance_**:
 1) Run  ```cd evaluate | python3 eval_sysOptF1_crossAlg_synSysInnovGauss1030_bSCgsParsim_REDCSmo_mi300.py 2>&1 | tee logs_eval_sysOptF1_crossAlg_synSysInnovGauss1030_bSCgsParsim_REDCSmo_mi300.out```, which will run the bulk of statistical comparisons between REDCLIFF-S and other algorithms
 2) Run  ```cd evaluate | python3 summ_offDiagF1_eval_sysOptF1_crossAlg_synSysIG1030_bSCgsParsim_REDCSmo_mi300.py 2>&1 | tee logs_summ_offDiagF1_eval_sysOptF1_crossAlg_synSysIG1030_bSCgsParsim_REDCSmo_mi300.out```, which will summarize key statistics of interest from step 1
 3) Run  ```cd evaluate | python3 plotCrossExpSummaries_eval_sysOptF1_crossAlg_synSysIG1030_bSCgsParsim_REDCSmo_mi300.py 2>&1 | tee logs_plotCrossExpSummaries_eval_sysOptF1_crossAlg_synSysIG1030_bSCgsParsim_REDCSmo_mi300.out```, which will plot key statistics of interest from prior step(s)
 4) For further analyses, review the code cells/content under the "Synthetic Systems Experiment Analyses" header in evaluate/ICML2025_REDCLIFF_S_CMLP_Experiments_and_Analyses_CodeRepo_Notebook.ipynb, which should draw from files output by the previous 3 steps (e.g. 'stats_by_alg_key_dict_fold0.pkl')

**_Comparing Supervised Causal Discovery Algorithms on Synthetic System 12-11-2 Dataset Performance_**:
```diff 
- ... (TO-DO: Update Information) ...
```

**_Comparing 'models' Algorithms on D4IC Dataset Performance_**:
 1) Run  ```cd evaluate | python3 eval_sysOptF1_crossAlg_d4IC_HSNR_bCgsParsim_REDCvNEWcMLP.py 2>&1 | tee logs_eval_sysOptF1_crossAlg_d4IC_HSNR_bCgsParsim_REDCvNEWcMLP.out```, which will run the bulk of statistical comparisons between REDCLIFF-S and other algorithms on D4IC HSNR data
 2) Repeat step 1, substituting in all combinations of [HSNR, MSNR, LSNR], [bCgsParsim, bCgs1v1223,], and [REDCvNEWcMLP, REDCvOGcMLP] (note that scripts named with 'bCgsParsim' should correspond to the results reported in [1]
 3) For further analyses, review the code cells/content under the "D4IC Experiment Analyses" header (esp. following the "Computing Complexity Score of D4IC HSNR Networks 01/27/2025" sub-header) in evaluate/ICML2025_REDCLIFF_S_CMLP_Experiments_and_Analyses_CodeRepo_Notebook.ipynb, which should draw from files output by the previous 3 steps (e.g. 'stats_by_alg_key_dict_fold0.pkl')

**_Evaluating REDCLIFF-S Hyperparameter Ablations (Table 3 [1])_**:
 1) Run  ```cd evaluate | python3 eval_sysOptF1_crossAlg_d4IC_HSNR_BSCgs4ParSmo0v0104_RAbl0CosSim.py 2>&1 | tee logs_eval_sysOptF1_crossAlg_d4IC_HSNR_BSCgs4ParSmo0v0104_RAbl0CosSim.out```, which will run the bulk of statistical comparisons between the CosSim-_ablated_ REDCLIFF-S model and other algorithms on D4IC HSNR data
 2) Repeat step 1 for the eval_sysOptF1_crossAlg_d4IC_HSNR_BSCgs4ParSmo0v0104_RAbl1Fac.py, eval_sysOptF1_crossAlg_d4IC_HSNR_BSCgs4ParSmo0v0104_RAblFixFac.py, and eval_sysOptF1_crossAlg_d4IC_HSNR_BSCgs4ParSmo0v0104_RAblU.py ablation evaluation scripts
 3) For further analyses, review the code cells/content under the "D4IC Experiment Analyses" header (esp. following the "Ablation Summaries" sub-header) in evaluate/ICML2025_REDCLIFF_S_CMLP_Experiments_and_Analyses_CodeRepo_Notebook.ipynb, which should draw from files output by the previous 3 steps (esp. the logs files)

**_Evaluating Hyperparameter Grid Searches for 'models' Algorithms_**:

To determine the best hyperparameter settings for a grid search over hyperparameters of an algorithm in the 'models' module, adapt the evaluate/eval_gs_REDCLIFF_S_CMLP_SocPref100hz0326AvgReg_gsSmo1_tstParamsBSCgsSmo1_dataFULL.py script (and associated cached_args file) to match the algorithm and/or hyperparameters being analyzed. Usually, this should just mean reducing the script to just inspect 'training_loss_history' and/or 'val_loss_history' (for non-REDCLIFF-S algorithms), which can be most easily done by running ```ctrl+F factor_loss``` inside evaluate/eval_gs_REDCLIFF_S_CMLP_SocPref100hz0326AvgReg_gsSmo1_tstParamsBSCgsSmo1_dataFULL.py and substituting any references to 'factor_loss' with references to the stopping criteria (again, usually something like 'val_loss_history') and then removing references to any other 'tracked statistic' (e.g. 'roc_auc', 'deltacon0', 'mse', 'gc_cosine_sim', 'l1', etc).

**_Analyzing REDCLIFF-S Models of Real-World Data (TST and SP)_**:
```diff 
- ... (TO-DO: Update Information) ...
```

