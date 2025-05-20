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
 - [2] TIDYBENCH 
```diff 
- ... (TO-DO: Update Information) ...
```
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
 2) Create a new conda environment via ```conda env create --yourRedcliffEnvironment --file=redcliff-s-icml2025.yml```
 3) Activate the new conda environment via ```conda activate yourRedcliffEnvironment```
 4) Set up your local file structure. The repository assumes you have the following folders in your system - you may change them as needed, but this will require editing numerous scripts (esp. any ending in "\_cached_args.txt" that you wish to run):
    * 
```diff 
- ... (TO-DO: Update Information) ...
```
If you intend on running comparisons between REDCLIFF-S and the supervised causal discovery algorithms from Table 2 [1], then run the preceding steps, but substitute the command in Step 2 with ```conda env create --yourRedcliffEnvironment --file=redcliff-s-icml2025-eval-env.yml```. Please note that we suggest only doing this if necessary, as the dependencies in redcliff-s-icml2025-eval-env.yml are primarily installed via pip, whereas redcliff-s-icml2025.yml places far more emphasis on conda installations.

---
## Dataset Curration and Preparation :open_file_folder:

In this section we lay out the steps for downloading/generating and preparing data files for use in this repository.

**_Synthetic Systems Datasets_**:
 1) currate the dataset(s) by running ```python3 data/currate_sVARwInnovativeContinuousGaussianNoise_data_etNL.py```
 2) 'clean'/organize the dataset(s) by running ```python3 data/clean_sVARwInnovativeContinuousGaussianNoise_data_etNL.py```
 3) re-organize data into format for supervised causal discovery task (see Table 2 of [1]) by running ```python3 data/aggregate_synthetic_systems_datasets.py```
 4) load/manage data with classes and functions from data/synthetic_datasets.py

**_D4IC Dataset(s)_**:
 1) Downlaod the original Dream4 Challenge Dataset from <https://www.synapse.org/Portal/filehandle?ownerId=syn3049712&ownerType=ENTITY&fileName=DREAM4_InSilico_Size10.zip&preview=false&wikiId=74630>, and extract contents into ```/datasets/dream4/preprocessed/size10_individual_noStateLabels```
 2) Run ```python3 data/dream4.py``` to preprocess original Dream4 Challenge data prior to D4IC data curration
 3) Run ```python3 dream4_insilicoCombo.py``` to currate the D4IC dataset(s)
 4) load/manage data with classes and functions from data/dream4_datasets.py

**_TST-100Hz Dataset(s)_**:
 1) Download the original Tail Suspension Test (TST) Dataset from <https://research.repository.duke.edu/concern/datasets/zc77sr31x?locale=en>, and extract contents into ```/public_TST_data/original_format```
 2) Run ```python3 data/tst_100HzLP.py``` to preprocess the TST dataset
 3) load/manage data with classes and functions from data/local_field_potential_datasets.py

**_Social Preference-100Hz Dataset(s)_**:
 1) As directed in the original paper [3], obtain the original Social Preference (SP) Dataset by reaching out to the lead contact, Kafui Dzirasa (kafui.dzirasa@duke.edu)
 2) Run ```python3 data/socialPreference_100HzLP.py``` to preprocess the SP dataset
 3) load/manage data with classes and functions from data/local_field_potential_datasets.py

---
## Model Training :steam_locomotive::railway_car::railway_car: 

How to train various models and search hyperparameters 
```diff 
- ... (TO-DO: Update Information) ...
```

---
## Evaluation and Results Analysis :bar_chart::chart_with_downwards_trend::chart_with_upwards_trend:

How to run analyses to evaluate models and perform experiments from original paper 
```diff 
- ... (TO-DO: Update Information) ...
```
