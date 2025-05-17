# redcliff-s-hypothesizing-dynamic-causal-graphs <br/><br/> :wavy_dash::curly_loop::cyclone::loop::bulb::pencil:
Code corresponding to ICML 2025 paper "Generating Hypotheses of Dynamic Causal Graphs in Neuroscience: Leveraging Generative Factor Models of Observed Time Series"

We ask that any derivative works which draw significantly from this project please cite our work as: diff - ... (TO-DO: Update Information) ...

**_References_**: 
 - [1] O.G. REDCLIFF-S PAPER <span style="color:red">... (TO-DO: Update Information) ...</span>
 - [2] TIDYBENCH <span style="color:red">... (TO-DO: Update Information) ...</span>

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

To set up your local system(s) for running the code in this repository, complete the following steps:
 1) Set up Anaconda on your system(s)
 2) Create a new conda environment via ```conda env create --yourRedcliffEnvironment --file=redcliff-s-icml2025.yml```
 3) Activate the new conda environment via ```conda activate yourRedcliffEnvironment```
 4) Set up your local file structure. The repository assumes you have the following folders in your system - you may change them as needed, but this will require editing numerous scripts (esp. any ending in "\_cached_args.txt" that you wish to run):
    * <span style="color:red">... (TO-DO: Update Information) ...</span>

---
## Dataset Curration and Preparation :open_file_folder:

How to download/generate and prepare data files <span style="color:red">... (TO-DO: Update Information) ...</span>

**_Synthetic Systems Datasets_**:
 1) currate the dataset(s) by running ```python3 data/currate_sVARwInnovativeContinuousGaussianNoise_data_etNL.py```
 2) 'clean'/organize the dataset(s) by running ```python3 data/clean_sVARwInnovativeContinuousGaussianNoise_data_etNL.py```
 3) re-organize data into format for supervised causal discovery task (see Table 2 of [1]) by running ```python3 data/aggregate_synthetic_systems_datasets.py```
 4) load/manage with classes and functions from ```python3 data/synthetic_datasets.py```

**_D4IC Dataset(s)_**:
 1) Downlaod the original Dream4 Challenge Dataset from <https://www.synapse.org/Portal/filehandle?ownerId=syn3049712&ownerType=ENTITY&fileName=DREAM4_InSilico_Size10.zip&preview=false&wikiId=74630>, and extract contents into ```/datasets/dream4/preprocessed/size10_individual_noStateLabels```
 2) <span style="color:red">... (TO-DO: Update Information) ...</span>

**_TST-100Hz Dataset(s)_**:
 1) <span style="color:red">... (TO-DO: Update Information) ...</span>

**_Social Preference-100Hz Dataset(s)_**:
 1) <span style="color:red">... (TO-DO: Update Information) ...</span>

---
## Model Training :steam_locomotive::railway_car::railway_car: 

How to train various models and search hyperparameters <span style="color:red">... (TO-DO: Update Information) ...</span>

---
## Evaluation and Results Analysis :bar_chart::chart_with_downwards_trend::chart_with_upwards_trend:

How to run analyses to evaluate models and perform experiments from original paper <span style="color:red">... (TO-DO: Update Information) ...</span>
