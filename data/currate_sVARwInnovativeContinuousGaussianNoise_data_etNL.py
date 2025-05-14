# <><><>
import os
import json
import copy
import time
import random
import argparse
import numpy as np
from itertools import product

from data.data_utils import save_data, save_cached_args_file_for_data, generate_synthetic_data, generate_lagged_ajacency_graphs_for_factor_model

np.random.seed(9999)
random.seed(9999)



def generate_datasets_for_experiments(save_path, num_folds=5, possible_system_node_edge_factor_configs=["nodes3_edgesPerGraph1_factors2",], possible_noise_levels=[0.1,0.5,1.0], possible_noise_types=["gaussian","white","superpositional"],
                                      possible_graph_restriction_settings=[{"make_factors_orthogonal":False,"make_factors_singular_components":False},{"make_factors_orthogonal":True,"make_factors_singular_components":True}],
                                      possible_factor_linearity_settings=["Linear", "Nonlinear", "Linear_and_Nonlinear"], possible_label_types=["Oracle", "OneHot"], 
                                      nonlinear_off_diag_edge_activations=[[lambda x: x, lambda x: x], [lambda x: np.min((x,0)), lambda x: np.max((x,0))]], 
                                      num_lags=2, off_diag_edge_strengths=[0.1,1.], diag_receiving_node_forgetting_coeffs=[0.1,1.], 
                                      diag_sending_node_forgetting_coeffs=[0.9,1.], max_formulation_attempts=100, num_samples_per_file=120, 
                                      sample_recording_len=100, burnin_period=50, num_training_samples_per_class_label=1040, num_val_samples_per_class_label=240, 
                                      shuffle_seed=0):
    CONFIG_STR_NODE_LOC = 0
    CONFIG_STR_EDG_LOC = 1
    CONFIG_STR_FAC_LOC = 2
    CONFIG_STR_NODE_NAME = "nodes"
    CONFIG_STR_EDG_NAME = "edgesPerGraph"
    CONFIG_STR_FAC_NAME = "factors"
    
    print("generate_datasets_for_experiments: START")
    # create cross-product of fold_id, noise level, noise type, number of factors, and whether factor graphs are 'restricted' or not
    fold_ids = [i for i in range(num_folds)]
    FOLD_ID_IND = 0
    NOISE_LEVEL_IND = 1
    NOISE_TYPE_IND = 2
    system_node_edge_factor_configs_IND = 3
    GRAPH_RESTRICTION_SETTING_IND = 4
    FACTOR_LINEARITY_SETTINGS_IND = 5
    LABEL_TYPES_IND = 6

    parameters_to_be_parallelized = list(product(
        fold_ids,
        possible_noise_levels,
        possible_noise_types,
        possible_system_node_edge_factor_configs,
        possible_graph_restriction_settings,
        possible_factor_linearity_settings, 
        possible_label_types, 
    ))
    random.Random(shuffle_seed).shuffle(parameters_to_be_parallelized) # shuffle so that queued jobs do not favor any particular setting over another in terms of execution order

    # select data settings for current task id
    taskID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    task_param_settings = [x for x in parameters_to_be_parallelized[taskID-1]]
    time.sleep(0.1*taskID)
    
    # define additional param args now that NUM_NODES_IN_SYSTEM has been selected
    system_node_edge_factor_config = task_param_settings[system_node_edge_factor_configs_IND].split('_')
    num_nodes = int(system_node_edge_factor_config[CONFIG_STR_NODE_LOC][len(CONFIG_STR_NODE_NAME):])
    num_edges_per_graph = int(system_node_edge_factor_config[CONFIG_STR_EDG_LOC][len(CONFIG_STR_EDG_NAME):])
    num_factors = int(system_node_edge_factor_config[CONFIG_STR_FAC_LOC][len(CONFIG_STR_FAC_NAME):])
    
    num_supervised_factors = num_factors
    if num_supervised_factors > num_factors: 
        print("generate_datasets_for_experiments: STOPPING EARLY DUE TO MIS-MATCHED FACTOR-RELATED ARGS")
        return taskID
    edge_type_setting = task_param_settings[FACTOR_LINEARITY_SETTINGS_IND]
    label_type_setting = task_param_settings[LABEL_TYPES_IND]
    base_frequencies = np.pi*np.array([i*707+i%2 for i in range(num_nodes)]).reshape(num_nodes,1)/120000
    innovations_mu = np.zeros((num_nodes,1))
    innovations_var = np.ones((num_nodes,1))
    innovations_amp_coeffs = np.ones((num_nodes,1))
    
    if num_edges_per_graph < num_nodes-1 and task_param_settings[GRAPH_RESTRICTION_SETTING_IND]["make_factors_singular_components"]:
        print("generate_datasets_for_experiments: REQUESTED NUMBER OF EDGES PER GRAPH PRECLUDES THE REQUIREMENT THAT FACTORS BE SINGULAR COMPONENTS - OVERWRITING make_factors_singular_components RESTRICTION TO BE FALSE", flush=True)
        task_param_settings[GRAPH_RESTRICTION_SETTING_IND]["make_factors_singular_components"] = False
    
    # base dataset size off of number of supervised factors
    num_samples_in_train_set = num_training_samples_per_class_label*(num_supervised_factors + 1) # include samples for one extra 'UNKNOWN' factor
    num_samples_in_val_set = num_val_samples_per_class_label*(num_supervised_factors + 1) # include samples for one extra 'UNKNOWN' factor

    # prepare save_path for storing new dataset(s)
    retriction_setting_string = "o"
    if task_param_settings[GRAPH_RESTRICTION_SETTING_IND]["make_factors_orthogonal"]:
        retriction_setting_string = retriction_setting_string + "T"
    else:
        retriction_setting_string = retriction_setting_string + "F"
    retriction_setting_string = retriction_setting_string + "sc"
    if task_param_settings[GRAPH_RESTRICTION_SETTING_IND]["make_factors_singular_components"]:
        retriction_setting_string = retriction_setting_string + "T"
    else:
        retriction_setting_string = retriction_setting_string + "F"

    experiment_folder_name = "_".join([
        "numF"+str(num_factors),
        "numSF"+str(num_supervised_factors),
        "numN"+str(num_nodes),
        "numE"+str(num_edges_per_graph),
        "edges"+str(edge_type_setting),
        "labels"+str(label_type_setting), 
        "noiT-"+str(task_param_settings[NOISE_TYPE_IND]),
        "noiL-"+str(task_param_settings[NOISE_LEVEL_IND]).replace(".","-"),
        retriction_setting_string,
    ])

    save_path = save_path + os.sep + experiment_folder_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path + os.sep + "fold_"+str(task_param_settings[FOLD_ID_IND])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plot_save_path_for_training_data = save_path + os.sep + "plots_of_training_data"
    if not os.path.exists(plot_save_path_for_training_data):
        os.mkdir(plot_save_path_for_training_data)
    plot_save_path_for_validation_data = save_path + os.sep + "plots_of_validation_data"
    if not os.path.exists(plot_save_path_for_validation_data):
        os.mkdir(plot_save_path_for_validation_data)
    save_path_for_training_data = save_path + os.sep + "train"
    if not os.path.exists(save_path_for_training_data):
        os.mkdir(save_path_for_training_data)
    save_path_for_validation_data = save_path + os.sep + "validation"
    if not os.path.exists(save_path_for_validation_data):
        os.mkdir(save_path_for_validation_data)
        
    print("generate_datasets_for_experiments: save_path == ", save_path)
    print("generate_datasets_for_experiments: plot_save_path_for_training_data == ", plot_save_path_for_training_data)
    print("generate_datasets_for_experiments: plot_save_path_for_validation_data == ", plot_save_path_for_validation_data)
    print("generate_datasets_for_experiments: save_path_for_training_data == ", save_path_for_training_data)
    print("generate_datasets_for_experiments: save_path_for_validation_data == ", save_path_for_validation_data, flush=True)

    # expand nonlinear_off_diag_edge_activations to account for current number of factors
    updated_nonlinear_off_diag_edge_activations = None
    assert len(nonlinear_off_diag_edge_activations) == 2 # currently assumes (1st) half (rounding up) of nonlinear_off_diag_edge_activations are linearly activated and the other half have one pattern of nonlinear activations
    if edge_type_setting == "Linear":
        updated_nonlinear_off_diag_edge_activations = [copy.deepcopy(nonlinear_off_diag_edge_activations[0]) for _ in range(num_factors)]
    elif edge_type_setting == "Nonlinear": 
        updated_nonlinear_off_diag_edge_activations = [copy.deepcopy(nonlinear_off_diag_edge_activations[1]) for _ in range(num_factors)]
    elif edge_type_setting == "Linear_and_Nonlinear": 
        updated_nonlinear_off_diag_edge_activations = [copy.deepcopy(nonlinear_off_diag_edge_activations[0]) for _ in range(num_factors//2, num_factors)] + [copy.deepcopy(nonlinear_off_diag_edge_activations[1]) for _ in range(num_factors//2)]
    else:
        raise ValueError("generate_datasets_for_experiments: UNRECOGNIZED edge_type_setting == "+str(edge_type_setting))
    
    # obtain dynamical system adjacency tensors / edge activations
    print("generate_datasets_for_experiments: GENERATING GRAPHS FOR FOLD ", task_param_settings[FOLD_ID_IND], " DYNAMICAL SYSTEMS")
    graphs, graph_activations = generate_lagged_ajacency_graphs_for_factor_model(
        save_path, 
        num_nodes,
        num_lags,
        num_factors,
        task_param_settings[GRAPH_RESTRICTION_SETTING_IND]["make_factors_orthogonal"],
        task_param_settings[GRAPH_RESTRICTION_SETTING_IND]["make_factors_singular_components"],
        rand_seed=task_param_settings[FOLD_ID_IND]*333, # try to ensure generated graphs remain the same within a fold (across different experiment hyperparameters, e.g. noise settings)
        off_diag_edge_strengths=off_diag_edge_strengths,
        diag_receiving_node_forgetting_coeffs=diag_receiving_node_forgetting_coeffs,
        diag_sending_node_forgetting_coeffs=diag_sending_node_forgetting_coeffs,
        num_edges_per_graph=num_edges_per_graph,
        max_formulation_attempts=max_formulation_attempts,
        nonlinear_off_diag_edge_activations=updated_nonlinear_off_diag_edge_activations
    )

    # generate training set according to task id settings
    print("generate_datasets_for_experiments: GENERATING FOLD", task_param_settings[FOLD_ID_IND], " TRAINING SET", flush=True)
    train_samples = generate_synthetic_data(
        plot_save_path_for_training_data, 
        num_samples_in_train_set,
        sample_recording_len,
        label_type_setting, 
        burnin_period,
        num_nodes,
        num_factors,
        num_supervised_factors, 
        num_lags,
        graphs,
        graph_activations,
        base_frequencies,
        innovations_mu,
        innovations_var,
        innovations_amp_coeffs,
        task_param_settings[NOISE_LEVEL_IND],
        NOISE_TYPE=task_param_settings[NOISE_TYPE_IND]
    )

    # save training set
    print("generate_datasets_for_experiments: SAVING FOLD", task_param_settings[FOLD_ID_IND], " TRAINING SET", flush=True)
    save_data(save_path_for_training_data, train_samples, num_samples_in_train_set, num_samples_per_file, file_prefix="subset_")

    # generate validation set according to task id settings
    print("generate_datasets_for_experiments: GENERATING FOLD", task_param_settings[FOLD_ID_IND], " VALIDATION SET", flush=True)
    val_samples = generate_synthetic_data(
        plot_save_path_for_validation_data, 
        num_samples_in_val_set,
        sample_recording_len,
        label_type_setting, 
        burnin_period,
        num_nodes,
        num_factors,
        num_supervised_factors, 
        num_lags,
        graphs,
        graph_activations,
        base_frequencies,
        innovations_mu,
        innovations_var,
        innovations_amp_coeffs,
        task_param_settings[NOISE_LEVEL_IND],
        NOISE_TYPE=task_param_settings[NOISE_TYPE_IND]
    )

    # save validation set
    print("generate_datasets_for_experiments: SAVING FOLD", task_param_settings[FOLD_ID_IND], " VALIDATION SET", flush=True)
    save_data(save_path_for_validation_data, val_samples, num_samples_in_val_set, num_samples_per_file, file_prefix="subset_")
    
    save_cached_args_file_for_data(save_path, num_nodes, graphs, "data_fold"+str(task_param_settings[FOLD_ID_IND])+"_cached_args_sensitive.txt")

    print("generate_datasets_for_experiments: STOP")
    return taskID

#########################################################################################


if __name__=="__main__":
    parse=argparse.ArgumentParser(description='Default .pkl/.npy data generation')
    parse.add_argument(
        "-cached_args_file",
        default="currate_sVARwInnovativeContinuousGaussianNoise_data_etNL_cached_args.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    args = parse.parse_args()

    print("__MAIN__: BEGINNING DATA CURRATION")
    with open(args.cached_args_file, 'r') as infile:
        args_dict = json.load(infile)
        assert "data_save_path" in args_dict.keys()
    
    NUM_FOLDS = 5
    POSSIBLE_system_node_edge_factor_configs = [
        'nodes3_edgesPerGraph1_factors1', 'nodes3_edgesPerGraph1_factors2', 'nodes3_edgesPerGraph1_factors3', 'nodes3_edgesPerGraph1_factors4', 
        'nodes3_edgesPerGraph1_factors5', 'nodes3_edgesPerGraph2_factors1', 'nodes3_edgesPerGraph2_factors2', 'nodes6_edgesPerGraph2_factors1', 
        'nodes6_edgesPerGraph2_factors2', 'nodes6_edgesPerGraph2_factors3', 'nodes6_edgesPerGraph2_factors4', 'nodes6_edgesPerGraph2_factors5', 
        'nodes6_edgesPerGraph2_factors6', 'nodes6_edgesPerGraph2_factors7', 'nodes6_edgesPerGraph2_factors8', 'nodes6_edgesPerGraph2_factors9', 
        'nodes6_edgesPerGraph2_factors10', 'nodes6_edgesPerGraph4_factors1', 'nodes6_edgesPerGraph4_factors2', 'nodes6_edgesPerGraph4_factors3', 
        'nodes6_edgesPerGraph4_factors4', 'nodes6_edgesPerGraph4_factors5', 'nodes6_edgesPerGraph4_factors6', 'nodes6_edgesPerGraph6_factors1', 
        'nodes6_edgesPerGraph6_factors2', 'nodes6_edgesPerGraph6_factors3', 'nodes6_edgesPerGraph6_factors4', 'nodes6_edgesPerGraph8_factors1', 
        'nodes6_edgesPerGraph8_factors2', 'nodes6_edgesPerGraph8_factors3', 'nodes6_edgesPerGraph10_factors1', 'nodes6_edgesPerGraph10_factors2', 
        'nodes6_edgesPerGraph12_factors1', 'nodes6_edgesPerGraph12_factors2', 'nodes6_edgesPerGraph14_factors1', 'nodes12_edgesPerGraph11_factors1', 
        'nodes12_edgesPerGraph11_factors2', 'nodes12_edgesPerGraph11_factors3', 'nodes12_edgesPerGraph11_factors4', 'nodes12_edgesPerGraph11_factors5', 
        'nodes12_edgesPerGraph11_factors6', 'nodes12_edgesPerGraph11_factors7', 'nodes12_edgesPerGraph11_factors8', 'nodes12_edgesPerGraph11_factors9', 
        'nodes12_edgesPerGraph11_factors10', 'nodes12_edgesPerGraph12_factors1', 'nodes12_edgesPerGraph12_factors2', 'nodes12_edgesPerGraph12_factors3', 
        'nodes12_edgesPerGraph12_factors4', 'nodes12_edgesPerGraph12_factors5', 'nodes12_edgesPerGraph12_factors6', 'nodes12_edgesPerGraph12_factors7', 
        'nodes12_edgesPerGraph12_factors8', 'nodes12_edgesPerGraph12_factors9', 'nodes12_edgesPerGraph22_factors1', 'nodes12_edgesPerGraph22_factors2', 
        'nodes12_edgesPerGraph22_factors3', 'nodes12_edgesPerGraph22_factors4', 'nodes12_edgesPerGraph22_factors5', 'nodes12_edgesPerGraph33_factors1', 
        'nodes12_edgesPerGraph33_factors2', 'nodes12_edgesPerGraph33_factors3', 'nodes12_edgesPerGraph44_factors1', 'nodes12_edgesPerGraph44_factors2', 
        'nodes12_edgesPerGraph55_factors1', 'nodes12_edgesPerGraph55_factors2', 
    ]
    POSSIBLE_NOISE_LEVELS = [0.0, 1.0, 4.0, ]
    POSSIBLE_NOISE_TYPES = ["gaussian",]
    POSSIBLE_GRAPH_RESTRICTION_SETTINGS = [
        {"make_factors_orthogonal":False,"make_factors_singular_components":False}, 
    ]
    POSSIBLE_FACTOR_LINEARITY_SETTINGS = [
        "Nonlinear", 
    ]
    POSSIBLE_LABEL_TYPES = [
        "Oracle", 
        "OneHot", 
    ]
    # the above settings yield 5*66*3*2 = 1980 total jobs
    
    NONLINEAR_OFF_DIAG_ACTIVATION_FUNCS = [
        [lambda x: x, lambda x: x],  
        [lambda x: np.min((x,0)), lambda x: np.max((x,0))]
    ]
    NUM_LAG_DEPENDENCIES = 2
    OFF_DIAG_EDGE_STRENGTHS = [0.3,0.3]
    DIAG_RECEIVING_NODE_FORGETTING_COEFFS = [0.6,0.6]
    DIAG_SENDING_NODE_FORGETTING_COEFFS = [1.,1.]
    MAX_GRAPH_FORMULATION_ATTEMPTS = 100
    MAX_NUM_SAMPLES_PER_FILE = 120 # for debugging
    NUM_TIMESTEPS_IN_RECORDINGS = 100
    NUM_BURNIN_TIMESTEPS = 10
    NUM_TRAINING_SAMPLES_PER_CLASS_LABEL = 1040 # for debugging
    NUM_VAL_SAMPLES_PER_CLASS_LABEL = 240 # for debugging
    SHUFFLING_SEED_FOR_TASKS = 0
    
    taskID = generate_datasets_for_experiments(
        args_dict["data_save_path"], 
        num_folds=NUM_FOLDS, 
        possible_system_node_edge_factor_configs=POSSIBLE_system_node_edge_factor_configs, 
        possible_noise_levels=POSSIBLE_NOISE_LEVELS, 
        possible_noise_types=POSSIBLE_NOISE_TYPES,
        possible_graph_restriction_settings=POSSIBLE_GRAPH_RESTRICTION_SETTINGS,
        possible_factor_linearity_settings=POSSIBLE_FACTOR_LINEARITY_SETTINGS, 
        possible_label_types=POSSIBLE_LABEL_TYPES, 
        nonlinear_off_diag_edge_activations=NONLINEAR_OFF_DIAG_ACTIVATION_FUNCS, 
        num_lags=NUM_LAG_DEPENDENCIES, 
        off_diag_edge_strengths=OFF_DIAG_EDGE_STRENGTHS, 
        diag_receiving_node_forgetting_coeffs=DIAG_RECEIVING_NODE_FORGETTING_COEFFS,
        diag_sending_node_forgetting_coeffs=DIAG_SENDING_NODE_FORGETTING_COEFFS, 
        max_formulation_attempts=MAX_GRAPH_FORMULATION_ATTEMPTS, 
        num_samples_per_file=MAX_NUM_SAMPLES_PER_FILE, 
        sample_recording_len=NUM_TIMESTEPS_IN_RECORDINGS,
        burnin_period=NUM_BURNIN_TIMESTEPS, 
        num_training_samples_per_class_label=NUM_TRAINING_SAMPLES_PER_CLASS_LABEL, 
        num_val_samples_per_class_label=NUM_VAL_SAMPLES_PER_CLASS_LABEL, 
        shuffle_seed=SHUFFLING_SEED_FOR_TASKS
    )

    print("__MAIN__: TASK ", taskID, " ALL DONE!!!")
    pass


"""
TASK_IDs: 
['nodes3_edgesPerGraph1_factors1', 'nodes3_edgesPerGraph1_factors2', 'nodes3_edgesPerGraph1_factors3', 'nodes3_edgesPerGraph1_factors4', 'nodes3_edgesPerGraph1_factors5', 'nodes3_edgesPerGraph2_factors1', 'nodes3_edgesPerGraph2_factors2', 'nodes6_edgesPerGraph2_factors1', 'nodes6_edgesPerGraph2_factors2', 'nodes6_edgesPerGraph2_factors3', 'nodes6_edgesPerGraph2_factors4', 'nodes6_edgesPerGraph2_factors5', 'nodes6_edgesPerGraph2_factors6', 'nodes6_edgesPerGraph2_factors7', 'nodes6_edgesPerGraph2_factors8', 'nodes6_edgesPerGraph2_factors9', 'nodes6_edgesPerGraph2_factors10', 'nodes6_edgesPerGraph4_factors1', 'nodes6_edgesPerGraph4_factors2', 'nodes6_edgesPerGraph4_factors3', 'nodes6_edgesPerGraph4_factors4', 'nodes6_edgesPerGraph4_factors5', 'nodes6_edgesPerGraph4_factors6', 'nodes6_edgesPerGraph6_factors1', 'nodes6_edgesPerGraph6_factors2', 'nodes6_edgesPerGraph6_factors3', 'nodes6_edgesPerGraph6_factors4', 'nodes6_edgesPerGraph8_factors1', 'nodes6_edgesPerGraph8_factors2', 'nodes6_edgesPerGraph8_factors3', 'nodes6_edgesPerGraph10_factors1', 'nodes6_edgesPerGraph10_factors2', 'nodes6_edgesPerGraph12_factors1', 'nodes6_edgesPerGraph12_factors2', 'nodes6_edgesPerGraph14_factors1', 'nodes12_edgesPerGraph11_factors1', 'nodes12_edgesPerGraph11_factors2', 'nodes12_edgesPerGraph11_factors3', 'nodes12_edgesPerGraph11_factors4', 'nodes12_edgesPerGraph11_factors5', 'nodes12_edgesPerGraph11_factors6', 'nodes12_edgesPerGraph11_factors7', 'nodes12_edgesPerGraph11_factors8', 'nodes12_edgesPerGraph11_factors9', 'nodes12_edgesPerGraph11_factors10', 'nodes12_edgesPerGraph12_factors1', 'nodes12_edgesPerGraph12_factors2', 'nodes12_edgesPerGraph12_factors3', 'nodes12_edgesPerGraph12_factors4', 'nodes12_edgesPerGraph12_factors5', 'nodes12_edgesPerGraph12_factors6', 'nodes12_edgesPerGraph12_factors7', 'nodes12_edgesPerGraph12_factors8', 'nodes12_edgesPerGraph12_factors9', 'nodes12_edgesPerGraph22_factors1', 'nodes12_edgesPerGraph22_factors2', 'nodes12_edgesPerGraph22_factors3', 'nodes12_edgesPerGraph22_factors4', 'nodes12_edgesPerGraph22_factors5', 'nodes12_edgesPerGraph33_factors1', 'nodes12_edgesPerGraph33_factors2', 'nodes12_edgesPerGraph33_factors3', 'nodes12_edgesPerGraph44_factors1', 'nodes12_edgesPerGraph44_factors2', 'nodes12_edgesPerGraph55_factors1', 'nodes12_edgesPerGraph55_factors2', 'nodes24_edgesPerGraph24_factors1', 'nodes24_edgesPerGraph24_factors2', 'nodes24_edgesPerGraph24_factors3', 'nodes24_edgesPerGraph24_factors4', 'nodes24_edgesPerGraph24_factors5', 'nodes24_edgesPerGraph24_factors6', 'nodes24_edgesPerGraph24_factors7', 'nodes24_edgesPerGraph24_factors8', 'nodes24_edgesPerGraph24_factors9', 'nodes24_edgesPerGraph24_factors10', 'nodes24_edgesPerGraph46_factors1', 'nodes24_edgesPerGraph46_factors2', 'nodes24_edgesPerGraph46_factors3', 'nodes24_edgesPerGraph46_factors4', 'nodes24_edgesPerGraph46_factors5', 'nodes24_edgesPerGraph46_factors6', 'nodes24_edgesPerGraph46_factors7', 'nodes24_edgesPerGraph46_factors8', 'nodes24_edgesPerGraph46_factors9', 'nodes24_edgesPerGraph46_factors10', 'nodes24_edgesPerGraph92_factors1', 'nodes24_edgesPerGraph92_factors2', 'nodes24_edgesPerGraph92_factors3', 'nodes24_edgesPerGraph92_factors4', 'nodes24_edgesPerGraph92_factors5', 'nodes24_edgesPerGraph138_factors1', 'nodes24_edgesPerGraph138_factors2', 'nodes24_edgesPerGraph138_factors3', 'nodes24_edgesPerGraph184_factors1', 'nodes24_edgesPerGraph184_factors2', 'nodes24_edgesPerGraph230_factors1', 'nodes24_edgesPerGraph230_factors2', 'nodes36_edgesPerGraph36_factors1', 'nodes36_edgesPerGraph36_factors2', 'nodes36_edgesPerGraph36_factors3', 'nodes36_edgesPerGraph36_factors4', 'nodes36_edgesPerGraph36_factors5', 'nodes36_edgesPerGraph36_factors6', 'nodes36_edgesPerGraph36_factors7', 'nodes36_edgesPerGraph36_factors8', 'nodes36_edgesPerGraph36_factors9', 'nodes36_edgesPerGraph36_factors10', 'nodes36_edgesPerGraph105_factors1', 'nodes36_edgesPerGraph105_factors2', 'nodes36_edgesPerGraph105_factors3', 'nodes36_edgesPerGraph105_factors4', 'nodes36_edgesPerGraph105_factors5', 'nodes36_edgesPerGraph105_factors6', 'nodes36_edgesPerGraph105_factors7', 'nodes36_edgesPerGraph105_factors8', 'nodes36_edgesPerGraph105_factors9', 'nodes36_edgesPerGraph105_factors10', 'nodes36_edgesPerGraph210_factors1', 'nodes36_edgesPerGraph210_factors2', 'nodes36_edgesPerGraph210_factors3', 'nodes36_edgesPerGraph210_factors4', 'nodes36_edgesPerGraph210_factors5', 'nodes36_edgesPerGraph315_factors1', 'nodes36_edgesPerGraph315_factors2', 'nodes36_edgesPerGraph315_factors3', 'nodes36_edgesPerGraph420_factors1', 'nodes36_edgesPerGraph420_factors2', 'nodes36_edgesPerGraph525_factors1', 'nodes36_edgesPerGraph525_factors2']

# CODE FOR GENERATING TASK_IDs: ####################################################################################################################
import numpy as np

f = lambda N: (N**2. - N)/(2.*N) # formula for computing max num of directed edges per node in a graph from nodes
POSSIBLE_num_nodes = [3,6,12,24,36,]#[3,4,5,6,7,8,9,10,]
print("POSSIBLE_num_nodes == ", POSSIBLE_num_nodes)
POSSIBLE_num_edges_per_graph = [[i for i in range(1,int(x*f(x))) if i%(max(1,int(x*f(x))//6))==0 or i == x] for x in POSSIBLE_num_nodes] # [x*f(x) for x in POSSIBLE_num_nodes]
print("POSSIBLE_num_edges_per_graph == ", POSSIBLE_num_edges_per_graph)
concat_pnepg_list = []
for nEdge_list in POSSIBLE_num_edges_per_graph:
    concat_pnepg_list = concat_pnepg_list + nEdge_list  
print("len(concat_pnepg_list) == ", len(concat_pnepg_list))

MAX_NUM_FACTORS_TO_MODEL = 10
POSSIBLE_num_orthogonal_factors = []
for num_Nodes,possible_edgesPGraph in zip(POSSIBLE_num_nodes,POSSIBLE_num_edges_per_graph):
    curr_graph_size_factor_settings = []
    for num_edges in possible_edgesPGraph:
        curr_edge_config_factor_settings = []
        for coverage_level in [0.1,0.3,0.5,0.7,0.9,]: # 1.0 corresponds to fully connnected / undirected graph,
            for num_factors in range(1,MAX_NUM_FACTORS_TO_MODEL+1):
                if num_factors*num_edges <= coverage_level*(num_Nodes**2. - num_Nodes) and num_factors not in curr_edge_config_factor_settings:
                    curr_edge_config_factor_settings.append(num_factors)
        curr_graph_size_factor_settings.append(curr_edge_config_factor_settings)
    POSSIBLE_num_orthogonal_factors.append(curr_graph_size_factor_settings)
print("POSSIBLE_num_orthogonal_factors == ", POSSIBLE_num_orthogonal_factors)
print("len(POSSIBLE_num_orthogonal_factors) == ", len(POSSIBLE_num_orthogonal_factors))
concat_pnof_list = []
for n_edges_list in POSSIBLE_num_orthogonal_factors:
    for nFactors_list in n_edges_list:
        concat_pnof_list = concat_pnof_list + nFactors_list  
print("concat_pnof_list == ", concat_pnof_list)
print("len(concat_pnof_list) == ", len(concat_pnof_list))

print("\n TASK_IDS \n")
all_task_ids = []
for i,num_nodes_in_graph in enumerate(POSSIBLE_num_nodes):
    for j,num_edges_in_graph in enumerate(POSSIBLE_num_edges_per_graph[i]):
        for num_factors in POSSIBLE_num_orthogonal_factors[i][j]:
            all_task_ids.append("_".join(["nodes"+str(num_nodes_in_graph),"edgesPerGraph"+str(num_edges_in_graph),"factors"+str(num_factors),]))
print("len(all_task_ids) == ", len(all_task_ids))
print("all_task_ids == ", all_task_ids)
"""