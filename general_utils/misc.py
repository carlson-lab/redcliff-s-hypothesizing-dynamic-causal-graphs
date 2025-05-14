import os
import pickle as pkl
import numpy as np
import torch
import copy

from general_utils.metrics import compute_cosine_similarity, solve_linear_sum_assignment_between_graph_options



class TopKParser():
    def __init__(self, k_func=lambda x: 1):
        self.k_func = k_func
        pass
    def get_k_from_dataset_str(self, data_str):
        prefix_excluded_data_str = data_str.split("_numE")[1]
        num_options_str = prefix_excluded_data_str.split("_")[0]
        num_options = int(num_options_str)
        return self.k_func(num_options)
    
def apply_top_k_filter_to_edges(A, k=None):
    """
    Notes:
        see https://stackoverflow.com/questions/58070203/find-top-k-largest-item-of-a-list-in-original-order-in-python
        and https://www.w3schools.com/python/python_ref_functions.asp
    """
    if k is None:
        return A
    try:
        A = A.detach().numpy()
    except:
        pass
    original_shape = A.shape
    flat_A = A.flatten()
    top_k_values = sorted(flat_A)[-1*k:]
    top_k_A = np.array(list(map(lambda x : x if x in top_k_values else 0., flat_A))).reshape(*original_shape)
    return top_k_A
    
def normalize_numpy_array(A):
    return A/np.max(A)

def mask_diag_elements_of_square_numpy_array(orig_A):
    offDiag_A = copy.deepcopy(orig_A)
    assert len(offDiag_A.shape) == 2
    assert offDiag_A.shape[0] == offDiag_A.shape[1]
    for j in range(offDiag_A.shape[0]):
        offDiag_A[j,j] = 0.
    return offDiag_A

def place_list_elements_on_zero_to_one_scale(elements):
    min_val = np.min(elements)
    max_val = np.max(elements)
    rescaling_factor = max_val - min_val
    rescaled_elements = [1.*(x - min_val) / rescaling_factor for x in elements]
    return rescaled_elements

def obtain_factor_score_weightings_across_recording(model, recorded_signal, num_supervised_factors, num_timesteps_to_score, num_timesteps_in_input_history):
    """
    model: REDCLIFF-S
    recorded_signal: recording of shape (1, T >= num_timesteps_to_score+num_timesteps_in_input_history, num_channels)
    """
    assert recorded_signal.shape[0] == 1
    assert recorded_signal.shape[1] >= num_timesteps_to_score+num_timesteps_in_input_history
    weightings = np.zeros((num_supervised_factors, num_timesteps_to_score))
    for i in range(num_timesteps_in_input_history, num_timesteps_in_input_history+num_timesteps_to_score):
        curr_factor_weightings, _ = model.factor_score_embedder(recorded_signal[:,i-num_timesteps_in_input_history:i,:])
        weightings[:,i-num_timesteps_in_input_history] = weightings[:,i-num_timesteps_in_input_history] + curr_factor_weightings[0,:num_supervised_factors].detach().numpy()
    return weightings
    
def obtain_factor_score_classifications_across_recording(model, recorded_signal, num_supervised_factors, num_timesteps_to_score, num_timesteps_in_input_history):
    """
    model: REDCLIFF-S
    recorded_signal: recording of shape (1, T >= num_timesteps_to_score+num_timesteps_in_input_history, num_channels)
    """
    assert recorded_signal.shape[0] == 1
    assert recorded_signal.shape[1] >= num_timesteps_to_score+num_timesteps_in_input_history
    weightings = np.zeros((num_supervised_factors, num_timesteps_to_score))
    for i in range(num_timesteps_in_input_history, num_timesteps_in_input_history+num_timesteps_to_score):
        _, curr_factor_class_preds = model.factor_score_embedder(recorded_signal[:,i-num_timesteps_in_input_history:i,:])
        weightings[:,i-num_timesteps_in_input_history] = weightings[:,i-num_timesteps_in_input_history] + curr_factor_class_preds[0,:num_supervised_factors].detach().numpy()
    return weightings

def sort_unsupervised_estimates(graph_estimates, true_graphs, cost_criteria="CosineSimilarity", unsupervised_start_index=0, return_sorting_inds=False):
    matched_est_inds, matched_ground_truth_inds = solve_linear_sum_assignment_between_graph_options(graph_estimates[unsupervised_start_index:], true_graphs[unsupervised_start_index:], cost_criteria="CosineSimilarity")
    sorted_ests = [None for _ in range(len(true_graphs[unsupervised_start_index:]))]
    for (est_ind, gt_ind) in zip(matched_est_inds, matched_ground_truth_inds):
        sorted_ests[gt_ind] = graph_estimates[unsupervised_start_index:][est_ind]
    unsorted_ests = [graph_estimates[unsupervised_start_index:][i] for i in range(len(graph_estimates[unsupervised_start_index:])) if i not in matched_est_inds]
    if return_sorting_inds:
        return graph_estimates[:unsupervised_start_index]+sorted_ests+unsorted_ests, matched_est_inds, matched_ground_truth_inds
    return graph_estimates[:unsupervised_start_index]+sorted_ests+unsorted_ests

def get_avg_cosine_similarity_between_combos(list_of_elements):
    avg_cos_sim = 0.
    num_combos = 0.
    for elem_ind1, elem1 in enumerate(list_of_elements):
        for elem_ind2, elem2 in enumerate(list_of_elements):
            if elem_ind1 < elem_ind2:
                elem1 = elem1 / np.max(elem1)
                elem2 = elem2 / np.max(elem2)
                avg_cos_sim += compute_cosine_similarity(elem1, elem2)
                num_combos += 1.
    avg_cos_sim = avg_cos_sim/num_combos
    return avg_cos_sim

def get_topk_graph_mask(A, k, for_no_lag=True):
    if for_no_lag:
        A = A.sum(axis=2)
    k_largest_val = sorted(A.reshape(-1))[-1*k]
    topk_mask = A >= k_largest_val
    A_topk = topk_mask * A
    return A_topk, k_largest_val
    
def get_preds_from_masked_normalized_matrix(matrix, pred_scale, mask_thresh):
    max_val = None
    try:
        max_val = torch.max(torch.from_numpy(matrix))
    except:
        max_val = torch.max(matrix)
    matrix = matrix/max_val
    mask = matrix >= mask_thresh
    return pred_scale * matrix * mask 

def add_subdir_to_path(orig_path, subdir_name_elements):
    subdir_name = "_".join(subdir_name_elements)
    updated_dir = orig_path+os.sep+subdir_name
    if not os.path.exists(updated_dir):
        os.mkdir(updated_dir)
    return updated_dir

def flatten_GC_estimate_with_lags(GC):
    num_true_driven_chans = GC.shape[0]
    num_true_driving_chans = GC.shape[1]
    num_true_lags = GC.shape[2]
    flattened_GC = np.zeros((num_true_driven_chans, num_true_driving_chans*num_true_lags))
    for l in range(num_true_lags):
        flattened_GC[:,l*num_true_driving_chans:(l+1)*num_true_driving_chans] = GC[:,:,l]
    return flattened_GC

def unflatten_GC_estimate_with_lags(GC):
    num_true_driven_chans = GC.shape[0]
    num_true_lags = GC.shape[1] // num_true_driven_chans
    unflattened_GC = np.zeros((num_true_driven_chans, num_true_driven_chans, num_true_lags))
    for l in range(num_true_lags):
        unflattened_GC[:,:,l] = GC[:,l*num_true_driven_chans:(l+1)*num_true_driven_chans]
    return unflattened_GC

def flatten_GC_estimate_with_lags_and_gradient_tracking(GC):
    num_true_driven_chans = GC.size()[0]
    num_true_driving_chans = GC.size()[1]
    num_true_lags = GC.size()[2]
    flattened_GC = torch.zeros((num_true_driven_chans, num_true_driving_chans*num_true_lags))
    if torch.cuda.is_available():
        flattened_GC = flattened_GC.to('cuda')
    for l in range(num_true_lags):
        flattened_GC[:,l*num_true_driving_chans:(l+1)*num_true_driving_chans] = GC[:,:,l]
    return flattened_GC

def flatten_directed_spectrum_features(x):
    """
    input:
        x: (n,n,m) np array with n nodes/channels and m directed spectrum features
    returns:
        x_flat: (n,m*(2*n - 1)) np array in which row and column values corresponding to n in the original input x are expressed as rows in x_flat
    """
    assert len(x.shape) == 3
    assert x.shape[0] == x.shape[1]
    (n,_,m) = x.shape
    x_flat = np.zeros((n,m*(2*n - 1)))
    for i in range(m):
        cur_start_col = i*(2*n - 1)
        for j in range(n):
            x_flat[j,cur_start_col:cur_start_col+n] = x_flat[j,cur_start_col:cur_start_col+n] + x[j,:,i]
            x_flat[j,cur_start_col+n:cur_start_col+n+j] = x_flat[j,cur_start_col+n:cur_start_col+n+j] + x[:j,j,i]
            x_flat[j,cur_start_col+n+j:cur_start_col+(2*n-1)] = x_flat[j,cur_start_col+n+j:cur_start_col+(2*n-1)] + x[j+1:,j,i]
    return x_flat

def unflatten_directed_spectrum_features(x_flat):
    """
    input:
        x_flat: (n,m*(2*n - 1)) np array in which row and column values corresponding to n in the original input x are expressed as rows in x_flat 
    returns:
        x:  (n,n,m) np array with n nodes/channels and m directed spectrum features
    """
    assert len(x_flat.shape) == 2
    n = x_flat.shape[0]
    m = int(x_flat.shape[1]/(2*n-1))
    x = np.zeros((n,n,m))
    for i in range(m):
        cur_start_col = i*(2*n - 1)
        for j in range(n):
            x[j,:,i] = x_flat[j,cur_start_col:cur_start_col+n] + x[j,:,i]
            x[:j,j,i] = x_flat[j,cur_start_col+n:cur_start_col+n+j] + x[:j,j,i]
            x[j+1:,j,i] = x_flat[j,cur_start_col+n+j:cur_start_col+(2*n-1)] + x[j+1:,j,i]
    return x

def make_kfolds_cv_splits(data, labels, num_folds=10):
    print("general_utils.misc.make_kfolds_cv_splits: START")
    print("general_utils.misc.make_kfolds_cv_splits: len(data) == ", len(data))
    print("general_utils.misc.make_kfolds_cv_splits: len(labels) == ", len(labels))
    print("general_utils.misc.make_kfolds_cv_splits: num_folds == ", num_folds)
    assert len(data) == len(labels)
    kfolds_struct = {
        i: {"train":None, "validation":None} for i in range(num_folds)
    }
    # compute how many samples per fold
    min_num_val_samples_per_fold = len(data) // num_folds
    assert min_num_val_samples_per_fold > 0
    num_folds_with_extra_val_samp = len(data) % num_folds
    # make folds
    indices = [i for i in range(len(data))]
    for fold_id in range(num_folds):
        curr_num_val_samps = min_num_val_samples_per_fold
        if fold_id < num_folds_with_extra_val_samp:
            curr_num_val_samps += 1
        val_start_ind = fold_id*min_num_val_samples_per_fold
        kfolds_struct[fold_id]["train"] = [[data[ind], labels[ind]] for ind in indices[:val_start_ind]] + [[data[ind], labels[ind]] for ind in indices[val_start_ind+curr_num_val_samps:]]
        kfolds_struct[fold_id]["validation"] = [[data[ind], labels[ind]] for ind in indices[val_start_ind:val_start_ind+curr_num_val_samps]]
    print("general_utils.misc.make_kfolds_cv_splits: STOP")
    return kfolds_struct

def save_cv_split(train_data, val_data, cv_id, save_path):
    cv_root_path = save_path+os.sep+"fold_"+str(cv_id)
    os.mkdir(cv_root_path)
    cv_train_path = cv_root_path+os.sep+"train"
    os.mkdir(cv_train_path)
    cv_val_path = cv_root_path+os.sep+"validation"
    os.mkdir(cv_val_path)
    with open(cv_train_path+os.sep+"subset_0.pkl", "wb") as outfile:
        pkl.dump(train_data, outfile)
    with open(cv_val_path+os.sep+"subset_0.pkl", "wb") as outfile:
        pkl.dump(val_data, outfile)
    pass