import torch
import numpy as np
import copy
from scipy.linalg import null_space
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress, spearmanr, rankdata, sem # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html and https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html and https://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python-handle-ties
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_auc_score



def compute_optimal_f1(labels, pred_logits):
    """
    See: 
     - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
     - https://stackoverflow.com/questions/70902917/how-to-calculate-precision-recall-and-f1-for-entity-prediction#:~:text=The%20F1%20score%20of%20a,a%20class%20in%20one%20metric.
     - https://stackoverflow.com/questions/57060907/compute-maximum-f1-score-using-precision-recall-curve
    """
    precision, recall, thresholds = precision_recall_curve(labels, pred_logits)
    precision = precision[:-1] # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
    recall = recall[:-1] # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
    
    f1_scores_by_threshold = (2.0 * precision * recall) / (precision + recall)
    for ind, f1 in enumerate(f1_scores_by_threshold):
        if not np.isfinite(f1):
            f1_scores_by_threshold[ind] = 0.
            
    opt_threshold = thresholds[np.argmax(f1_scores_by_threshold)]
    opt_f1 = np.max(f1_scores_by_threshold)
    assert np.isfinite(opt_f1)
    return opt_threshold, opt_f1


def compute_f1(labels, pred_logits, pred_cutoff):
    """
    See: 
     - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
     - https://stackoverflow.com/questions/38015181/accuracy-score-valueerror-cant-handle-mix-of-binary-and-continuous-target
    """
    preds = [int(x>pred_cutoff) for x in pred_logits]
    f1 = f1_score(labels, preds)
    return f1

def compute_true_PosNeg_and_false_PosNeg_rates(labels, preds, pred_cutoff=None):
    """see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html"""
    if pred_cutoff is not None:
        preds = [int(x>pred_cutoff) for x in preds]
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return tp, tn, fp, fn

def compute_sensitivity(labels, preds, pred_cutoff=None):
    """see sensitivity in https://www.geeksforgeeks.org/calculate-sensitivity-specificity-and-predictive-values-in-caret/"""
    tp, _, _, fn = compute_true_PosNeg_and_false_PosNeg_rates(labels, preds, pred_cutoff=pred_cutoff)
    return tp / (tp + fn)

def compute_specificity(labels, preds, pred_cutoff=None):
    """see specificity in https://www.geeksforgeeks.org/calculate-sensitivity-specificity-and-predictive-values-in-caret/"""
    _, tn, fp, _ = compute_true_PosNeg_and_false_PosNeg_rates(labels, preds, pred_cutoff=pred_cutoff)
    return tn / (tn + fp)

def compute_positive_likelihood_ratio(labels, preds, pred_cutoff=None):
    """see LR+ in https://scikit-learn.org/stable/modules/generated/sklearn.metrics.class_likelihood_ratios.html"""
    sensitivity = compute_sensitivity(labels, preds, pred_cutoff=pred_cutoff)
    specificity = compute_specificity(labels, preds, pred_cutoff=pred_cutoff)
    return sensitivity / (1. - specificity)

def compute_negative_likelihood_ratio(labels, preds, pred_cutoff=None):
    """see LR- in https://scikit-learn.org/stable/modules/generated/sklearn.metrics.class_likelihood_ratios.html"""
    sensitivity = compute_sensitivity(labels, preds, pred_cutoff=pred_cutoff)
    specificity = compute_specificity(labels, preds, pred_cutoff=pred_cutoff)
    return (1. - sensitivity) / specificity

def convert_variable_to_rank_variable(variable, method='average'): # see https://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python-handle-ties
    try:
        orig_shape = variable.shape
        return rankdata(variable, method=method).reshape(*orig_shape)
    except:
        return rankdata(variable, method=method)

def compute_covariance_betw_two_variables(X, Y):
    """
    the covariance matrix is symmetric 2x2, so the upper-right element gives cov(x,y)
    Notes:
     - see https://en.wikipedia.org/wiki/Covariance_matrix#:~:text=Any%20covariance%20matrix%20is%20symmetric,of%20each%20element%20with%20itself). 
     - as well as https://campus.datacamp.com/courses/statistical-thinking-in-python-part-1/quantitative-exploratory-data-analysis?ex=14
    """
    return np.cov(X,Y)[0,1]

def compute_spearman_numerator_cov_of_ranked_variables(X, Y):
    """
    see definition of spearman correlation coefficient at https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    """
    rX = convert_variable_to_rank_variable(X)
    rY = convert_variable_to_rank_variable(Y)
    return compute_covariance_betw_two_variables(X, Y)

# # SPECTRAL DISTANCE METRICS AND ASSOCIATED FUNCTIONALITY #########################################################################################################################################

# def get_sorted_eigenvalues(A):
#     """
#     Returns the sorted (from greatest-to-least) eigenvalues of matrix A

#     See:
#      - https://www.sciencedirect.com/science/article/pii/S0031320308000927?via%3Dihub
#      - https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html

#     """
#     return eig_vals

# DeltaCon0 AND VARIANTS #########################################################################################################################################################################

def get_k_length_path_adjacencies(A, k):
    if k == 1:
        return A
    
    Ak = None
    try:
        Ak = torch.zeros(A.size()) + A
    except:
        Ak = copy.deepcopy(A)
        
    for i in range(1,k):
        try:
            Ak = torch.matmul(Ak, A)
        except:
            Ak = np.dot(Ak, A)

    return Ak


def matsusita_distance(S1, S2):
    """
    Equation 7.3 of http://reports-archive.adm.cs.cmu.edu/anon/2015/CMU-CS-15-126.pdf
    """
    return np.sqrt(np.sum((np.sqrt(S1) - np.sqrt(S2))**2.))

def compute_node_affinity_matrix(I, D, A, eps):
    """
    see equation 7.2 of http://reports-archive.adm.cs.cmu.edu/anon/2015/CMU-CS-15-126.pdf
    """
    return np.linalg.inv(I + ((eps**2.)*D) - (eps*A))

def approximate_node_affinity_matrix_without_echo_cancellation(I, A, eps, max_path_length=None):
    """
    see Intuition 1 on page 120 of http://reports-archive.adm.cs.cmu.edu/anon/2015/CMU-CS-15-126.pdf
    """
    # ensure the requested apporximation requires path lengths between 1 and n-1
    assert A.shape[0] > 1
    if max_path_length is None:
        max_path_length = A.shape[0] - 1
    else:
        assert max_path_length < A.shape[0]
        assert max_path_length > 0
    # compute approximate affinity matrix
    S_approx = I
    Ak_adjacency_matrices = []
    for i in range(1, max_path_length+1):
        Ak = get_k_length_path_adjacencies(A, i)
        Ak_adjacency_matrices.append(Ak)
        S_approx = S_approx + (1.*(eps**i)*Ak)
    return S_approx, Ak_adjacency_matrices

def deltacon0(A1, A2, eps, make_graphs_undirected=False):
    """
    Algorithm 7.4 of http://reports-archive.adm.cs.cmu.edu/anon/2015/CMU-CS-15-126.pdf

    Note: in this implementation, we *optionally* enforce (but not by default) that A1 or A2 represent undirected graphs
    """
    G1 = copy.deepcopy(A1)
    G2 = copy.deepcopy(A2)
    assert G1.shape == G2.shape
    assert len(G1.shape) == 2
    assert G1.shape[0] == G1.shape[1]
    num_nodes = G1.shape[0]
    if make_graphs_undirected:
        for i in range(num_nodes):
            for j in range(i):
                G1[i,j] = max(G1[i,j], G1[j,i])
                G1[j,i] = max(G1[i,j], G1[j,i])
                G2[i,j] = max(G2[i,j], G2[j,i])
                G2[j,i] = max(G2[i,j], G2[j,i])
    # compute degree matrices of graphs 1 and 2
    D1 = np.diag(np.sum(G1, axis=0)) # we compute the degree from the sum of the rows; for granger causal discovery experiments (with directed graphs), this would correspond with "in-degree" of graph 1
    D2 = np.diag(np.sum(G2, axis=0))
    # compute the affinity between nodes in each graph
    S1 = compute_node_affinity_matrix(np.eye(num_nodes), D1, G1, eps)
    S2 = compute_node_affinity_matrix(np.eye(num_nodes), D2, G2, eps)
    # compute deltacon0 similarity between graphs 1 and 2
    d = matsusita_distance(S1, S2)
    return 1/(1+d)

def deltacon0_with_directed_degrees(A1, A2, eps, in_degree_coeff=1., out_degree_coeff=1.):
    """
    Variation of Algorithm 7.4 in http://reports-archive.adm.cs.cmu.edu/anon/2015/CMU-CS-15-126.pdf
    This variation takes into account the potential 'directed-ness' of A1 and A2 by computing separate "in-degree" and "out-degree" 
        matrices, and then averaging the resulting matsusita distances between the input and output 'directions'
    Note: in this implementation, we do *NOT* enforce that A1 or A2 represent undirected graphs
    """
    assert A1.shape == A2.shape
    assert len(A1.shape) == 2
    assert A1.shape[0] == A1.shape[1]
    num_nodes = A1.shape[0]
    # compute degree matrices of graphs 1 and 2
    D1in = np.diag(np.sum(A1, axis=0)) # we compute the degree from the sum of the rows; for granger causal discovery experiments (with directed graphs), this would correspond with "in-degree" of graph 1
    D2in = np.diag(np.sum(A2, axis=0)) 
    D1out = np.diag(np.sum(A1, axis=1)) # we compute the degree from the sum of the rows; for granger causal discovery experiments (with directed graphs), this would correspond with "in-degree" of graph 1
    D2out = np.diag(np.sum(A2, axis=1)) 
    # compute the affinity between nodes in each graph
    S1in = compute_node_affinity_matrix(np.eye(num_nodes), D1in, A1, eps)
    S2in = compute_node_affinity_matrix(np.eye(num_nodes), D2in, A2, eps)
    S1out = compute_node_affinity_matrix(np.eye(num_nodes), D1out, A1, eps)
    S2out = compute_node_affinity_matrix(np.eye(num_nodes), D2out, A2, eps)
    # compute deltacon0 similarity between graphs 1 and 2
    d_in = matsusita_distance(S1in, S2in)
    d_out = matsusita_distance(S1out, S2out)
    d = ((in_degree_coeff*d_in) + (out_degree_coeff*d_out)) / 2.
    return 1/(1+d)

def deltaffinity(A1, A2, eps, max_path_length=None):
    """
    Essentially DeltaCon without neighbor influence attenuation / echo cancelation (i.e. no degree matrices)
    """
    assert A1.shape == A2.shape
    assert len(A1.shape) == 2
    assert A1.shape[0] == A1.shape[1]
    num_nodes = A1.shape[0]
    if max_path_length is None:
        max_path_length = num_nodes - 1
    # compute the (approximate) affinity between nodes in each graph
    S1, _ = approximate_node_affinity_matrix_without_echo_cancellation(np.eye(num_nodes), A1, eps, max_path_length=max_path_length)
    S2, _ = approximate_node_affinity_matrix_without_echo_cancellation(np.eye(num_nodes), A2, eps, max_path_length=max_path_length)
    # compute deltaffinity similarity between graphs 1 and 2
    d = matsusita_distance(S1, S2)
    return 1/(1+d)

def path_length_mse(A1, A2, max_path_length=None):
    """
    Essentially Deltaffinity with mse directly on A^k instead of matsusita_distance on approximate S
    """     
    assert A1.shape == A2.shape
    assert len(A1.shape) == 2
    assert A1.shape[0] == A1.shape[1]
    num_nodes = A1.shape[0]
    if max_path_length is None:
        max_path_length = num_nodes - 1
    # get all adjacency matrices for k-length paths
    A1_paths = [get_k_length_path_adjacencies(A1, k) for k in range(1, max_path_length+1)]
    A2_paths = [get_k_length_path_adjacencies(A2, k) for k in range(1, max_path_length+1)]
    # compute the mse between each k-length path adjancency matrix
    mse_by_path_length = [((A1k - A2k)**2.).mean() for (A1k, A2k) in zip(A1_paths, A2_paths)]
    # the final distance is the sum across all mse terms
    return sum(mse_by_path_length), mse_by_path_length # see https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy

def path_length_mse_torch_compatible(A1, A2, max_path_length=None):
    """
    Essentially Deltaffinity with mse directly on A^k instead of matsusita_distance on approximate S
    """
    assert A1.size() == A2.size()
    assert len(A1.size()) == 2
    assert A1.size()[0] == A1.size()[1]
    num_nodes = A1.size()[0]
    if max_path_length is None:
        max_path_length = num_nodes - 1
    # get all adjacency matrices for k-length paths
    A1_paths = [get_k_length_path_adjacencies(A1, k) for k in range(1, max_path_length+1)]
    A2_paths = [get_k_length_path_adjacencies(A2, k) for k in range(1, max_path_length+1)]
    # compute the mse between each k-length path adjancency matrix
    mse_by_path_length = [((A1k - A2k)**2.).mean() for (A1k, A2k) in zip(A1_paths, A2_paths)]
    # the final distance is the sum across all mse terms
    return torch.sum(mse_by_path_length), mse_by_path_length # see https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy

##################################################################################################################################################################################################


def solve_linear_sum_assignment_between_graph_options(graph_estimates, true_graphs, cost_criteria="CosineSimilarity", inf_approximation=10000000000.):
    """
    Function for setting-up and running scipy.optimize.linear_sum_assignment # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    """
    num_workers = len(graph_estimates)
    num_jobs = len(true_graphs)
    # compute cost matrix based on criteria
    cost_matrix = np.zeros((num_workers, num_jobs))
    for w in range(num_workers):
        for j in range(num_jobs):
            curr_cost = None
            if cost_criteria == "CosineSimilarity":
                curr_cost = compute_cosine_similarity(graph_estimates[w], true_graphs[j])
            else:
                raise NotImplementedError()
            cost_matrix[w,j] += curr_cost
    print("cost_matrix == ", cost_matrix)
    finite_cost_mask = np.isfinite(cost_matrix)
    print("finite_cost_mask == ", finite_cost_mask)
    non_finite_cost_mask = 1 - finite_cost_mask
    print("non_finite_cost_mask == ", non_finite_cost_mask)
    cost_matrix[~np.isfinite(cost_matrix)] = 0.
    print("cost_matrix == ", cost_matrix)
    non_finite_cost_mask = inf_approximation*non_finite_cost_mask
    print("non_finite_cost_mask == ", non_finite_cost_mask)
    standardized_cost_matrix = cost_matrix + non_finite_cost_mask
    print("standardized_cost_matrix == ", standardized_cost_matrix)
    return linear_sum_assignment(standardized_cost_matrix)

def get_symmetric_graph_Laplacian(A):
    """
    Computes the (symmetric) graph laplacian of A+A.T
    See: 
         - wiki page on graph laplacian: https://en.wikipedia.org/wiki/Laplacian_matrix#:~:text=The%20number%20of%20connected%20components,is%20called%20the%20spectral%20gap.
         - wiki page on degree matrix: https://en.wikipedia.org/wiki/Degree_matrix
    """
    symm_A = A + A.T
    return np.diag(np.sum(symm_A, axis=1))-symm_A

def get_number_of_connected_components(A, add_self_connections=True):
    if add_self_connections:
        A = A + np.eye(A.shape[0])
    L = get_symmetric_graph_Laplacian(A)
    Z = null_space(L) # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.null_space.html
    num_comps = Z.shape[1] # see wiki page on graph laplacian: https://en.wikipedia.org/wiki/Laplacian_matrix#:~:text=The%20number%20of%20connected%20components,is%20called%20the%20spectral%20gap.
    return num_comps

def compute_cosine_similarity(A, B, epsilon=0.00000001):
    """ 
    Computes the cosine similarity between A and B 
    where A and B are np.array objects of the same size/shape
    References: see 
     - https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/
     - for addressing numerical instabilities: 
         * prompted ChatGPT, which suggested adding epsilon to the denominator
         * cross-referenced with torch.nn.functional.cosine_similarity implementation at https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html
    """
    A = A.flatten()
    B = B.flatten()
    A_norm = np.linalg.norm(A)
    B_norm = np.linalg.norm(B)
    if not np.isfinite(A_norm):
        A_norm = -1.
    if not np.isfinite(B_norm):
        B_norm = -1.
    return np.dot(A.T, B) / (max(A_norm,epsilon)*max(B_norm,epsilon))


def compute_cosine_similarity_betw_pytorch_tensors(A, B, include_diag=True):
    """ 
    Computes the cosine similarity between A and B 
    where A and B are torch.Tensor objects of the same size/shape
    References: see https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html
    """
    if not include_diag:
        I = None
        if len(A.size()) == 2:
            I = torch.eye(A.size()[0])
        elif len(A.size()) == 3:
            I = torch.zeros(A.size())
            for l in range(A.size()[2]):
                I[:,:,l] = I[:,:,l] + torch.eye(A.size()[0])
        else:
            raise NotImplementedError()
        
        device_of_A = A.get_device()
        if device_of_A != -1: # see https://pytorch.org/docs/stable/generated/torch.Tensor.get_device.html
            I = I.to('cuda', device_of_A) # see https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
            
        A = A - I
        B = B - I
        
    A = A.flatten().view(1,-1)
    B = B.flatten().view(1,-1)
    cos_sim = torch.nn.functional.cosine_similarity(A, B)
    return cos_sim


def compute_cosine_similarities_within_set_of_pytorch_tensors(tensors_to_compare, include_diag=True):
    if len(tensors_to_compare) <= 1:
        return None
    cos_sims = []
    for ind1, A in enumerate(tensors_to_compare):
        for ind2, B in enumerate(tensors_to_compare):
            if ind1 < ind2:
                cos_sims.append(compute_cosine_similarity_betw_pytorch_tensors(A, B, include_diag=include_diag))
    cos_sims = torch.Tensor(cos_sims).view(1,-1)
    return cos_sims

def compute_mse(A, B):
    # see https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy
    return ((A - B)**2).mean()

def l1_norm_difference(A_hat, A):
    l1_diff = None
    try:
        l1_diff = torch.abs(torch.norm(A_hat, 1) - torch.norm(torch.from_numpy(A), 1))
    except:
        l1_diff = torch.abs(torch.norm(torch.from_numpy(A_hat), 1) - torch.norm(torch.from_numpy(A), 1))
    return l1_diff


def get_f1_score(A_hat, A):
    """
    see https://en.wikipedia.org/wiki/F-score
    """
    # ensure inputs are in proper format
    try:
        A_hat = torch.from_numpy(A_hat)
    except:
        pass
    try:
        A = torch.from_numpy(A)
    except:
        pass

    # get masks of positive and negative values
    pos_pred_mask = 1.*(A_hat > 0.)
    neg_pred_mask = 1.*(A_hat == 0.)
    pos_label_mask = 1.*(A > 0.)
    neg_label_mask = 1.*(A == 0.)

    # compare predictions to labels
    true_positives = pos_pred_mask * pos_label_mask
    true_negatives = neg_pred_mask * neg_label_mask
    false_positives = pos_pred_mask - true_positives
    false_negatives = neg_pred_mask - true_negatives

    # compute f1 score
    precision = torch.sum(true_positives) / (torch.sum(true_positives) + torch.sum(false_positives))
    recall = torch.sum(true_positives) / (torch.sum(true_positives) + torch.sum(false_negatives))
    f1 = np.inf
    if float(precision + recall) == 0.:
        f1 = 0.
    else:
        f1 = float(2.*(precision * recall) / (precision + recall))
    return f1


class DAGNessLoss(torch.nn.Module):
    def __init__(self,):
        super(DAGNessLoss, self).__init__()
        pass
    def forward(self, W0): # see equation 4 in https://arxiv.org/pdf/2207.07908.pdf
        if len(W0.size()) == 3 and W0.size()[2] == 1:
            W0 = W0[:,:,0]
        assert len(W0.size()) == 2
        assert W0.size()[0] == W0.size()[1]
        N = W0.size()[0]
        return (torch.trace(torch.exp(W0*W0)) - N)**2.


def compare_gc_estimates_to_true_causal_graph(model, args_dict):
    GC_temporal = torch.from_numpy(args_dict["true_lagged_GC_tensor"]).cpu().data.numpy()
    GC_nontemporal = torch.from_numpy(args_dict["true_nontemporal_GC_tensor"]).cpu().data.numpy()
    # get predicted GC in various forms
    GC_noLag_est = None
    GC_lag_est = None
    GC_noLag_factor_ests = None
    GC_lag_factor_ests = None
    if "cLSTM" in args_dict["model_type"]:
        GC_noLag_factor_ests = model.GC(threshold=False, combine_wavelet_representations=True, rank_wavelets=False)
        for j,gc in enumerate(GC_noLag_factor_ests):
            if j==0:
                GC_noLag_est = gc
            else:
                GC_noLag_est = GC_noLag_est + gc
    elif "cMLP" in args_dict["model_type"]:
        GC_noLag_factor_ests = model.GC(threshold=False, ignore_lag=True, combine_wavelet_representations=True, rank_wavelets=False)
        for j,gc in enumerate(GC_noLag_factor_ests):
            if j==0:
                GC_noLag_est = gc
            else:
                GC_noLag_est = GC_noLag_est + gc
        GC_lag_factor_ests = model.GC(threshold=False, ignore_lag=False, combine_wavelet_representations=True, rank_wavelets=False)
        for j,gc in enumerate(GC_lag_factor_ests):
            if j==0:
                GC_lag_est = gc
            else:
                GC_lag_est = GC_lag_est + gc
    elif args_dict["model_type"] == "DCSFA":
        GC_noLag_factor_ests = model.GC(threshold=False, ignore_features=True)
        for j,gc in enumerate(GC_noLag_factor_ests):
            if j==0:
                GC_noLag_est = gc
            else:
                GC_noLag_est = GC_noLag_est + gc
    elif args_dict["model_type"] == "DGCNN":
        if "wvNone" in model_file:
            GC_noLag_est = model.GC(threshold=False, combine_node_feature_edges=False)
        else:
            GC_noLag_est = model.GC(threshold=False, combine_node_feature_edges=True)
    elif args_dict["model_type"] == "DYNOTEARS":
        GC_noLag_factor_ests = [model.GC()]
        for j,gc in enumerate(GC_noLag_factor_ests):
            if j==0:
                GC_noLag_est = gc
            else:
                GC_noLag_est = GC_noLag_est + gc
    else:
        raise(NotImplementedError)
    
    # perform normalization/thresholding operations on GC estimates
    if GC_noLag_est is not None:
        max_val = None
        try:
            max_val = torch.max(torch.from_numpy(GC_noLag_est))
        except:
            max_val = torch.max(GC_noLag_est)
        GC_noLag_est = 1.6*GC_noLag_est/max_val
        mask = GC_noLag_est >= 0. # max val for true noLag GC is 1.6, with 0.4 as a min true activation => 0.25 is target min val, but add wiggle room
        GC_noLag_est = GC_noLag_est * mask
    if GC_lag_est is not None:
        max_val = torch.max(GC_lag_est)
        GC_lag_est = 1.6*GC_lag_est/max_val
        mask = GC_lag_est >= 0. # max val for true lag GC is 1.6, with 0.4 as a min true activation => 0.25 is target min val, but add wiggle room
        GC_lag_est = GC_lag_est * mask
    if GC_noLag_factor_ests is not None:
        for j, gc_est in enumerate(GC_noLag_factor_ests):
            max_val = None
            try:
                max_val = torch.max(torch.from_numpy(gc_est))
            except:
                max_val = torch.max(gc_est)
            gc_est = 0.4*gc_est/max_val
            mask = gc_est >= 0.0 # only val for true noLag GC factor is 0.4, add wiggle room
            gc_est = gc_est * mask
            GC_noLag_factor_ests[j] = gc_est
    if GC_lag_factor_ests is not None:
        for j, gc_est in enumerate(GC_lag_factor_ests):
            max_val = None
            try:
                max_val = torch.max(torch.from_numpy(gc_est))
            except:
                max_val = torch.max(gc_est)
            gc_est = 0.4*gc_est/max_val
            mask = gc_est >= 0.0 # only val for true noLag GC factor is 0.4, add wiggle room
            gc_est = gc_est * mask
            GC_lag_factor_ests[j] = gc_est

    # compute mse between true and predicted GC
    GC_noLag_mse = np.inf
    GC_lag_mse = np.inf
    GC_noLag_factor_mses = [np.inf, np.inf, np.inf, np.inf]
    GC_lag_factor_mses = [np.inf, np.inf, np.inf, np.inf]

    if GC_noLag_est is not None:
        try:
            GC_noLag_mse = torch.nn.functional.mse_loss(torch.from_numpy(GC_noLag_est), torch.from_numpy(GC_nontemporal))
        except:
            GC_noLag_mse = torch.nn.functional.mse_loss(GC_noLag_est, torch.from_numpy(GC_nontemporal))
    if GC_lag_est is not None:
        smallest_lag = min(GC_lag_est.size()[2], GC_temporal.shape[2])
        GC_lag_mse = torch.nn.functional.mse_loss(GC_lag_est[:,:,:smallest_lag], torch.from_numpy(GC_temporal)[:,:,:smallest_lag])
    if GC_noLag_factor_ests is not None:
        for j, (gc_est, gc) in enumerate(zip(GC_noLag_factor_ests, args_dict["true_nontemporal_GC_tensor_factors"])):
            if j < model.num_supervised_factors:
                try:
                    GC_noLag_factor_mses[j] = torch.nn.functional.mse_loss(torch.from_numpy(gc_est), torch.from_numpy(gc))
                except:
                    try:
                        GC_noLag_factor_mses[j] = torch.nn.functional.mse_loss(gc_est, torch.from_numpy(gc))
                    except:
                        try:
                            GC_noLag_factor_mses[j] = torch.nn.functional.mse_loss(gc_est, gc)
                        except:
                            GC_noLag_factor_mses[j] = torch.nn.functional.mse_loss(torch.from_numpy(gc_est), gc)
            else:
                GC_noLag_factor_mses[j] = 0.
    if GC_lag_factor_ests is not None:
        for j, (gc_est, gc) in enumerate(zip(GC_lag_factor_ests, args_dict["true_lagged_GC_tensor_factors"])):
            if j < model.num_supervised_factors:
                smallest_lag = min(gc_est.size()[2], gc.shape[2])
                GC_lag_factor_mses[j] = torch.nn.functional.mse_loss(gc_est[:,:,:smallest_lag], torch.from_numpy(gc)[:,:,:smallest_lag])
            else:
                GC_lag_factor_mses[j] = 0.

    # compute difference betw. true GC L1 norm and predicted GC L1 norm
    GC_noLag_l1_diff = np.inf
    GC_lag_l1_diff = np.inf
    GC_noLag_factor_l1_diffs = [np.inf, np.inf, np.inf, np.inf]
    GC_lag_factor_l1_diffs = [np.inf, np.inf, np.inf, np.inf]

    if GC_noLag_est is not None:
        try:
            GC_noLag_l1_diff = torch.abs(torch.norm(GC_noLag_est, 1) - torch.norm(torch.from_numpy(GC_nontemporal), 1))
        except:
            GC_noLag_l1_diff = torch.abs(torch.norm(torch.from_numpy(GC_noLag_est), 1) - torch.norm(torch.from_numpy(GC_nontemporal), 1))
    if GC_lag_est is not None:
        GC_lag_l1_diff = torch.abs(torch.norm(GC_lag_est, 1) - torch.norm(torch.from_numpy(GC_temporal), 1))
    if GC_noLag_factor_ests is not None:
        for j, (gc_est, gc) in enumerate(zip(GC_noLag_factor_ests, args_dict["true_nontemporal_GC_tensor_factors"])):
            if j < model.num_supervised_factors:
                try:
                    GC_noLag_factor_l1_diffs[j] = torch.abs(torch.norm(gc_est, 1) - torch.norm(torch.from_numpy(gc), 1))
                except:
                    GC_noLag_factor_l1_diffs[j] = torch.abs(torch.norm(torch.from_numpy(gc_est), 1) - torch.norm(torch.from_numpy(gc), 1))
            else:
                GC_noLag_factor_l1_diffs[j] = 0.
    if GC_lag_factor_ests is not None:
        for j, (gc_est, gc) in enumerate(zip(GC_lag_factor_ests, args_dict["true_lagged_GC_tensor_factors"])):
            if j < model.num_supervised_factors:
                try:
                    GC_lag_factor_l1_diffs[j] = torch.abs(torch.norm(gc_est, 1) - torch.norm(torch.from_numpy(gc), 1))
                except:
                    GC_lag_factor_l1_diffs[j] = torch.abs(torch.norm(torch.from_numpy(gc_est), 1) - torch.norm(torch.from_numpy(gc), 1))
            else:
                GC_lag_factor_l1_diffs[j] = 0.

    # score the network and place it in rankings
    GC_noLag_score = GC_noLag_l1_diff + GC_noLag_mse
    GC_lag_score = GC_lag_l1_diff + GC_lag_mse
    GC_noLag_factor_scores = 0.
    for (l1, mse) in zip(GC_noLag_factor_l1_diffs, GC_noLag_factor_mses):
        GC_noLag_factor_scores += l1+mse
    GC_lag_factor_scores = 0.
    for (l1, mse) in zip(GC_lag_factor_l1_diffs, GC_lag_factor_mses):
        GC_lag_factor_scores += l1+mse
    return GC_noLag_score, GC_lag_score, GC_noLag_factor_scores, GC_lag_factor_scores
