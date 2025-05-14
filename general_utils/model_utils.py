import torch
import copy
import os
import pickle as pkl
import numpy as np

from sklearn.metrics import roc_auc_score

from data.synthetic_datasets import load_normalized_synthetic_wVAR_data_train_test_split, load_normalized_synthetic_wVAR_data_train_test_split_as_matrices
from data.local_field_potential_datasets import load_normalized_lfp_data_train_test_split, load_normalized_lfp_data_train_test_split_as_matrices
from data.dream4_datasets import load_normalized_DREAM4_data_train_test_split, load_normalized_DREAM4_data_train_test_split_as_matrices, load_normalized_DREAM4_data_train_test_split_as_tensors
from general_utils.misc import get_avg_cosine_similarity_between_combos, sort_unsupervised_estimates
from general_utils.metrics import get_f1_score, deltacon0, deltacon0_with_directed_degrees, deltaffinity, path_length_mse, compute_cosine_similarity




def track_receiver_operating_characteristic_stats_for_redcliff_models(GC, CURR_GC_EST, f1score_histories, roc_auc_histories, remove_self_connections=False):
    for thresh_key in f1score_histories.keys():
        sample_counter = 0.
        running_f1_scores = []
        running_roc_aucs = []

        for s, curr_gc_est in enumerate(CURR_GC_EST):
            for i, gc_est in enumerate(curr_gc_est[:len(GC)]):
                
                # prep true gc graph for comparisson
                curr_true_gc = np.sum(GC[i], axis=2)
                if remove_self_connections:
                    assert len(curr_true_gc.shape) == 2
                    assert curr_true_gc.shape[0] == curr_true_gc.shape[1]
                    for j in range(curr_true_gc.shape[0]):
                        curr_true_gc[j,j] = 0.
                        
                if np.max(curr_true_gc) != 0.: # necessary when true_gc is not known - in these cases, it may be estimated to be the identity and/or zero matrix
                    curr_true_gc = curr_true_gc / np.max(curr_true_gc)
                
                # prep estimated gc graph for comparisson
                curr_est = gc_est.detach().cpu().numpy()
                if len(curr_est.shape) == 3:
                    curr_est = np.sum(curr_est, axis=2)
                if remove_self_connections:
                    assert len(curr_est.shape) == 2
                    assert curr_est.shape[0] == curr_est.shape[1]
                    for j in range(curr_est.shape[0]):
                        curr_est[j,j] = 0.
                        
                if np.max(curr_est) != 0.: # avoid a divide by zero error
                    curr_est = curr_est / np.max(curr_est)
                mask = curr_est > thresh_key
                curr_est = curr_est * mask
                
                # set up labels for roc-auc analysis
                roc_auc_labels = [int(l) for l in curr_true_gc.flatten()]
                
                if s == 0:
                    running_f1_scores.append(get_f1_score(curr_est, curr_true_gc))
                    if np.sum(roc_auc_labels) == 0:
                        running_roc_aucs.append(0.5)# see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
                    else:
                        running_roc_aucs.append(roc_auc_score(roc_auc_labels, curr_est.flatten()))# see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
                else:
                    running_f1_scores[i] += get_f1_score(curr_est, curr_true_gc)
                    if np.sum(roc_auc_labels) == 0:
                        running_roc_aucs[i] += 0.5 # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
                    else:
                        running_roc_aucs[i] += roc_auc_score(roc_auc_labels, curr_est.flatten())# see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
                    
            sample_counter += 1.
            
        if len(f1score_histories[thresh_key]) != len(running_f1_scores): # this can happen when 'primary_gc_est_mode' is embedder-dependant, such as in 'fixed_embedder_exclusive'
            print("")
            if len(running_f1_scores) == 1 and len(f1score_histories[thresh_key]) > 1: # case where model is only capable of producing a single GC estimate for multiple supervised system states
                for i, _ in enumerate(f1score_histories[thresh_key]):
                    f1score_histories[thresh_key][i].append(running_f1_scores[0]/sample_counter)
                    roc_auc_histories[thresh_key][i].append(running_roc_aucs[0]/sample_counter)
            else: # case where the model may be generating more GC estimates than there are supervised system states - in this case, we assume the supervised system states are ordered at the front of all GC-related lists (both true and estimated)
                assert len(f1score_histories[thresh_key]) < len(running_f1_scores)
                for i, _ in enumerate(f1score_histories[thresh_key]):
                    f1score_histories[thresh_key][i].append(running_f1_scores[i]/sample_counter)
                    roc_auc_histories[thresh_key][i].append(running_roc_aucs[i]/sample_counter)
        else:                                                            # the standard case, where there is one gc estimate per supervised class
            for i, _ in enumerate(f1score_histories[thresh_key]):
                f1score_histories[thresh_key][i].append(running_f1_scores[i]/sample_counter)
                roc_auc_histories[thresh_key][i].append(running_roc_aucs[i]/sample_counter)
                
    return f1score_histories, roc_auc_histories


def track_deltacon0_related_stats_for_redcliff_models(GC, CURR_GC_EST, num_chans, deltacon0_histories, deltacon0_with_directed_degrees_histories, 
                                                      deltaffinity_histories, path_length_mse_histories, deltaConEps=0.1, in_degree_coeff=1., 
                                                      out_degree_coeff=1., remove_self_connections=False):
    sample_counter = 0.
    running_deltacon0_scores = []
    running_deltacon0_with_directed_degrees_scores = []
    running_deltaffinity_scores = []
    running_path_length_mse_scores = dict()
    for s, curr_gc_est in enumerate(CURR_GC_EST):
        for i, gc_est in enumerate(curr_gc_est[:len(GC)]):
            
            curr_true_gc = np.sum(GC[i], axis=2)
            if remove_self_connections:
                assert len(curr_true_gc.shape) == 2
                assert curr_true_gc.shape[0] == curr_true_gc.shape[1]
                for j in range(curr_true_gc.shape[0]):
                    curr_true_gc[j,j] = 0.
                    
            if np.max(curr_true_gc) != 0.: # necessary when true_gc is not known - in these cases, it may be estimated to be the identity and/or zero matrix
                curr_true_gc = curr_true_gc / np.max(curr_true_gc)
            
            curr_est = gc_est.detach().cpu().numpy()
            if len(curr_est.shape) == 3:
                curr_est = np.sum(curr_est, axis=2)
                if remove_self_connections:
                    assert len(curr_est.shape) == 2
                    assert curr_est.shape[0] == curr_est.shape[1]
                    for j in range(curr_est.shape[0]):
                        curr_est[j,j] = 0.
                        
            if np.max(curr_true_gc) != 0.: # avoid a divide by zero error
                curr_est = curr_est / np.max(curr_est)
            
            _, curr_path_length_mses = path_length_mse(curr_true_gc, curr_est, max_path_length=None)
            if s == 0:
                running_deltacon0_scores.append(deltacon0(curr_true_gc, curr_est, deltaConEps))
                running_deltacon0_with_directed_degrees_scores.append(deltacon0_with_directed_degrees(curr_true_gc, curr_est, deltaConEps, in_degree_coeff=in_degree_coeff, out_degree_coeff=out_degree_coeff))
                running_deltaffinity_scores.append(deltaffinity(curr_true_gc, curr_est, deltaConEps, max_path_length=None))
                for (path_length, mse) in zip(range(1,num_chans), curr_path_length_mses):
                    if path_length not in running_path_length_mse_scores.keys():
                        running_path_length_mse_scores[path_length] = [0. for _ in range(len(curr_gc_est))]
                    running_path_length_mse_scores[path_length][i] += mse
            else:
                running_deltacon0_scores[i] += deltacon0(curr_true_gc, curr_est, deltaConEps)
                running_deltacon0_with_directed_degrees_scores[i] += deltacon0_with_directed_degrees(curr_true_gc, curr_est, deltaConEps, in_degree_coeff=in_degree_coeff, out_degree_coeff=out_degree_coeff)
                running_deltaffinity_scores[i] += deltaffinity(curr_true_gc, curr_est, deltaConEps, max_path_length=None)
                for (path_length, mse) in zip(range(1,num_chans), curr_path_length_mses):
                    running_path_length_mse_scores[path_length][i] += mse
        sample_counter += 1.
        
    if len(deltacon0_histories) != len(running_deltacon0_scores): # this can happen when 'primary_gc_est_mode' is embedder-dependant, such as in 'fixed_embedder_exclusive'
        if len(running_deltacon0_scores) == 1 and len(deltacon0_histories) > 1: # case where model is only capable of producing a single GC estimate for multiple supervised system states
            for i, _ in enumerate(deltacon0_histories):
                deltacon0_histories[i].append(running_deltacon0_scores[0]/sample_counter)
                deltacon0_with_directed_degrees_histories[i].append(running_deltacon0_with_directed_degrees_scores[0]/sample_counter)
                deltaffinity_histories[i].append(running_deltaffinity_scores[0]/sample_counter)
        else: # case where the model may be generating more GC estimates than there are supervised system states - in this case, we assume the supervised system states are ordered at the front of all GC-related lists (both true and estimated)
            assert len(deltacon0_histories) < len(running_deltacon0_scores)
            for i, _ in enumerate(deltacon0_histories):
                deltacon0_histories[i].append(running_deltacon0_scores[i]/sample_counter)
                deltacon0_with_directed_degrees_histories[i].append(running_deltacon0_with_directed_degrees_scores[i]/sample_counter)
                deltaffinity_histories[i].append(running_deltaffinity_scores[i]/sample_counter)
    else:                                                            # the standard case, where there is one gc estimate per supervised class
        for i, _ in enumerate(deltacon0_histories):
            deltacon0_histories[i].append(running_deltacon0_scores[i]/sample_counter)
            deltacon0_with_directed_degrees_histories[i].append(running_deltacon0_with_directed_degrees_scores[i]/sample_counter)
            deltaffinity_histories[i].append(running_deltaffinity_scores[i]/sample_counter)
            for path_length in running_path_length_mse_scores.keys():
                path_length_mse_histories[path_length][i].append(running_path_length_mse_scores[path_length][i]/sample_counter)
            
    return deltacon0_histories, deltacon0_with_directed_degrees_histories, deltaffinity_histories, path_length_mse_histories


def track_l1_norm_stats_of_gc_ests_from_redcliff_models(CURR_GC_EST, gc_factor_l1_loss_histories):
    running_l1_losses = []
    sample_counter = 0.
    for s, curr_gc_est in enumerate(CURR_GC_EST):
        for est_num, gc_est in enumerate(curr_gc_est):
            curr_norm = None
            try:
                gc_est = gc_est / np.max(gc_est)
                curr_norm = torch.norm(torch.from_numpy(gc_est), 1)
            except:
                gc_est = gc_est / torch.max(gc_est)
                curr_norm = torch.norm(gc_est, 1)

            if s == 0:
                running_l1_losses.append(curr_norm.detach().cpu().numpy())
            else:
                running_l1_losses[est_num] += curr_norm.detach().cpu().numpy()
        sample_counter += 1.
    
    running_l1_losses = [x / sample_counter for x in running_l1_losses]
    for i in range(len(gc_factor_l1_loss_histories)):
        gc_factor_l1_loss_histories[i].append(running_l1_losses[i])
        
    curr_l1_loss = sum(running_l1_losses)
                
    return curr_l1_loss, gc_factor_l1_loss_histories


def track_cosine_similarity_stats_of_gc_ests_from_redcliff_models(CURR_GC_EST, cosine_sim_histories, label_offset=0):
    curr_cos_sims = dict()
    sample_counter = 0.
    for s, curr_gc_est in enumerate(CURR_GC_EST):
        for est_ind1, gc_est1 in enumerate(curr_gc_est):
            for est_ind2, gc_est2 in enumerate(curr_gc_est):
                if est_ind1 < est_ind2:
                    gc_est1 = gc_est1 / np.max(gc_est1)
                    gc_est2 = gc_est2 / np.max(gc_est2)
                    if s == 0:
                        curr_cos_sims[str(est_ind1+label_offset)+"and"+str(est_ind2+label_offset)] = compute_cosine_similarity(gc_est1, gc_est2)
                    else:
                        curr_cos_sims[str(est_ind1+label_offset)+"and"+str(est_ind2+label_offset)] += compute_cosine_similarity(gc_est1, gc_est2)
        sample_counter += 1.
    
    for key in curr_cos_sims.keys():
        cosine_sim_histories[key].append(curr_cos_sims[key]/sample_counter)
        
    return cosine_sim_histories


def apply_mlp_prox_penalty(W, hidden, p, lag, lam, lr, penalty):
    if penalty == 'GL':
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam))) * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL':
        norm = torch.norm(W, dim=0, keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam))) * torch.clamp(norm - (lr * lam), min=0.0))
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam))) * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        for i in range(lag):
            norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            W.data[:, :, :(i+1)] = ((W.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam))) * torch.clamp(norm - (lr * lam), min=0.0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)
    return W


def prox_update(network, lam, lr, model_type="cLSTM", penalty=None):
    '''Perform in place proximal update on first layer weight matrix.'''
    if model_type == "cLSTM":
        W = network.lstm.weight_ih_l0
        norm = torch.norm(W, dim=0, keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lam * lr))) * torch.clamp(norm - (lr * lam), min=0.0))
        network.lstm.flatten_parameters()
    elif model_type == "cMLP":
        assert penalty is not None
        W = network.layers[0].weight
        hidden, p, lag = W.shape
        network.layers[0].weight = apply_mlp_prox_penalty(W, hidden, p, lag, lam, lr, penalty)
    elif model_type == "BidirectionalCMLP":
        assert penalty is not None
        W = network.fwd_layers[0].weight
        hidden, p, lag = W.shape
        network.fwd_layers[0].weight = apply_mlp_prox_penalty(W, hidden, p, lag, lam, lr, penalty)
    elif model_type == "cMLP_MultiLag":
        assert penalty is not None
        for lag_ind in range(len(network.layers[0].layers)):
            W = network.layers[0].layers[lag_ind].weight
            hidden, p, lag = W.shape
            network.layers[0].layers[lag_ind].weight = apply_mlp_prox_penalty(W, hidden, p, lag, lam, lr, penalty)
    else:
        raise NotImplementedError("Unrecognized model_type=="+str(model_type))


def apply_mlp_regularize_penalty(W, hidden, p, lag, lam, penalty):
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(W, dim=(0, 2)))
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(W, dim=(0, 2))) + torch.sum(torch.norm(W, dim=0)))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2))) for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def regularize(network, lam, model_type="cLSTM", penalty=None):
    '''Calculate regularization term for first layer weight matrix.'''
    if model_type == "cLSTM":
        W = network.lstm.weight_ih_l0
        return lam * torch.sum(torch.norm(W, dim=0))
    elif model_type == "cMLP":
        W = network.layers[0].weight
        hidden, p, lag = W.shape
        return apply_mlp_regularize_penalty(W, hidden, p, lag, lam, penalty)
    elif model_type == "BidirectionalCMLP":
        W = network.fwd_layers[0].weight
        hidden, p, lag = W.shape
        return apply_mlp_regularize_penalty(W, hidden, p, lag, lam, penalty)
    elif model_type == "cMLP_MultiLag":
        penalty_sum = 0.
        for lag_ind in range(len(network.layers[0].layers)):
            W = network.layers[0].layers[lag_ind].weight
            hidden, p, lag = W.shape
            penalty_sum += apply_mlp_regularize_penalty(W, hidden, p, lag, lam, penalty)
        return penalty_sum
    else:
        raise NotImplementedError("Unrecognized model_type=="+str(model_type))


def ridge_regularize(network, lam, model_type="cLSTM"):
    '''Apply ridge penalty at linear layer and hidden-hidden weights.'''
    if model_type == "cLSTM":
        return lam * (torch.sum(network.linear.weight ** 2) + torch.sum(network.lstm.weight_hh_l0 ** 2))
    elif "MLP" in model_type:
        if "Bidirectional" in model_type:
            fwd_sqrd_weights = [torch.sum(fc.weight ** 2) for fc in network.fwd_layers[1:]]
            rev_sqrd_weights = [torch.sum(fc.weight ** 2) for fc in network.rev_layers[1:]]
            return lam * sum(fwd_sqrd_weights+rev_sqrd_weights)
        else:
            return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])
    else:
        raise NotImplementedError("Unrecognized model_type=="+str(model_type))


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params
    pass


def generate_signal_from_sequential_factor_model(model, x0, sim_steps):
    batch_size = x0.size()[0]
    context_size = x0.size()[1]
    num_channels = x0.size()[2]
    x_sim = torch.zeros(batch_size, sim_steps, num_channels)
    if torch.cuda.is_available():
        x_sim = x_sim.to(device="cuda")
    hidden=None
    x0_hist = [x0]
            
    for t in range(sim_steps):
        x0 = x0_hist[-1]
        combined_pred, _, hidden, _ = model(x0, hidden)
        x_sim[:,t,:] = combined_pred[:,0,:]
        new_x0 = torch.zeros(x0.size()).to(x0.device)
        new_x0[:,:context_size-1,:] = x0[:,1:,:]
        new_x0[:,context_size-1,:] = x_sim[:,t,:]
        x0_hist.append(new_x0)

    return x_sim


def create_model_instance(args_dict, employ_version_with_smoothing_loss=False):
    from models.clstm_fm import cLSTM_FM
    from models.cmlp_fm import cMLP_FM
    from models.redcliff_s_clstm import REDCLIFF_S_CLSTM
    from models.redcliff_s_cmlp import REDCLIFF_S_CMLP
    from models.redcliff_s_cmlp_withStateSmoothing import REDCLIFF_S_CMLP_withStateSmoothing
    from models.redcliff_s_dgcnn import REDCLIFF_S_DGCNN
    from models.dcsfa_nmf import FullDCSFAModel
    from models.dcsfa_nmf_vanillaDirSpec import FullDCSFAModel as FullDCSFAModel_GCv2
    from models.dgcnn import DGCNN_Model
    from models.dynotears import DYNOTEARS_Model
    from models.dynotears_vanilla import DYNOTEARS_Model as DYNOTEARS_Model_V0
    from models.navar import NAVAR, NAVARLSTM

    model = None
    if "cMLP" in args_dict["model_type"] or ("CMLP" in args_dict["model_type"] and "REDCLIFF" in args_dict["model_type"]):
        if "REDCLIFF" in args_dict["model_type"]:
            if args_dict["X_train"] is not None:
                print("create_model_instance: args_dict['data_set_name'] == ", args_dict['data_set_name'])
                print("create_model_instance: original args_dict['num_supervised_factors'] == ", args_dict['num_supervised_factors'])
                _, y0 = next(iter(args_dict["X_train"])) # see https://stackoverflow.com/questions/53570732/get-single-random-example-from-pytorch-dataloader
                print("create_model_instance: y0.size()[ == ", y0.size())
                old_n_sup_factors = args_dict["num_supervised_factors"]
                args_dict["num_supervised_factors"] = min(y0.size()[1], old_n_sup_factors)
                if old_n_sup_factors != args_dict["num_supervised_factors"]:
                    print("create_model_instance: WARNING! cMLP_FM WILL ONLY HAVE ", args_dict["num_supervised_factors"], " SUPERVISED NETWORKS DUE TO NUMBER OF AVAILABLE LABELS IN DATASET")
                old_n_factors = args_dict["num_factors"]
                args_dict["num_factors"] = max(args_dict["num_supervised_factors"], old_n_factors)
                if old_n_factors != args_dict["num_factors"]:
                    print("create_model_instance: WARNING! cMLP_FM WILL HAVE ", args_dict["num_factors"], " NETWORKS DUE TO NUMBER OF SUPERVISED NETWORKS", flush=True)
            if "_S_" in args_dict["model_type"]:
                if not employ_version_with_smoothing_loss:
                    model = REDCLIFF_S_CMLP(
                        args_dict["num_channels"], 
                        args_dict["gen_lag"], 
                        args_dict["gen_hidden"], 
                        args_dict["embed_lag"], 
                        args_dict["embed_hidden_sizes"], 
                        args_dict["input_length"], 
                        args_dict["output_length"], 
                        args_dict["num_factors"], 
                        args_dict["num_supervised_factors"], 
                        args_dict["coeff_dict"], 
                        args_dict["use_sigmoid_restriction"], 
                        args_dict["factor_score_embedder_type"], 
                        args_dict["factor_score_embedder_args"], 
                        args_dict["primary_gc_est_mode"], 
                        args_dict["forward_pass_mode"], 
                        num_sims=args_dict["num_sims"], 
                        wavelet_level=args_dict["wavelet_level"], 
                        save_path=args_dict["save_path"], 
                        training_mode=args_dict["training_mode"], 
                        num_pretrain_epochs=args_dict["num_pretrain_epochs"], 
                        num_acclimation_epochs=args_dict["num_acclimation_epochs"]
                    ).float()
                else:
                    model = REDCLIFF_S_CMLP_withStateSmoothing(
                        args_dict["num_channels"], 
                        args_dict["gen_lag"], 
                        args_dict["gen_hidden"], 
                        args_dict["embed_lag"], 
                        args_dict["embed_hidden_sizes"], 
                        args_dict["input_length"], 
                        args_dict["output_length"], 
                        args_dict["num_factors"], 
                        args_dict["num_supervised_factors"], 
                        args_dict["coeff_dict"], 
                        args_dict["use_sigmoid_restriction"], 
                        args_dict["factor_score_embedder_type"], 
                        args_dict["factor_score_embedder_args"], 
                        args_dict["primary_gc_est_mode"], 
                        args_dict["forward_pass_mode"], 
                        num_sims=args_dict["num_sims"], 
                        wavelet_level=args_dict["wavelet_level"], 
                        save_path=args_dict["save_path"], 
                        training_mode=args_dict["training_mode"], 
                        num_pretrain_epochs=args_dict["num_pretrain_epochs"], 
                        num_acclimation_epochs=args_dict["num_acclimation_epochs"], 
                        STATE_SCORE_SMOOTHING_EPSILON=0.0001
                    ).float()
            else:
                raise NotImplementedError()
        else:
            print("model_utils.create_model_instance: cMLP_FM will be receiving the following args:")
            print("model_utils.create_model_instance: \t args_dict['num_channels'] == ", args_dict['num_channels'])
            print("model_utils.create_model_instance: \t args_dict['gen_lag'] == ", args_dict['gen_lag'])
            print("model_utils.create_model_instance: \t args_dict['gen_hidden'] == ", args_dict['gen_hidden'])
            print("model_utils.create_model_instance: \t args_dict['embed_hidden_sizes'] == ", args_dict['embed_hidden_sizes'])
            print("model_utils.create_model_instance: \t args_dict['input_length'] == ", args_dict['input_length'])
            print("model_utils.create_model_instance: \t args_dict['coeff_dict'] == ", args_dict['coeff_dict'])
            print("model_utils.create_model_instance: \t args_dict['num_sims'] == ", args_dict['num_sims'])
            print("model_utils.create_model_instance: \t args_dict['wavelet_level'] == ", args_dict['wavelet_level'])
            print("model_utils.create_model_instance: \t args_dict['save_path'] == ", args_dict['save_path'])
            model = cMLP_FM(
                args_dict["num_channels"], 
                args_dict["gen_lag"], 
                args_dict["gen_hidden"], 
                args_dict["embed_hidden_sizes"], 
                args_dict["input_length"], 
                args_dict["output_length"], 
                args_dict["coeff_dict"], 
                num_sims=args_dict["num_sims"], 
                wavelet_level=args_dict["wavelet_level"], 
                save_path=args_dict["save_path"]
            ).float()
        if torch.cuda.is_available():
            model = model.cuda(device="cuda")

    elif "cLSTM" in args_dict["model_type"] or ("CLSTM" in args_dict["model_type"] and "REDCLIFF" in args_dict["model_type"]):
        if "REDCLIFF" in args_dict["model_type"]:
            if "_S_" in args_dict["model_type"]:
                model = REDCLIFF_S_CLSTM(
                    args_dict["num_channels"], 
                    args_dict["gen_hidden"], 
                    args_dict["embed_hidden_sizes"], 
                    args_dict["embed_lag"], 
                    args_dict["num_factors"], 
                    args_dict["num_supervised_factors"], 
                    args_dict["coeff_dict"], 
                    args_dict["use_sigmoid_restriction"], 
                    args_dict["factor_score_embedder_type"], 
                    args_dict["factor_score_embedder_args"], 
                    args_dict["primary_gc_est_mode"], 
                    args_dict["forward_pass_mode"], 
                    num_sims=args_dict["num_sims"], 
                    wavelet_level=args_dict["wavelet_level"], 
                    save_path=args_dict["save_path"], 
                    training_mode=args_dict["training_mode"], 
                    num_pretrain_epochs=args_dict["num_pretrain_epochs"], 
                    num_acclimation_epochs=args_dict["num_acclimation_epochs"]
                ).float()
            else:
                raise NotImplementedError()
        else:
            model = cLSTM_FM(
                args_dict["num_channels"], 
                args_dict["gen_hidden"], 
                args_dict["embed_hidden_sizes"], 
                args_dict["context"], 
                args_dict["coeff_dict"], 
                num_sims=args_dict["num_sims"], 
                wavelet_level=args_dict["wavelet_level"], 
                save_path=args_dict["save_path"]
            ).float()
        if torch.cuda.is_available():
            model = model.cuda(device="cuda")

    elif "DCSFA" in args_dict["model_type"]:
        y0 = args_dict["y_train"][0,:]
        old_n_sup_nets = args_dict["n_sup_networks"]
        args_dict["n_sup_networks"] = min(len(y0), old_n_sup_nets)
        if args_dict["n_sup_networks"] == 1: # case where there is only one label and so supervision (esp. in terms of ROC-AUC) is impossible
            args_dict["n_sup_networks"] = 0
            sup_weight=args_dict["sup_weight"] = 0
            sup_recon_weight=args_dict["sup_recon_weight"] = 0
        elif old_n_sup_nets != args_dict["n_sup_networks"]:
            print("create_model_instance: DCSFA WILL ONLY HAVE ", args_dict["n_sup_networks"], " SUPERVISED NETWORKS DUE TO NUMBER OF AVAILABLE LABELS IN DATASET")
        if "vanilla" in args_dict["signal_format"]:
            model = FullDCSFAModel_GCv2(
                args_dict["num_channels"], 
                args_dict["num_high_level_node_features"], 
                args_dict["n_components"],
                args_dict["n_sup_networks"],
                args_dict["h"],
                args_dict["save_path"],
                momentum=args_dict["momentum"],
                lr=args_dict["lr"],
                device="auto",
                n_intercepts=1,
                optim_name="AdamW",
                recon_loss="MSE",
                recon_weight=args_dict["recon_weight"],
                sup_weight=args_dict["sup_weight"],
                sup_recon_weight=args_dict["sup_recon_weight"],
                use_deep_encoder=True,
                sup_recon_type="Residual",
                feature_groups=None,
                group_weights=None,
                fixed_corr=None,
                sup_smoothness_weight=args_dict["sup_smoothness_weight"],
                verbose=False,
            )
        else:
            model = FullDCSFAModel(
                args_dict["num_channels"], 
                args_dict["num_high_level_node_features"], 
                args_dict["n_components"],
                args_dict["n_sup_networks"],
                args_dict["h"],
                args_dict["save_path"],
                momentum=args_dict["momentum"],
                lr=args_dict["lr"],
                device="auto",
                n_intercepts=1,
                optim_name="AdamW",
                recon_loss="MSE",
                recon_weight=args_dict["recon_weight"],
                sup_weight=args_dict["sup_weight"],
                sup_recon_weight=args_dict["sup_recon_weight"],
                use_deep_encoder=True,
                sup_recon_type="Residual",
                feature_groups=None,
                group_weights=None,
                fixed_corr=None,
                sup_smoothness_weight=args_dict["sup_smoothness_weight"],
                verbose=False,
            )

    elif "DGCNN" in args_dict["model_type"] or ("DGCNN" in args_dict["model_type"] and "REDCLIFF" in args_dict["model_type"]):
        if "REDCLIFF" in args_dict["model_type"]:
            if "_S_" in args_dict["model_type"]:
                model = REDCLIFF_S_DGCNN(
                    args_dict["num_channels"], 
                    args_dict["num_graph_conv_layers"], 
                    args_dict["num_hidden_nodes"], 
                    args_dict["embed_hidden_sizes"], 
                    args_dict["gen_num_features_per_node"], 
                    args_dict["embed_num_features_per_node"], 
                    args_dict["num_factors"], 
                    args_dict["num_supervised_factors"], 
                    args_dict["coeff_dict"],  
                    args_dict["use_sigmoid_restriction"], 
                    args_dict["factor_score_embedder_type"], 
                    args_dict["factor_score_embedder_args"], 
                    args_dict["primary_gc_est_mode"], 
                    args_dict["forward_pass_mode"], 
                    num_sims=args_dict["num_sims"], 
                    wavelet_level=args_dict["wavelet_level"], 
                    save_path=args_dict["save_path"],  
                    training_mode=args_dict["training_mode"], 
                    num_pretrain_epochs=args_dict["num_pretrain_epochs"], 
                    num_acclimation_epochs=args_dict["num_acclimation_epochs"]
                ).float()
            else:
                raise NotImplementedError()
        else:
            model = DGCNN_Model(
                args_dict["num_channels"], 
                args_dict["num_wavelets_per_chan"], 
                args_dict["num_features_per_node"], 
                args_dict["num_graph_conv_layers"], 
                args_dict["num_hidden_nodes"], 
                args_dict["num_classes"]
            ).float()
        if torch.cuda.is_available():
            model = model.cuda(device="cuda")

    elif "DYNOTEARS" in args_dict["model_type"]:
        if "Vanilla" not in args_dict["model_type"]:
            model = DYNOTEARS_Model(
                lambda_w=args_dict["lambda_w"],
                lambda_a=args_dict["lambda_a"],
                max_iter=args_dict["max_iter"],
                h_tol=args_dict["h_tol"],
                w_threshold = args_dict["w_threshold"],
                tabu_edges=args_dict["tabu_edges"],
                tabu_parent_nodes=args_dict["tabu_parent_nodes"],
                tabu_child_nodes=args_dict["tabu_child_nodes"], 
                grad_step=args_dict["grad_step"], 
                wa_est=args_dict["wa_est"], 
                rho=args_dict["rho"], 
                alpha=args_dict["alpha"], 
                h_value=args_dict["h_value"], 
                h_new=args_dict["h_new"]
            )
        else:
            model = DYNOTEARS_Model_V0(
                lambda_w=args_dict["lambda_w"],
                lambda_a=args_dict["lambda_a"],
                max_iter=args_dict["max_iter"],
                h_tol=args_dict["h_tol"],
                w_threshold = args_dict["w_threshold"],
                tabu_edges=args_dict["tabu_edges"],
                tabu_parent_nodes=args_dict["tabu_parent_nodes"],
                tabu_child_nodes=args_dict["tabu_child_nodes"]
            )
    
    elif "NAVAR" in args_dict["model_type"]:
        if "MLP" in args_dict["model_type"]:
            model = NAVAR(
                args_dict["num_nodes"], 
                args_dict["num_hidden"], 
                args_dict["maxlags"], 
                hidden_layers=args_dict["hidden_layers"], 
                dropout=args_dict["dropout"]
            )
        elif "LSTM" in args_dict["model_type"]:
            model = NAVARLSTM(
                args_dict["num_nodes"], 
                args_dict["num_hidden"], 
                args_dict["maxlags"], 
                hidden_layers=args_dict["hidden_layers"], 
                dropout=args_dict["dropout"]
            )
        else:
            raise NotImplementedError()
        
    else:
        raise ValueError("read_in_data_args: model_type == "+str(args_dict["model_type"])+" IS NOT SUPPORTED (CURRENTLY)")

    return model


def get_data_for_model_training(args_dict, grid_search=True, dataset_category="synthetic_wVAR", average_region_map=None, average_label_over_time_steps=True, pad_X=None):
    data = None
    if "_S_" in args_dict["model_type"] or "cMLP" in args_dict["model_type"] or "cLSTM" in args_dict["model_type"] or "DGCNN" in args_dict["model_type"] or ("DYNOTEARS" in args_dict["model_type"] and "Vanilla" not in args_dict["model_type"]):
        X_train = None
        X_val = None
        if dataset_category == "synthetic_wVAR":
            X_train, X_val = load_normalized_synthetic_wVAR_data_train_test_split(
                args_dict["data_root_path"], 
                args_dict["batch_size"], 
                signal_format=args_dict["signal_format"], 
                shuffle=True, 
                shuffle_seed=0, 
                train_portion=0.8, 
                grid_search=grid_search
            )
        elif dataset_category == "local_field_potential":
            X_train, X_val = load_normalized_lfp_data_train_test_split(
                args_dict["data_root_path"], 
                args_dict["batch_size"], 
                signal_format=args_dict["signal_format"], 
                shuffle=True, 
                shuffle_seed=0, 
                train_portion=0.8, 
                grid_search=grid_search, 
                average_region_map=average_region_map
            )
        elif dataset_category == "DREAM4":
            X_train, X_val = load_normalized_DREAM4_data_train_test_split(
                args_dict["data_root_path"], 
                args_dict["batch_size"], 
                signal_format=args_dict["signal_format"], 
                shuffle=True, 
                shuffle_seed=0, 
                train_portion=0.8, 
                grid_search=grid_search
            )
        else:
            raise NotImplementedError("general_utils.model_utils.get_data_for_model_training: UNRECOGNIZED dataset_category == ", dataset_category)
        data = [X_train, None, X_val, None]

    elif "DCSFA" in args_dict["model_type"]:
        X_train = None
        y_train = None
        X_val = None
        y_val = None
        if dataset_category == "synthetic_wVAR":
            X_train, y_train, X_val, y_val = load_normalized_synthetic_wVAR_data_train_test_split_as_matrices(
                args_dict["data_root_path"], 
                signal_format=args_dict["signal_format"], 
                shuffle=True, 
                shuffle_seed=0, 
                max_num_features_per_series=args_dict["num_node_features"], 
                train_portion=0.8, 
                dirspec_params=args_dict["dirspec_params"], 
                grid_search=grid_search, 
                average_label_over_time_steps=average_label_over_time_steps, #True
                pad_X=pad_X
            )
        elif dataset_category == "local_field_potential":
            X_train, y_train, X_val, y_val = load_normalized_lfp_data_train_test_split_as_matrices(
                args_dict["data_root_path"], 
                signal_format=args_dict["signal_format"], 
                shuffle=True, 
                shuffle_seed=0, 
                max_num_features_per_series=args_dict["num_node_features"], 
                train_portion=0.8, 
                dirspec_params=args_dict["dirspec_params"], 
                grid_search=grid_search, 
                average_region_map=average_region_map
            )
        elif dataset_category == "DREAM4":
            X_train, y_train, X_val, y_val = load_normalized_DREAM4_data_train_test_split_as_matrices(
                args_dict["data_root_path"], 
                signal_format=args_dict["signal_format"], 
                shuffle=True, 
                shuffle_seed=0, 
                max_num_features_per_series=args_dict["num_node_features"], 
                train_portion=0.8, 
                dirspec_params=args_dict["dirspec_params"], 
                grid_search=grid_search
            )
        else:
            raise NotImplementedError("general_utils.model_utils.get_data_for_model_training: UNRECOGNIZED dataset_category == ", dataset_category)
        data = [X_train, y_train, X_val, y_val]

    elif ("DYNOTEARS" in args_dict["model_type"] and "Vanilla" in args_dict["model_type"]) or "NAVAR" in args_dict["model_type"]:
        assert dataset_category == "DREAM4"
        X_train, y_train, X_val, y_val = load_normalized_DREAM4_data_train_test_split_as_tensors(
            args_dict["data_root_path"], 
            signal_format=args_dict["signal_format"], 
            shuffle=True, 
            shuffle_seed=0, 
            max_num_features_per_series=None, 
            train_portion=0.8, 
            dirspec_params=None, 
            grid_search=grid_search
        )
        data = [X_train, y_train, X_val, y_val]
    else:
        raise ValueError("read_in_data_args: model_type == "+str(args_dict["model_type"])+" IS NOT SUPPORTED (CURRENTLY)")

    return data


def call_model_fit_method(model, args_dict):
    if "cMLP" in args_dict["model_type"] or ("CMLP" in args_dict["model_type"] and "REDCLIFF" in args_dict["model_type"]):
        GEN_BETA_VALS = (0.9, 0.999)
        if "_S_" in args_dict["model_type"]:
            optimizerA = torch.optim.Adam(
                model.gen_model[0].parameters(), 
                lr=args_dict["embed_lr"], 
                betas=GEN_BETA_VALS, 
                eps=args_dict['embed_eps'], 
                weight_decay=args_dict['embed_weight_decay']
            )
            optimizerB = torch.optim.Adam(
                model.gen_model[1].parameters(), 
                lr=args_dict["gen_lr"], 
                betas=GEN_BETA_VALS, 
                eps=args_dict['gen_eps'], 
                weight_decay=args_dict['gen_weight_decay']
            )
            _ = model.fit(
                args_dict['save_path'], 
                args_dict['X_train'], 
                optimizerA, 
                optimizerB, 
                args_dict["input_length"], 
                args_dict["output_length"],
                args_dict["num_sims"], 
                args_dict["max_iter"],
                args_dict['X_val'], 
                lookback=args_dict["lookback"], 
                check_every=args_dict["check_every"], 
                verbose=args_dict["verbose"], 
                GC=args_dict["true_GC_factors"], 
                deltaConEps=args_dict["deltaConEps"], #0.1, 
                in_degree_coeff=args_dict["in_degree_coeff"], #1., 
                out_degree_coeff=args_dict["out_degree_coeff"], #1., 
                prior_factors_path=args_dict["prior_factors_path"], #None, 
                cost_criteria=args_dict["cost_criteria"], #"CosineSimilarity", 
                unsupervised_start_index=args_dict["unsupervised_start_index"], #0, 
                max_factor_prior_batches=args_dict["max_factor_prior_batches"], #10, 
                stopping_criteria_forecast_coeff=args_dict["stopping_criteria_forecast_coeff"], #1., 
                stopping_criteria_factor_coeff=args_dict["stopping_criteria_factor_coeff"], #1., 
                stopping_criteria_cosSim_coeff=args_dict["stopping_criteria_cosSim_coeff"] #1.
            )
        else:
            optimizerA = torch.optim.Adam(
                model.gen_model.parameters(), 
                lr=args_dict["gen_lr"], 
                betas=GEN_BETA_VALS, 
                eps=args_dict['gen_eps'], 
                weight_decay=args_dict['gen_weight_decay']
            )
            _ = model.fit(
                args_dict['save_path'], 
                args_dict['X_train'], 
                optimizerA, 
                args_dict["input_length"], 
                args_dict["output_length"], 
                args_dict["num_sims"], 
                args_dict["max_iter"], 
                lookback=args_dict["lookback"], 
                check_every=args_dict["check_every"], 
                verbose=args_dict["verbose"], 
                GC=args_dict["true_GC_tensor"], 
                X_val=args_dict['X_val']
            )
        
    elif "cLSTM" in args_dict["model_type"] or ("CLSTM" in args_dict["model_type"] and "REDCLIFF" in args_dict["model_type"]):
        GEN_BETA_VALS = (0.9, 0.999)
        if "_S_" in args_dict["model_type"]:
            optimizerA = torch.optim.Adam(
                model.gen_model[0].parameters(), 
                lr=args_dict["embed_lr"], 
                betas=GEN_BETA_VALS, 
                eps=args_dict['embed_eps'], 
                weight_decay=args_dict['embed_weight_decay']
            )
            optimizerB = torch.optim.Adam(
                model.gen_model[1].parameters(), 
                lr=args_dict["gen_lr"], 
                betas=GEN_BETA_VALS, 
                eps=args_dict['gen_eps'], 
                weight_decay=args_dict['gen_weight_decay']
            )
            _ = model.fit(
                args_dict['save_path'], 
                args_dict['X_train'], 
                optimizerA, 
                optimizerB, 
                args_dict["context"], 
                args_dict["max_input_length"],
                args_dict["num_sims"], 
                args_dict["max_iter"],
                args_dict['X_val'], 
                lookback=args_dict["lookback"], 
                check_every=args_dict["check_every"], 
                verbose=args_dict["verbose"], 
                GC=args_dict["true_GC_factors"], 
                deltaConEps=args_dict["deltaConEps"], #0.1, 
                in_degree_coeff=args_dict["in_degree_coeff"], #1., 
                out_degree_coeff=args_dict["out_degree_coeff"], #1., 
                prior_factors_path=args_dict["prior_factors_path"], #None, 
                cost_criteria=args_dict["cost_criteria"], #"CosineSimilarity", 
                unsupervised_start_index=args_dict["unsupervised_start_index"], #0, 
                max_factor_prior_batches=args_dict["max_factor_prior_batches"], #10, 
                stopping_criteria_forecast_coeff=args_dict["stopping_criteria_forecast_coeff"], #1., 
                stopping_criteria_factor_coeff=args_dict["stopping_criteria_factor_coeff"], #1., 
                stopping_criteria_cosSim_coeff=args_dict["stopping_criteria_cosSim_coeff"] #1.
            )
        else:
            optimizerA = torch.optim.Adam(
                model.gen_model.parameters(), 
                lr=args_dict["gen_lr"], 
                betas=GEN_BETA_VALS, 
                eps=args_dict['gen_eps'], 
                weight_decay=args_dict['gen_weight_decay']
            )
            _ = model.fit(
                args_dict['save_path'], 
                args_dict['X_train'], 
                optimizerA, 
                args_dict["context"], 
                args_dict["max_input_length"], 
                args_dict["num_sims"], 
                args_dict["max_iter"], 
                lookback=args_dict["lookback"], 
                check_every=args_dict["check_every"], 
                verbose=args_dict["verbose"], 
                GC=args_dict["true_GC_tensor"], 
                X_val=args_dict['X_val']
            )

    elif "DCSFA" in args_dict["model_type"]:
        pretrain = True
        if model.n_sup_networks < 1:
            print("model_utils.call_model_fit_method: WARNING - there are fewer than 1 supervised network(s), so the model pretrain argument is being set to False")
            pretrain = False
        model.fit(
            args_dict["X_train"],
            args_dict["y_train"],
            y_pred_weights=None,
            task_mask=None,
            intercept_mask=None,
            y_sample_groups=None,
            n_epochs=args_dict["n_epochs"],
            n_pre_epochs=args_dict["n_pre_epochs"],
            nmf_max_iter=args_dict["nmf_max_iter"],
            batch_size=args_dict["batch_size"],
            lr=args_dict["lr"],
            pretrain=pretrain,
            verbose=False,
            X_val=args_dict["X_val"],
            y_val=args_dict["y_val"],
            y_pred_weights_val=None,
            task_mask_val=None,
            best_model_name=args_dict["best_model_name"],
        )
        _ = model.evaluate(
            args_dict["X_val"], 
            args_dict["y_val"], 
            args_dict["true_GC_tensor"], 
            args_dict["save_path"], 
            threshold=False, 
            ignore_features=True
        )
        
    elif "DGCNN" in args_dict["model_type"] or ("DGCNN" in args_dict["model_type"] and "REDCLIFF" in args_dict["model_type"]):
        GEN_BETA_VALS = (0.9, 0.999)
        if "_S_" in args_dict["model_type"]:
            optimizerA = torch.optim.Adam(
                model.gen_model[0].parameters(), 
                lr=args_dict["embed_lr"], 
                betas=GEN_BETA_VALS, 
                eps=args_dict['embed_eps'], 
                weight_decay=args_dict['embed_weight_decay']
            )
            optimizerB = torch.optim.Adam(
                model.gen_model[1].parameters(), 
                lr=args_dict["gen_lr"], 
                betas=GEN_BETA_VALS, 
                eps=args_dict['gen_eps'], 
                weight_decay=args_dict['gen_weight_decay']
            )
            _ = model.fit(
                args_dict['save_path'], 
                args_dict['X_train'], 
                optimizerA, 
                optimizerB, 
                args_dict["num_sims"], 
                args_dict["max_iter"],
                args_dict['X_val'], 
                lookback=args_dict["lookback"], 
                check_every=args_dict["check_every"], 
                verbose=args_dict["verbose"], 
                GC=args_dict["true_GC_factors"], 
                deltaConEps=args_dict["deltaConEps"], #0.1, 
                in_degree_coeff=args_dict["in_degree_coeff"], #1., 
                out_degree_coeff=args_dict["out_degree_coeff"], #1., 
                prior_factors_path=args_dict["prior_factors_path"], #None, 
                cost_criteria=args_dict["cost_criteria"], #"CosineSimilarity", 
                unsupervised_start_index=args_dict["unsupervised_start_index"], #0, 
                max_factor_prior_batches=args_dict["max_factor_prior_batches"], #10, 
                stopping_criteria_forecast_coeff=args_dict["stopping_criteria_forecast_coeff"], #1., 
                stopping_criteria_factor_coeff=args_dict["stopping_criteria_factor_coeff"], #1., 
                stopping_criteria_cosSim_coeff=args_dict["stopping_criteria_cosSim_coeff"] #1.
            )
        else:
            optimizerA = torch.optim.Adam(
                model.dgcnn.parameters(), 
                lr=args_dict["gen_lr"], 
                betas=GEN_BETA_VALS, 
                eps=args_dict['gen_eps'], 
                weight_decay=args_dict['gen_weight_decay']
            )
            if "REDCLIFF" in args_dict["model_type"]:
                _ = model.fit(
                    args_dict['save_path'], 
                    args_dict['X_train'], 
                    optimizerA, 
                    args_dict["num_sims"], 
                    args_dict["max_iter"], 
                    lookback=args_dict["lookback"], 
                    check_every=args_dict["check_every"], 
                    verbose=args_dict["verbose"], 
                    GC=args_dict["true_GC_tensor"], 
                    X_val=args_dict['X_val']
                )
            else:
                _ = model.fit(
                    args_dict['save_path'], 
                    args_dict["X_train"], 
                    optimizerA, 
                    args_dict["max_iter"],
                    lookback=args_dict["lookback"], 
                    check_every=args_dict["check_every"], 
                    verbose=args_dict["verbose"], 
                    GC=args_dict["true_GC_tensor"], 
                    val_loader=args_dict["X_val"]
                )

    elif "DYNOTEARS" in args_dict["model_type"]:
        if "Vanilla" not in args_dict["model_type"]:
            model.fit(
                args_dict['save_root_path'], 
                args_dict["max_data_iter"], 
                args_dict["X_train"], 
                args_dict["X_val"], 
                iter_start=args_dict["iter_start"],
                lag_size=args_dict["lag_size"], 
                num_iters_prior_to_stop=args_dict["num_iters_prior_to_stop"],
                reuse_rho=args_dict["reuse_rho"],
                reuse_alpha=args_dict["reuse_alpha"],
                reuse_h_val=args_dict["reuse_h_val"],
                reuse_h_new=args_dict["reuse_h_new"],
                GC_orig=args_dict["true_GC_tensor"],
                check_every=args_dict["check_every"]
            )
        else:
            model.fit(
                args_dict['save_root_path'], 
                args_dict["X_train"], 
                args_dict["X_val"], 
                lag_size=args_dict["lag_size"], 
                GC_orig=args_dict["true_GC_tensor"],
                save_a_est=True
            )

    elif "NAVAR" in args_dict["model_type"]:
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args_dict["learning_rate"], 
            weight_decay=args_dict["weight_decay"]
        )

        if "MLP" in args_dict["model_type"]:
            _ = model.fit(
                args_dict["save_path"], 
                args_dict["X_train"], 
                args_dict["y_train"], 
                criterion, 
                optimizer, 
                X_val=args_dict["X_val"],#None, 
                Y_val=args_dict["y_val"],#,None, 
                val_proportion=args_dict["val_proportion"],#0.0, 
                epochs=args_dict["epochs"],#,200, 
                batch_size=args_dict["batch_size"],#,300, 
                split_timeseries=args_dict["split_timeseries"],#,False, 
                check_every=args_dict["check_every"],#1000
                lambda1=args_dict["lambda1"]
            )
        elif "LSTM" in args_dict["model_type"]:
            _ = model.fit(
                args_dict["save_path"], 
                args_dict["X_train"], 
                args_dict["y_train"], 
                criterion, 
                optimizer, 
                X_val=args_dict["X_val"],#None, 
                Y_val=args_dict["y_val"],#,None, 
                val_proportion=args_dict["val_proportion"],#0.0, 
                epochs=args_dict["epochs"],#,200, 
                batch_size=args_dict["batch_size"],#,300, 
                check_every=args_dict["check_every"],#1000
                lambda1=args_dict["lambda1"]
            )
        else:
            raise NotImplementedError()
        
    else:
        raise ValueError("read_in_data_args: model_type == "+str(args_dict["model_type"])+" IS NOT SUPPORTED (CURRENTLY)")

    pass




def call_model_eval_method(model, eval_model_type, args_dict):
    performance_components = None
    if "cMLP" in eval_model_type:
        assert "REDCLIFF" not in eval_model_type
        n_series = model.num_series
        avg_forecasting_loss, \
        avg_adj_penalty, \
        avg_dagness_reg_loss, \
        avg_dagness_lag_loss, \
        avg_dagness_node_loss, \
        avg_combo_loss = model.validate_training(
            args_dict["X_val"], 
            args_dict["input_length"], 
            args_dict["output_length"], 
            n_series, #args_dict["num_series"], 
            report_normalized_loss_components=True
        )
        performance_components = [
            avg_forecasting_loss, 
            avg_adj_penalty, 
            avg_dagness_reg_loss, 
            avg_dagness_lag_loss, 
            avg_dagness_node_loss, 
            avg_combo_loss
        ]

        curr_gc_est = model.GC(threshold=False, ignore_lag=False, combine_wavelet_representations=False, rank_wavelets=False)
        assert len(curr_gc_est) == 1
        curr_l1_loss = None
        try:
            gc_est = curr_gc_est[0] / np.max(curr_gc_est[0])
            curr_l1_loss = torch.norm(torch.from_numpy(gc_est), 1)
        except:
            gc_est = curr_gc_est[0] / torch.max(curr_gc_est[0])
            curr_l1_loss = torch.norm(gc_est, 1)
        
        curr_l1_loss = curr_l1_loss.detach().numpy()
        performance_components += performance_components + [curr_l1_loss]

    elif "REDCLIFF" in eval_model_type and "cLSTM" not in eval_model_type and "CLSTM" not in eval_model_type and "DGCNN" not in eval_model_type:
        n_series = model.num_series
        avg_forecasting_loss = None
        avg_factor_loss = None
        avg_factor_cos_sim_penalty = None
        avg_fw_l1_penalty = None
        avg_adj_penalty = None
        avg_dagness_reg_loss = None
        avg_dagness_lag_loss = None
        avg_dagness_node_loss = None
        avg_combo_loss = None
        if model.num_supervised_factors > 0:
            avg_forecasting_loss, \
            avg_factor_loss, \
            avg_factor_cos_sim_penalty, \
            avg_fw_l1_penalty, \
            avg_adj_penalty, \
            avg_dagness_reg_loss, \
            avg_dagness_lag_loss, \
            avg_dagness_node_loss, \
            avg_combo_loss, \
            _, _, _, _, _ = model.validate_training(
                args_dict["X_val"], 
                args_dict["output_length"], 
                n_series, 
                factor_score_val_acc_history=[], 
                factor_score_val_tpr_history=[], 
                factor_score_val_tnr_history=[], 
                factor_score_val_fpr_history=[], 
                factor_score_val_fnr_history=[]
            )
        else:
            avg_forecasting_loss, \
            avg_factor_loss, \
            avg_factor_cos_sim_penalty, \
            avg_fw_l1_penalty, \
            avg_adj_penalty, \
            avg_dagness_reg_loss, \
            avg_dagness_lag_loss, \
            avg_dagness_node_loss, \
            avg_combo_loss = model.validate_training(
                args_dict["X_val"], 
                args_dict["input_length"], 
                args_dict["output_length"], 
                n_series#args_dict["num_series"]
            )
        performance_components = [
            avg_forecasting_loss, 
            avg_factor_loss, 
            avg_factor_cos_sim_penalty, 
            avg_fw_l1_penalty, 
            avg_adj_penalty, 
            avg_dagness_reg_loss, 
            avg_dagness_lag_loss, 
            avg_dagness_node_loss, 
            avg_combo_loss
        ]

    elif "REDCLIFF" in eval_model_type and ("cLSTM" in eval_model_type or "CLSTM" in eval_model_type):
        n_series = model.num_series
        avg_forecasting_loss = None
        avg_factor_loss = None
        avg_factor_cos_sim_penalty = None
        avg_fw_l1_penalty = None
        avg_adj_penalty = None
        avg_dagness_reg_loss = None
        avg_combo_loss = None
        if model.num_supervised_factors > 0:
            avg_forecasting_loss, \
            avg_factor_loss, \
            avg_factor_cos_sim_penalty, \
            avg_fw_l1_penalty, \
            avg_adj_penalty, \
            avg_dagness_reg_loss, \
            avg_combo_loss, \
            _, _, _, _, _ = model.validate_training(
                args_dict["X_val"], 
                args_dict["max_input_length"], 
                args_dict["context"], 
                n_series, 
                factor_score_val_acc_history=[], 
                factor_score_val_tpr_history=[], 
                factor_score_val_tnr_history=[], 
                factor_score_val_fpr_history=[], 
                factor_score_val_fnr_history=[]
            )
        else:
            avg_forecasting_loss, \
            avg_factor_loss, \
            avg_factor_cos_sim_penalty, \
            avg_fw_l1_penalty, \
            avg_adj_penalty, \
            avg_dagness_reg_loss, \
            avg_combo_loss = model.validate_training(
                args_dict["X_val"], 
                args_dict["max_input_length"], 
                args_dict["context"], 
                n_series#args_dict["num_series"]
            )
        performance_components = [
            avg_forecasting_loss, 
            avg_factor_loss, 
            avg_factor_cos_sim_penalty, 
            avg_fw_l1_penalty, 
            avg_adj_penalty, 
            avg_dagness_reg_loss, 
            avg_combo_loss
        ]

    elif "REDCLIFF" in eval_model_type and "DGCNN" in eval_model_type:
        n_series = model.num_series
        avg_forecasting_loss = None
        avg_factor_loss = None
        avg_factor_cos_sim_penalty = None
        avg_fw_l1_penalty = None
        avg_adj_penalty = None
        avg_dagness_reg_loss = None
        avg_combo_loss = None
        if model.num_supervised_factors > 0:
            avg_forecasting_loss, \
            avg_factor_loss, \
            avg_factor_cos_sim_penalty, \
            avg_fw_l1_penalty, \
            avg_adj_penalty, \
            avg_dagness_reg_loss, \
            avg_combo_loss, \
            _, _, _, _, _ = model.validate_training(
                args_dict["X_val"], 
                n_series, 
                factor_score_val_acc_history=[], 
                factor_score_val_tpr_history=[], 
                factor_score_val_tnr_history=[], 
                factor_score_val_fpr_history=[], 
                factor_score_val_fnr_history=[]
            )
        else:
            avg_forecasting_loss, \
            avg_factor_loss, \
            avg_factor_cos_sim_penalty, \
            avg_fw_l1_penalty, \
            avg_adj_penalty, \
            avg_dagness_reg_loss, \
            avg_combo_loss = model.validate_training(
                args_dict["X_val"], 
                n_series
            )
        performance_components = [
            avg_forecasting_loss, 
            avg_factor_loss, 
            avg_factor_cos_sim_penalty, 
            avg_fw_l1_penalty, 
            avg_adj_penalty, 
            avg_dagness_reg_loss, 
            avg_combo_loss
        ]
        
    elif "cLSTM" in eval_model_type:
        assert "REDCLIFF" not in eval_model_type
        n_series = model.num_series
        avg_forecasting_loss, \
        avg_adj_penalty, \
        avg_dagness_loss, \
        avg_smooth_loss, \
        avg_val_combo_loss = model.training_sim_eval(
            args_dict["X_val"],
            args_dict["max_input_length"], 
            args_dict["context"], 
            n_series, 
            return_loss_componentwise=True, 
            report_normalized_loss_components=False
        )
        performance_components = [
            avg_forecasting_loss, 
            avg_adj_penalty, 
            avg_dagness_loss, 
            avg_smooth_loss, 
            avg_val_combo_loss
        ]

        curr_gc_est = model.GC(threshold=False, combine_wavelet_representations=False, rank_wavelets=False)
        assert len(curr_gc_est) == 1
        curr_l1_loss = None
        try:
            curr_l1_loss = torch.norm(torch.from_numpy(curr_gc_est[0]), 1).detach().numpy()
        except:
            curr_l1_loss = torch.norm(curr_gc_est[0], 1).detach().numpy()

        performance_components += performance_components + [curr_l1_loss]
        
    elif "DCSFA" in eval_model_type:
        recon_mse, \
        avg_recon_mse, \
        score_mse, \
        avg_score_mse, \
        gc_mse = model.evaluate(
            args_dict["X_val"], 
            args_dict["y_val"], 
            args_dict["true_GC_tensor"], 
            args_dict["save_root_path"], 
            threshold=False, 
            ignore_features=True
        )
        performance_components = [
            recon_mse, 
            avg_recon_mse, 
            score_mse, 
            avg_score_mse, 
            gc_mse
        ]

    elif "DGCNN" in eval_model_type:
        assert "REDCLIFF" not in eval_model_type
        avg_factor_loss = model.training_eval(args_dict["X_val"])

        # Check for early stopping.
        curr_gc_est = model.GC(threshold=False, combine_node_feature_edges=False)
        max_val = None
        try:
            max_val = torch.max(torch.from_numpy(curr_gc_est))
        except:
            max_val = torch.max(curr_gc_est)
        curr_gc_est = 1.6*curr_gc_est/max_val
        mask = curr_gc_est >= 0. # max val for true noLag GC is 1.6, with 0.4 as a min true activation => 0.25 is target min val, but add wiggle room
        curr_gc_est = curr_gc_est * mask
        curr_l1_loss = None
        try:
            curr_l1_loss = torch.norm(torch.from_numpy(curr_gc_est), 1)
        except:
            curr_l1_loss = torch.norm(curr_gc_est, 1)

        performance_components = [avg_factor_loss, curr_l1_loss.detach().numpy()]

    elif "DYNOTEARS" in eval_model_type:
        curr_avg_val_loss = model.evaluate(
            args_dict["X_val"], 
            args_dict["save_root_path"], 
            lag_size=args_dict["lag_size"]
        )
        performance_components = [curr_avg_val_loss]
        
    else:
        raise ValueError("read_in_data_args: model_type == "+str(eval_model_type)+" IS NOT SUPPORTED (CURRENTLY)")

    return performance_components