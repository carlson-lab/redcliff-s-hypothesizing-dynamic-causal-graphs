import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import os
import pickle as pkl
from sklearn.metrics import roc_auc_score, confusion_matrix

from models.cmlp import cMLP
from models.redcliff_factor_score_embedders import cEmbedder, DGCNN_Embedder, MLPClassifier, MLPClassifierForSingleObjective, MLPClassifierForMultipleObjectives
from general_utils.metrics import DAGNessLoss, get_f1_score, compute_cosine_similarity, deltacon0, deltacon0_with_directed_degrees, deltaffinity, path_length_mse, compute_cosine_similarities_within_set_of_pytorch_tensors
from general_utils.misc import flatten_GC_estimate_with_lags_and_gradient_tracking, unflatten_GC_estimate_with_lags, sort_unsupervised_estimates
from general_utils.model_utils import restore_parameters, track_receiver_operating_characteristic_stats_for_redcliff_models, track_deltacon0_related_stats_for_redcliff_models, track_l1_norm_stats_of_gc_ests_from_redcliff_models, track_cosine_similarity_stats_of_gc_ests_from_redcliff_models
from general_utils.plotting import plot_curve, plot_curve_comparisson, plot_curve_comparisson_from_dict, plot_all_signal_channels, plot_gc_est_comparissons_by_factor, plot_x_simulation_comparisson



class REDCLIFF_S_CMLP(nn.Module):
    def __init__(self, num_chans, gen_lag, gen_hidden, embed_lag, embed_hidden_sizes, num_in_timesteps, num_out_timesteps,
                 num_factors, num_supervised_factors, coeff_dict, use_sigmoid_restriction, factor_score_embedder_type, 
                 factor_score_embedder_args, primary_gc_est_mode, forward_pass_mode, num_sims=1, wavelet_level=None, 
                 save_path=None, training_mode="pretrain_embedder_and_pretrain_factor_then_combined", num_pretrain_epochs=0,
                 num_acclimation_epochs=0):
        '''
        REDCLIFF_S_CMLP model with num_factors cMLP-based factors per time series.
        '''
        super(REDCLIFF_S_CMLP, self).__init__()
        self.MAX_NUM_SAMPS_FOR_GC_VIS = 5
        self.MAX_NUM_SAMPS_FOR_GC_PROGRESS_TRACKING = 40
        self.num_chans = num_chans # p in original cMLP implementation
        if wavelet_level is not None:
            self.num_series = num_chans*(wavelet_level+1)
        else:
            self.num_series = num_chans
        self.gen_lag = gen_lag
        self.gen_hidden = gen_hidden
        self.num_gen_hiddens = len(gen_hidden)
        self.embed_lag = embed_lag
        self.embed_hidden_sizes = embed_hidden_sizes
        self.num_in_timesteps = num_in_timesteps
        self.num_out_timesteps = num_out_timesteps
        self.num_factors_nK = num_factors
        self.num_supervised_factors = num_supervised_factors
        self.coeff_dict = coeff_dict
        self.FORECAST_COEFF = coeff_dict["FORECAST_COEFF"]
        self.FACTOR_SCORE_COEFF = coeff_dict["FACTOR_SCORE_COEFF"]
        self.FACTOR_COS_SIM_COEFF = coeff_dict["FACTOR_COS_SIM_COEFF"]
        self.FACTOR_WEIGHT_L1_COEFF = coeff_dict["FACTOR_WEIGHT_L1_COEFF"]
        self.ADJ_L1_REG_COEFF = coeff_dict["ADJ_L1_REG_COEFF"]
        self.DAGNESS_REG_COEFF = coeff_dict["DAGNESS_REG_COEFF"]
        self.DAGNESS_LAG_COEFF = coeff_dict["DAGNESS_LAG_COEFF"]
        self.DAGNESS_NODE_COEFF = coeff_dict["DAGNESS_NODE_COEFF"]
        self.num_sims = num_sims
        self.wavelet_level = wavelet_level

        assert training_mode in [
            "pretrain_embedder_then_acclimate_factors_then_combined", 
            "pretrain_embedder_then_post_train_factor_withComboCosSimL1FreezeByEpoch", 
            "pretrain_embedder_then_post_train_factor_withComboCosSimL1FreezeByBatch", 
            "pretrain_embedder_then_post_train_factor_withL1FreezeByEpoch", 
            "pretrain_embedder_then_post_train_factor_withL1FreezeByBatch", 
            "pretrain_embedder_then_post_train_factor", 
            "pretrain_embedder_and_pretrain_factor_then_combined", 
            "pretrain_embedder_then_combined", 
            "pretrain_factor_then_combined", 
            "combined"
        ]
        self.training_mode = training_mode
        if "pretrain" in training_mode:
            assert num_pretrain_epochs > 0
        else:
            assert num_pretrain_epochs == 0
        if "acclimate" in training_mode:
            assert num_acclimation_epochs > 0
        else:
            assert num_acclimation_epochs == 0
        self.num_pretrain_epochs = num_pretrain_epochs
        self.num_acclimation_epochs = num_acclimation_epochs
        
        assert forward_pass_mode in [
            "apply_factor_weights_at_each_sim_step", 
            "apply_factor_weights_after_sim_completion"
        ]
        self.forward_pass_mode = forward_pass_mode
        
        self.supervised_loss_fn = nn.MSELoss(reduction='mean')
        self.dagness_loss_fn = DAGNessLoss()
        
        # Set up embedder.
        self.use_sigmoid_restriction = use_sigmoid_restriction
        assert factor_score_embedder_type in ["cEmbedder","DGCNN","Vanilla_Embedder"]
        self.CAUSAL_EMBEDDER_TYPES = ["cEmbedder","DGCNN"]
        self.factor_score_embedder_type = factor_score_embedder_type
        self.factor_score_embedder_args = factor_score_embedder_args
        self.POSSIBLE_GC_EST_MODES = [
            "fixed_factor_exclusive",        # original mode
            "raw_embedder",                  # simply the first layer of the factor embedder, interpretted as GC estimate
            "conditional_factor_exclusive",  # original mode, but factors are weighted by factor embedder conditioned on input X
            "fixed_embedder_exclusive",      # the 'system' causal graph, i.e. the outer product of the factor embedder first layer
            "conditional_embedder_exclusive",    # the 'current' causal graph according to the embedder, i.e. the sum of outer products of each row of the raw_embedder GC graph with weights from factor embedder conditioned on input X
            "fixed_factor_fixed_embedder",       # each individual factor GC estimate added to the fixed_embedder_exclusive GC estimate
            "conditional_factor_fixed_embedder", # each factor GC estimate - weighted by embedder conditioned on X - added to the fixed_embedder_exclusive GC estimate
            "fixed_factor_conditional_embedder", # each individual factor GC estimate added to the conditional_embedder_exclusive GC estimate
            "conditional_factor_conditional_embedder", # each factor GC estimate - weighted by embedder conditioned on X - added to the conditional_embedder_exclusive GC estimate
        ]
        assert primary_gc_est_mode in self.POSSIBLE_GC_EST_MODES
        self.primary_gc_est_mode = primary_gc_est_mode
        
        if factor_score_embedder_type == "cEmbedder":
            condensed_embedder_args = [num_chans,num_supervised_factors,num_factors,use_sigmoid_restriction] + []
            for arg_tuple in factor_score_embedder_args: # arg_tuple is formatted as (arg_name, arg_val); here, they'll likely contain sigmoid_eccentricity_coeff, embed_lag, hidden parameters
                condensed_embedder_args.append(arg_tuple[1])
            condensed_embedder_args = condensed_embedder_args + [wavelet_level, save_path]
            self.factor_score_embedder = cEmbedder(*condensed_embedder_args) # args are: num_chans, num_class_preds, num_factor_preds, use_sigmoid_restriction, sigmoid_eccentricity_coeff, embed_lag, hidden, wavelet_level=None, save_path=None
        elif factor_score_embedder_type == "DGCNN":
            assert self.primary_gc_est_mode != "conditional_embedder_exclusive"
            dgcnn_wavelets = wavelet_level
            if dgcnn_wavelets is None:
                dgcnn_wavelets = 0
            condensed_embedder_args = [num_chans,dgcnn_wavelets+1] + []
            print("REDCLIFF_S_CMLP.__init__: factor_score_embedder_args == ", factor_score_embedder_args)
            assert len(factor_score_embedder_args) == 4
            for arg_tuple in factor_score_embedder_args: # arg_tuple is formatted as (arg_name, arg_val); here, they'll likely contain num_features_per_node==embed_lag, num_graph_conv_layers parameters, num_hidden_nodes, sigmoid_eccentricity_coeff
                condensed_embedder_args.append(arg_tuple[1])
            condensed_embedder_args = condensed_embedder_args + [use_sigmoid_restriction, num_factors, num_supervised_factors]
            self.factor_score_embedder = DGCNN_Embedder(*condensed_embedder_args) # args are num_channels, num_wavelets_per_chan, num_features_per_node, num_graph_conv_layers, num_hidden_nodes, sigmoid_eccentricity_coeff, use_sigmoid_restriction, num_factors, num_classes
        elif factor_score_embedder_type == "Vanilla_Embedder":
            if num_supervised_factors > 0:
                self.factor_score_embedder = MLPClassifierForMultipleObjectives(
                    self.num_series, embed_lag, num_factors, num_supervised_factors, embed_hidden_sizes, use_sigmoid_restriction
                )
            else:
                self.factor_score_embedder = MLPClassifierForSingleObjective(
                    self.num_series, embed_lag, num_factors, embed_hidden_sizes, use_sigmoid_restriction
                )
        else:
            raise NotImplementedError("factor_score_embedder_type == "+str(factor_score_embedder_type))
        
        assert self.factor_score_embedder is not None
        
        # Set up factors.
        self.factors = nn.ModuleList([cMLP(num_chans, gen_lag, gen_hidden, wavelet_level=wavelet_level, save_path=save_path) for _ in range(num_factors)])
        self.gen_model = nn.ModuleList([self.factor_score_embedder, self.factors])
        pass
    
    
    def initialize_factors_with_prior(self, prior_factors_path=None, X_train=None, cost_criteria="CosineSimilarity", unsupervised_start_index=0, max_batches=10):
        if prior_factors_path is not None:
            prior_model = torch.load(prior_factors_path)
            self.factors = prior_model.factors
            del prior_model
            
        if X_train is not None:
            unordered_factor_preds = []
            true_labels = []
            for batch_num, (X, Y) in enumerate(X_train):
                if batch_num < max_batches:
                    # Set up data.
                    if torch.cuda.is_available():
                        X = X.to(device="cuda")
                    if Y is not None and len(Y.size()) > 2:
                        if Y.size()[2] > max(self.gen_lag, self.embed_lag):
                            # we are only interested in the first predicted factor weighting here, so we grab the corresponding label 
                            # (note that Y has size batch_size x num_channels x num_recorded_timesteps)
                            Y = Y[:,:,max(self.gen_lag, self.embed_lag)] 
                        else:
                            Y = Y[:,:,0]
                    
                    if self.factor_score_embedder is not None:
                        self.factor_score_embedder.eval()
                    for f in self.factors:
                        f.eval()

                    # Make Prediction(s)
                    _, _, factor_weightings, _ = self.forward(X[:,:max(self.gen_lag, self.embed_lag),:])
                    factor_weightings = [x.detach().cpu().numpy() for x in factor_weightings[:1]] # just take the first factor weighting (which is most closely tied to the original signal)
                    unordered_factor_preds = unordered_factor_preds + factor_weightings
                    true_labels.append(Y.detach().numpy())

                    del X
                    del Y
                    del factor_weightings

            unordered_factor_preds = np.vstack(unordered_factor_preds)
            true_labels = np.vstack(true_labels)
            assert len(unordered_factor_preds.shape) == 2
            assert len(true_labels.shape) == 2
            labels_across_recordings_by_factor = [unordered_factor_preds[:,i] for i in range(unordered_factor_preds.shape[1])]
            labels_across_recordings_by_ground_truth = [true_labels[:,i] for i in range(true_labels.shape[1])]

            _, matched_est_inds, matched_ground_truth_inds = sort_unsupervised_estimates(
                labels_across_recordings_by_factor, labels_across_recordings_by_ground_truth, cost_criteria=cost_criteria, 
                unsupervised_start_index=unsupervised_start_index, return_sorting_inds=True
            )

            sorted_factors = [None for _ in range(len(matched_est_inds))]
            for (est_ind, gt_ind) in zip(matched_est_inds, matched_ground_truth_inds):
                sorted_factors[gt_ind] = self.factors[unsupervised_start_index:][est_ind]
            unsorted_factors = [self.factors[unsupervised_start_index:][i] for i in range(len(self.factors[unsupervised_start_index:])) if i not in matched_est_inds]

            self.factors = nn.ModuleList(sorted_factors+unsorted_factors)
        pass
    
    
    def resume_training_from_checkpoint(self, training_meta_data_path):
        print("REDCLIFF_S_CMLP.resume_training_from_checkpoint: RESUMING TRAINING FROM CHKPT SAVED AT training_meta_data_path==", training_meta_data_path)
        with open(training_meta_data_path, 'rb') as infile:
            training_checkpoint_meta_data = pkl.load(infile)
            self.chkpt_epoch = training_checkpoint_meta_data["epoch"]
            self.chkpt_avg_forecasting_loss = training_checkpoint_meta_data["avg_forecasting_loss"]
            self.chkpt_avg_factor_loss = training_checkpoint_meta_data["avg_factor_loss"]
            self.chkpt_avg_factor_cos_sim_penalty = training_checkpoint_meta_data["avg_factor_cos_sim_penalty"]
            self.chkpt_avg_fw_l1_penalty = training_checkpoint_meta_data["avg_fw_l1_penalty"]
            self.chkpt_avg_adj_penalty = training_checkpoint_meta_data["avg_adj_penalty"]
            self.chkpt_avg_dagness_reg_loss = training_checkpoint_meta_data["avg_dagness_reg_loss"]
            self.chkpt_avg_dagness_lag_loss = training_checkpoint_meta_data["avg_dagness_lag_loss"]
            self.chkpt_avg_dagness_node_loss = training_checkpoint_meta_data["avg_dagness_node_loss"]
            self.chkpt_avg_combo_loss = training_checkpoint_meta_data["avg_combo_loss"]
            self.chkpt_best_loss = training_checkpoint_meta_data["best_loss"]
            self.chkpt_best_it = training_checkpoint_meta_data["best_it"]
            self.chkpt_f1score_histories = training_checkpoint_meta_data["f1score_histories"]
            self.chkpt_f1score_OffDiag_histories = training_checkpoint_meta_data["f1score_OffDiag_histories"]
            self.chkpt_roc_auc_histories = training_checkpoint_meta_data["roc_auc_histories"]
            self.chkpt_roc_auc_OffDiag_histories = training_checkpoint_meta_data["roc_auc_OffDiag_histories"]
            if self.num_supervised_factors > 0:
                self.chkpt_factor_score_train_acc_history = training_checkpoint_meta_data["factor_score_train_acc_history"] 
                self.chkpt_factor_score_train_tpr_history = training_checkpoint_meta_data["factor_score_train_tpr_history"] 
                self.chkpt_factor_score_train_tnr_history = training_checkpoint_meta_data["factor_score_train_tnr_history"] 
                self.chkpt_factor_score_train_fpr_history = training_checkpoint_meta_data["factor_score_train_fpr_history"] 
                self.chkpt_factor_score_train_fnr_history = training_checkpoint_meta_data["factor_score_train_fnr_history"] 
                self.chkpt_factor_score_val_acc_history = training_checkpoint_meta_data["factor_score_val_acc_history"] 
                self.chkpt_factor_score_val_tpr_history = training_checkpoint_meta_data["factor_score_val_tpr_history"] 
                self.chkpt_factor_score_val_tnr_history = training_checkpoint_meta_data["factor_score_val_tnr_history"] 
                self.chkpt_factor_score_val_fpr_history = training_checkpoint_meta_data["factor_score_val_fpr_history"] 
                self.chkpt_factor_score_val_fnr_history = training_checkpoint_meta_data["factor_score_val_fnr_history"] 
            self.chkpt_gc_factor_l1_loss_histories = training_checkpoint_meta_data["gc_factor_l1_loss_histories"]
            self.chkpt_gc_factor_cosine_sim_histories = training_checkpoint_meta_data["gc_factor_cosine_sim_histories"]
            self.chkpt_gc_factorUnsupervised_cosine_sim_histories = training_checkpoint_meta_data["gc_factorUnsupervised_cosine_sim_histories"]
            self.chkpt_deltacon0_histories = training_checkpoint_meta_data["deltacon0_histories"]
            self.chkpt_deltacon0_with_directed_degrees_histories = training_checkpoint_meta_data["deltacon0_with_directed_degrees_histories"]
            self.chkpt_deltaffinity_histories = training_checkpoint_meta_data["deltaffinity_histories"]
            self.chkpt_path_length_mse_histories = training_checkpoint_meta_data["path_length_mse_histories"]
        print("REDCLIFF_S_CMLP.resume_training_from_checkpoint: RESUMING TRAINING FROM CHKPT MADE AT chkpt_epoch==", self.chkpt_epoch)
        print("REDCLIFF_S_CMLP.resume_training_from_checkpoint: RESUMING TRAINING WITH MODEL PARAMETERS SAVED AT chkpt_best_it==", self.chkpt_best_it)
        print("REDCLIFF_S_CMLP.resume_training_from_checkpoint: WARNING - OPTIMIZERS HAVE NOT BEEN CHECKPOINTED, MEANINIG THAT RESUMED TRAINING WILL BEGIN WITH FRESH OPTIMIZERS/GRADIENT DESCENT PARAMETERS THAT WOULD NEED TO BE MENTIONED IN ANY REPORTS", flush=True)
        pass
    
    
    def forward_pass_with_new_factor_weights_at_each_sim_step(self, X, factor_weightings=None):
        '''
        Perform forward pass.
        Args:
          X: torch tensor of shape (batch, T, p).
        ''' 
        # initialize vars
        generate_new_factor_weightings = True
        if factor_weightings is not None:
            generate_new_factor_weightings = False

        in_x = torch.zeros(*X.size())
        if torch.cuda.is_available():
            in_x = in_x.to(device="cuda")
        in_x = in_x + X

        inputs = [in_x]
        x_simulations = []
        factor_preds_over_sim = []
        factor_weighting_preds = []
        state_label_preds = []
            
        # make preds according to factor_weightings
        for s in range(self.num_sims):
            if generate_new_factor_weightings:
                factor_weightings = None
            state_logits = None
            factor_preds = []
            
            curr_sim_start = s
            if s > 0:
                if x_simulations[-1].size() == inputs[-1].size():
                    inputs.append(x_simulations[-1])
                else:
                    inputs.append(torch.cat([inputs[-1][:,x_simulations[-1].size()[1]:,:], x_simulations[-1]],dim=1))

            if generate_new_factor_weightings:
                if self.factor_score_embedder_type == "DGCNN":
                    factor_weightings, state_logits = self.factor_score_embedder(torch.transpose(inputs[s][:,-1*self.embed_lag:,:], 1,2))
                else:
                    factor_weightings, state_logits = self.factor_score_embedder(inputs[s][:,-1*self.embed_lag:,:])
            else:
                if self.factor_score_embedder_type == "DGCNN":
                    _, state_logits = self.factor_score_embedder(torch.transpose(inputs[s][:,-1*self.embed_lag:,:], 1,2))
                else:
                    _, state_logits = self.factor_score_embedder(inputs[s][:,-1*self.embed_lag:,:])

            if state_logits is None:
                state_label_preds.append(factor_weightings)
            else:
                state_label_preds.append(state_logits)
            
            combined_pred = None
            for i in range(self.num_factors_nK):
                # make prediction from one factor
                curr_pred = self.factors[i](inputs[s][:,-1*self.gen_lag:,:])
                # sum over factor predictions
                if i == 0:
                    combined_pred = factor_weightings[:,i].view(-1,1,1)*curr_pred
                else:
                    combined_pred = combined_pred + factor_weightings[:,i].view(-1,1,1)*curr_pred
                factor_preds.append(curr_pred)

            # record results of current sim step
            factor_preds_over_sim.append(factor_preds)
            factor_weighting_preds.append(factor_weightings)
            x_simulations.append(combined_pred)
            pass

        x_simulations = torch.cat(x_simulations, dim=1)
        return x_simulations, factor_preds_over_sim, factor_weighting_preds, state_label_preds
    
    
    def forward_pass_with_factor_weighting_at_sim_completion(self, X, factor_weightings=None): 
        '''
        Perform forward pass.
        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        # initialize vars
        state_logits = None
        if factor_weightings is None:
            if self.factor_score_embedder_type == "DGCNN":
                factor_weightings, state_logits = self.factor_score_embedder(torch.transpose(X[:,-1*self.embed_lag:,:], 1,2))
            else:
                factor_weightings, state_logits = self.factor_score_embedder(X[:,-1*self.embed_lag:,:])
        else:
            if self.factor_score_embedder_type == "DGCNN":
                _, state_logits = self.factor_score_embedder(torch.transpose(X[:,-1*self.embed_lag:,:], 1,2))
            else:
                _, state_logits = self.factor_score_embedder(X[:,-1*self.embed_lag:,:])

            if state_logits is None:
                state_logits = factor_weightings
        state_label_preds = [state_logits for _ in range(self.num_sims)]
            
        # make preds according to INDEPENDENT factors
        factor_preds_over_sim = []
        for i in range(self.num_factors_nK):
            # make prediction from one factor
            factor_preds = []
            curr_input = None
            for s in range(self.num_sims):
                curr_sim_start = s
                if s > 0:
                    if factor_preds[-1].size() == curr_input.size():
                        curr_input = factor_preds[-1]
                    else:
                        curr_input = torch.cat([curr_input[:,factor_preds[-1].size()[1]:,:], factor_preds[-1]],dim=1)
                else:
                    curr_input = torch.zeros(*X[:,-1*self.gen_lag:,:].size())
                    if torch.cuda.is_available():
                        in_x = in_x.to(device="cuda")
                    curr_input = curr_input + X[:,-1*self.gen_lag:,:]
                    
                curr_pred = self.factors[i](curr_input)
                # record results of current sim step
                factor_preds.append(curr_pred)

            # record results of current factor sim
            factor_preds = torch.cat(factor_preds, dim=1)
            factor_preds_over_sim.append(factor_preds)
            
        # make preds according to factor_weightings
        x_simulations = None
        for i in range(self.num_factors_nK):
                # sum over factor predictions
                if i == 0:
                    x_simulations = factor_weightings[:,i].view(-1,1,1)*factor_preds_over_sim[i]
                else:
                    x_simulations = x_simulations + factor_weightings[:,i].view(-1,1,1)*factor_preds_over_sim[i]

        return x_simulations, factor_preds_over_sim, [factor_weightings], state_label_preds
        
        
    def forward(self, X, factor_weightings=None):
        '''
        Perform forward pass.
        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        x_simulations = None
        factor_preds_over_sim = None
        factor_weighting_preds = None
        state_label_preds = None
        
        if self.forward_pass_mode == "apply_factor_weights_at_each_sim_step":
            x_simulations, \
            factor_preds_over_sim, \
            factor_weighting_preds, \
            state_label_preds = self.forward_pass_with_new_factor_weights_at_each_sim_step(X, factor_weightings=factor_weightings)
        elif self.forward_pass_mode == "apply_factor_weights_after_sim_completion":
            x_simulations, \
            factor_preds_over_sim, \
            factor_weighting_preds, \
            state_label_preds = self.forward_pass_with_factor_weighting_at_sim_completion(X, factor_weightings=factor_weightings)
        else:
            raise ValueError("redcliff_s_cmlp.forward: UNRECOGNIZED self.forward_pass_mode == "+str(self.forward_pass_mode))
        
        return x_simulations, factor_preds_over_sim, factor_weighting_preds, state_label_preds

    
    def GC(self, gc_est_mode, X=None, threshold=True, ignore_lag=True, combine_wavelet_representations=False, rank_wavelets=False):
        '''
        Extract learned Granger causality from each factor.
        Args:
          threshold: return norm of weights, or whether norm is nonzero.
        Returns:
          GCs: list of lists of (m x n x L) pytorch tensors. Entry (i, j, k) of each tensor indicates the effect of variable j in terms of
            Granger causality on variable i at lag k ***. The outer list represents 'sample index' (default is len 1 for non-batched queries) 
            and the inner list represents 'factor index' (default is len 1 for non-factorized queries)
        
        Notes: 
         *** If the architecture of the model's factors does not incorporate lags, a lag of L==1 will be added in dimension 2 of the gc_est tensors. 
        
         - POSSIBLE_GC_EST_MODES: 
              "fixed_factor_exclusive": the original mode
              "raw_embedder": simply the first layer of the factor embedder, interpretted as GC estimate
              "conditional_factor_exclusive": original mode, but factors are weighted by factor embedder conditioned on input X
              "fixed_embedder_exclusive": the 'system' causal graph, i.e. the outer product of the factor embedder first layer
              "conditional_embedder_exclusive": the 'current' causal graph according to the embedder, i.e. the outer products 
                                                of each row of the raw_embedder GC graph with weights from factor embedder 
                                                conditioned on input X
              "fixed_factor_fixed_embedder": each individual factor GC estimate added to the fixed_embedder_exclusive GC estimate
              "conditional_factor_fixed_embedder": each factor GC estimate - weighted by embedder conditioned on X - added to the 
                                                   fixed_embedder_exclusive GC estimate
              "fixed_factor_conditional_embedder": each individual factor GC estimate added to the conditional_embedder_exclusive 
                                                   GC estimate
              "conditional_factor_conditional_embedder": each factor GC estimate - weighted by embedder conditioned on X - added to 
                                                         the conditional_embedder_exclusive GC estimate
        '''
        if gc_est_mode == "fixed_factor_exclusive": # returns a list of a single list of tensors
            factor_gc_ests = [
                factor.GC(
                    threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, rank_wavelets=rank_wavelets
                ) for factor in self.factors
            ]
            if len(factor_gc_ests[0].size()) != 3:
                assert len(factor_gc_ests[0].size()) == 2 # sanity check
                assert factor_gc_ests[0].size()[0] == factor_gc_ests[0].size()[1] # sanity check
                orig_num_vars = factor_gc_ests[0].size()[0]
                factor_gc_ests = [x.view(orig_num_vars,orig_num_vars,1) for x in factor_gc_ests]
            return [factor_gc_ests]
        
        elif gc_est_mode == "raw_embedder": # returns a list of a list of a single of tensor
            assert self.factor_score_embedder_type in self.CAUSAL_EMBEDDER_TYPES 
            embedder_gc_layer = None
            if self.factor_score_embedder_type == "cEmbedder":
                embedder_gc_layer = self.factor_score_embedder.GC(
                    threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, rank_wavelets=rank_wavelets
                )
            elif self.factor_score_embedder_type == "DGCNN":
                embedder_gc_layer = self.factor_score_embedder.GC(threshold=threshold, combine_node_feature_edges=combine_wavelet_representations)
            else:
                raise ValueError()
            
            if len(embedder_gc_layer.size()) != 3:
                assert len(embedder_gc_layer.size()) == 2 # sanity check
                if self.factor_score_embedder_type == "DGCNN":
                    assert embedder_gc_layer.size()[0] == self.num_series # sanity check
                    assert embedder_gc_layer.size()[0] == embedder_gc_layer.size()[1] # sanity check
                    embedder_gc_layer = embedder_gc_layer.view(self.num_series, self.num_series, 1)
                else:
                    assert embedder_gc_layer.size()[0] == self.num_factors_nK # sanity check
                    orig_num_vars = embedder_gc_layer.size()[1]
                    embedder_gc_layer = embedder_gc_layer.view(self.num_factors_nK, orig_num_vars, 1)
            return [[embedder_gc_layer]]
                
        elif gc_est_mode == "conditional_factor_exclusive": # returns a list (with len batch_size) of lists (each of len num_factors) of tensors
            factor_weightings = None
            if self.factor_score_embedder_type == "DGCNN":
                factor_weightings, _ = self.factor_score_embedder(torch.transpose(X[:,-1*self.embed_lag:,:], 1,2))
            else:
                factor_weightings, _ = self.factor_score_embedder(X[:,-1*self.embed_lag:,:])
            
            factor_gc_ests = self.GC(
                "fixed_factor_exclusive", X=None, threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, 
                rank_wavelets=rank_wavelets
            )[0]
            conditional_gc_ests_by_sample = []
            for batch_ind in range(factor_weightings.size()[0]):
                weighted_factor_ests = []
                for factor_ind in range(factor_weightings.size()[1]):
                    weighted_factor_ests.append(factor_weightings[batch_ind,factor_ind]*factor_gc_ests[factor_ind])
                conditional_gc_ests_by_sample.append(weighted_factor_ests)
            return conditional_gc_ests_by_sample
        
        elif gc_est_mode == "fixed_embedder_exclusive": # returns a list of a list of a single of tensor
            raw_embedder_gc_est = self.GC(
                "raw_embedder", X=None, threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, 
                rank_wavelets=rank_wavelets
            )[0][0]
            assert len(raw_embedder_gc_est.size()) == 3
            num_vars_represented = raw_embedder_gc_est.size()[1]
            lag_dim_size = raw_embedder_gc_est.size()[2]
            sys_gc_est = None
            if self.factor_score_embedder_type == "DGCNN":
                assert raw_embedder_gc_est.size()[0] == raw_embedder_gc_est.size()[1]
                sys_gc_est = raw_embedder_gc_est
            else:
                assert raw_embedder_gc_est.size()[0] == self.num_factors_nK
                sys_gc_est = torch.matmul(raw_embedder_gc_est.transpose(0,2), raw_embedder_gc_est.transpose(0,2).transpose(1,2)).transpose(0,2)
            assert len(sys_gc_est.size()) == 3
            assert sys_gc_est.size()[0] == sys_gc_est.size()[1]
            assert sys_gc_est.size()[0] == num_vars_represented
            assert sys_gc_est.size()[2] == lag_dim_size
            return [[sys_gc_est]]
        
        elif gc_est_mode == "conditional_embedder_exclusive": # returns a list (with len batch_size) of lists (each of len num_factors) of tensors
            raw_embedder_gc_est = self.GC(
                "raw_embedder", X=None, threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, 
                rank_wavelets=rank_wavelets
            )[0][0]
            assert len(raw_embedder_gc_est.size()) == 3
            num_vars_represented = raw_embedder_gc_est.size()[1]
            lag_dim_size = raw_embedder_gc_est.size()[2]
            if self.factor_score_embedder_type == "DGCNN":
                raise ValueError("conditional_embedder_exclusive is not supported for model with DGCNN factor score embedder type")
            else:
                assert raw_embedder_gc_est.size()[0] == self.num_factors_nK
            
            factor_weightings = None
            if self.factor_score_embedder_type == "DGCNN":
                factor_weightings, _ = self.factor_score_embedder(torch.transpose(X[:,-1*self.embed_lag:,:], 1,2))
            else:
                factor_weightings, _ = self.factor_score_embedder(X[:,-1*self.embed_lag:,:])
            
            conditional_gc_ests_by_sample = []
            for batch_ind in range(factor_weightings.size()[0]):
                conditional_factor_ests = []
                for factor_ind in range(factor_weightings.size()[1]):
                    curr_factor_gc_contrib = raw_embedder_gc_est[factor_ind,:,:].view(1,num_vars_represented,lag_dim_size)
                    curr_factor_gc_contrib = factor_weightings[batch_ind,factor_ind] * torch.matmul(
                        curr_factor_gc_contrib.transpose(0,2), curr_factor_gc_contrib.transpose(0,2).transpose(1,2)
                    ).transpose(0,2)
                    conditional_factor_ests.append(curr_factor_gc_contrib)
                conditional_gc_ests_by_sample.append(conditional_factor_ests)
            return conditional_gc_ests_by_sample
        
        elif gc_est_mode == "fixed_factor_fixed_embedder": # returns a list of tensors
            factor_gc_ests = self.GC(
                "fixed_factor_exclusive", X=None, threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, 
                rank_wavelets=rank_wavelets
            )[0]
            embedder_gc_est = self.GC(
                "fixed_embedder_exclusive", X=None, threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, 
                rank_wavelets=rank_wavelets
            )[0][0]
            if not ignore_lag:
                return [[x[:,:,-1*min(self.gen_lag, self.embed_lag):]+embedder_gc_est[:,:,-1*min(self.gen_lag, self.embed_lag):] for x in factor_gc_ests]]
            return [[x+embedder_gc_est for x in factor_gc_ests]]
        
        elif gc_est_mode == "conditional_factor_fixed_embedder": # returns a list (with len batch_size) of lists (each of len num_factors) of tensors
            conditional_gc_ests_by_sample = self.GC(
                "conditional_factor_exclusive", X=X, threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, 
                rank_wavelets=rank_wavelets
            )
            embedder_gc_est = self.GC(
                "fixed_embedder_exclusive", X=None, threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, 
                rank_wavelets=rank_wavelets
            )[0][0]
            for batch_ind in range(X.size()[0]):
                for factor_ind in range(self.num_factors_nK):
                    if ignore_lag:
                        conditional_gc_ests_by_sample[batch_ind][factor_ind] = conditional_gc_ests_by_sample[batch_ind][factor_ind] + embedder_gc_est
                    else:
                        conditional_gc_ests_by_sample[batch_ind][factor_ind] = conditional_gc_ests_by_sample[batch_ind][factor_ind][:,:,-1*min(self.gen_lag, self.embed_lag):] + embedder_gc_est[:,:,-1*min(self.gen_lag, self.embed_lag):]
            return conditional_gc_ests_by_sample
            
        elif gc_est_mode == "fixed_factor_conditional_embedder": # returns a list (with len batch_size) of lists (each of len num_factors) of tensors
            factor_gc_ests = self.GC(
                "fixed_factor_exclusive", X=None, threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, 
                rank_wavelets=rank_wavelets
            )[0]
            conditional_gc_ests_by_sample = self.GC(
                "conditional_embedder_exclusive", X=X, threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, 
                rank_wavelets=rank_wavelets
            )
            for batch_ind in range(X.size()[0]):
                for factor_ind in range(self.num_factors_nK):
                    if ignore_lag:
                        conditional_gc_ests_by_sample[batch_ind][factor_ind] = conditional_gc_ests_by_sample[batch_ind][factor_ind] + factor_gc_ests[factor_ind]
                    else:
                        conditional_gc_ests_by_sample[batch_ind][factor_ind] = conditional_gc_ests_by_sample[batch_ind][factor_ind][:,:,-1*min(self.gen_lag, self.embed_lag):] + factor_gc_ests[factor_ind][:,:,-1*min(self.gen_lag, self.embed_lag):]
            return conditional_gc_ests_by_sample
        
        elif gc_est_mode == "conditional_factor_conditional_embedder": # returns a list (with len batch_size) of lists (each of len num_factors) of tensors
            factor_conditional_gc_ests_by_sample = self.GC(
                "conditional_factor_exclusive", X=X, threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, 
                rank_wavelets=rank_wavelets
            )
            embedder_conditional_gc_ests_by_sample = self.GC(
                "conditional_embedder_exclusive", X=X, threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, 
                rank_wavelets=rank_wavelets
            )
            conditional_gc_ests_by_sample = []
            for batch_ind in range(X.size()[0]):
                curr_sample_gc_ests_by_factor = []
                for factor_ind in range(self.num_factors_nK):
                    if ignore_lag:
                        curr_sample_gc_ests_by_factor.append(factor_conditional_gc_ests_by_sample[batch_ind][factor_ind] + embedder_conditional_gc_ests_by_sample[batch_ind][factor_ind])
                    else:
                        curr_sample_gc_ests_by_factor.append(factor_conditional_gc_ests_by_sample[batch_ind][factor_ind][:,:,-1*min(self.gen_lag, self.embed_lag):] + embedder_conditional_gc_ests_by_sample[batch_ind][factor_ind][:,:,-1*min(self.gen_lag, self.embed_lag):])
                conditional_gc_ests_by_sample.append(curr_sample_gc_ests_by_factor)
            return conditional_gc_ests_by_sample
        
        else:
            raise ValueError("GC EST MODE == "+str(gc_est_mode)+" IS NOT SUPPORTED")
        pass
    
    
    def compute_loss(self, conditioning_X, preds, targets, factor_scores, factor_labels, gc_est_mode, node_dag_scale=0.1, embedder_pretrain_loss=False, factor_pretrain_loss=False):
        gc = self.GC(gc_est_mode, X=conditioning_X, threshold=False, ignore_lag=True, combine_wavelet_representations=False, rank_wavelets=False)
        gc_lagged = self.GC(gc_est_mode, X=conditioning_X, threshold=False, ignore_lag=False, combine_wavelet_representations=False, rank_wavelets=False)

        # get supervised loss components
        forecasting_loss = self.FORECAST_COEFF*sum([self.supervised_loss_fn(preds[:, :, i], targets[:, :, i]) for i in range(self.num_series)])
        factor_loss = torch.tensor([0.0], requires_grad=True) # see https://discuss.pytorch.org/t/how-to-initialize-zero-loss-tensor/86888/2
        if torch.cuda.is_available():
            factor_loss = factor_loss.to(device="cuda")
        if factor_scores is not None and factor_scores[0] is not None and self.num_supervised_factors > 0:
            # ONLY UPDATE THE CLASSIFIER BASED ON THE FIRST PREDICTION, AS THAT INPUT WILL BE MORE GROUNDED IN THE ORIGINAL SIGNAL
            if len(factor_labels.size()) == 3:
                if factor_labels.size()[2] > max(self.gen_lag, self.embed_lag):
                    for y, yhat in zip([factor_labels[:,:,max(self.gen_lag, self.embed_lag)+l] for l in range(factor_labels.size()[2]-max(self.gen_lag, self.embed_lag))],factor_scores):
                        factor_loss = factor_loss + self.FACTOR_SCORE_COEFF*self.supervised_loss_fn(yhat[:,:self.num_supervised_factors], y[:,:self.num_supervised_factors])
                else: # case for datasets such as the DREAM4-based datasets
                    y = factor_labels[:,:,0]
                    yhat = factor_scores[0]
                    for lagged_yhat in factor_scores[1:]:
                        yhat = yhat + lagged_yhat
                    yhat = yhat / (1.*len(factor_scores))
                    factor_loss = factor_loss + self.FACTOR_SCORE_COEFF*self.supervised_loss_fn(yhat[:,:self.num_supervised_factors], y[:,:self.num_supervised_factors])
            elif len(factor_labels.size()) == 2: # case for DREAM4 orig. data
                y = factor_labels[:,:]
                yhat = factor_scores[0]
                for lagged_yhat in factor_scores[1:]:
                    yhat = yhat + lagged_yhat
                yhat = yhat / (1.*len(factor_scores))
                factor_loss = factor_loss + self.FACTOR_SCORE_COEFF*self.supervised_loss_fn(yhat[:,:self.num_supervised_factors], y[:,:self.num_supervised_factors])
            else:
                raise NotImplementedError("Cannot handle ground-truth labels with Y.size() == "+str(Y.size()))
        
        # get regularization penalties
        fw_l1_penalty = self.FACTOR_WEIGHT_L1_COEFF*(torch.norm(factor_scores[0], 1) - 1.)
        factor_cos_sim_penalty = None
        adj_l1_penalty = None
        reg_dagness_loss = None
        for sample_ind in range(len(gc)):
            if sample_ind == 0:
                if len(gc[sample_ind]) > 1:
                    factor_cos_sim_penalty = self.FACTOR_COS_SIM_COEFF*torch.sum(compute_cosine_similarities_within_set_of_pytorch_tensors(gc[sample_ind], include_diag=False))
                for f_ind, f_Adj in enumerate(gc_lagged[sample_ind]):
                    if f_ind == 0:
                        adj_l1_penalty = self.ADJ_L1_REG_COEFF*sum([torch.log(torch.tensor(i+2.))*torch.norm(f_Adj[:,:,i], 1) for i in range(f_Adj.size()[2])])# see https://pytorch.org/docs/stable/generated/torch.norm.html
                    else:
                        adj_l1_penalty += self.ADJ_L1_REG_COEFF*sum([torch.log(torch.tensor(i+2.))*torch.norm(f_Adj[:,:,i], 1) for i in range(f_Adj.size()[2])])# see https://pytorch.org/docs/stable/generated/torch.norm.html
            else: 
                if len(gc[sample_ind]) > 1:
                    factor_cos_sim_penalty += self.FACTOR_COS_SIM_COEFF*torch.sum(compute_cosine_similarities_within_set_of_pytorch_tensors(gc[sample_ind], include_diag=False))
                for _, f_Adj in enumerate(gc_lagged[sample_ind]):
                    adj_l1_penalty += self.ADJ_L1_REG_COEFF*sum([torch.log(torch.tensor(i+2.))*torch.norm(f_Adj[:,:,i], 1) for i in range(f_Adj.size()[2])])# see https://pytorch.org/docs/stable/generated/torch.norm.html

        # combine loss compoents and return
        combo_loss = None
        if embedder_pretrain_loss:
            assert not factor_pretrain_loss
            combo_loss = factor_loss + fw_l1_penalty
        elif factor_pretrain_loss:
            combo_loss = forecasting_loss + fw_l1_penalty + adj_l1_penalty# + reg_dagness_loss#  + lagged_dagness_loss + node_dagness_loss# DAGNESS LOSS REMOVED TO ENSURE NUMERICAL STABILITY, ESP. IN D4IC EXPERIMENTS - 12/20/2024
            if factor_cos_sim_penalty is not None:
                combo_loss += factor_cos_sim_penalty
        else:
            combo_loss = forecasting_loss + factor_loss + fw_l1_penalty + adj_l1_penalty# + reg_dagness_loss#  + lagged_dagness_loss + node_dagness_loss# DAGNESS LOSS REMOVED TO ENSURE NUMERICAL STABILITY, ESP. IN D4IC EXPERIMENTS - 12/20/2024
            if factor_cos_sim_penalty is not None:
                combo_loss += factor_cos_sim_penalty
        
        return combo_loss, [forecasting_loss, factor_loss, factor_cos_sim_penalty, fw_l1_penalty, adj_l1_penalty, reg_dagness_loss]

    
    def batch_update(self, epoch_num, batch_num, X, Y, optimizerA, optimizerB, output_length, best_model=None, training_status_of_each_factor=None, 
                     running_factor_score_confusion_matrix=None): 
        # Set up data.
        if torch.cuda.is_available():
            X = X.to(device="cuda")
            Y = Y.to(device="cuda")

        currently_pretraining_embedder = False
        currently_pretraining_factors = False
        currently_training_in_combination = False
        currently_post_training_factors = False
        currently_acclimating_factors = False
        if epoch_num <= self.num_pretrain_epochs-1:
            if "pretrain_embedder" in self.training_mode:
                currently_pretraining_embedder = True
            if "pretrain_factor" in self.training_mode:
                currently_pretraining_factors = True
        elif "acclimate_factors" in self.training_mode and epoch_num <= self.num_pretrain_epochs+self.num_acclimation_epochs-1:
            currently_acclimating_factors = True
        else:
            if "combined" in self.training_mode:
                currently_training_in_combination = True
            elif "post_train_factor" in self.training_mode:
                currently_post_training_factors = True
            else: 
                raise NotImplementedError()
        
        if currently_pretraining_embedder:
            # UPDATE factor score embedder
            # Prep parameters for updates
            self.factor_score_embedder.train()
            for f in self.factors:
                if currently_pretraining_embedder:
                    f.eval()
            optimizerA.zero_grad()
            # make predictions/forecast
            x_sims, _, _, state_label_preds = self.forward(X[:,:max(self.gen_lag, self.embed_lag),:])
            # Calculate loss 
            combined_loss, _ = self.compute_loss(
                X[:,:self.embed_lag,:], 
                x_sims, 
                X[:,max(self.gen_lag, self.embed_lag):max(self.gen_lag, self.embed_lag)+(self.num_sims*output_length),:], 
                state_label_preds, 
                Y, 
                self.primary_gc_est_mode, 
                node_dag_scale=0.1,
                embedder_pretrain_loss=currently_pretraining_embedder, 
                factor_pretrain_loss=False
            )
            combined_loss.backward()
            optimizerA.step()

            if running_factor_score_confusion_matrix is not None: 
                state_label_pred_mask = state_label_preds[0].detach().cpu().numpy()
                Y_mask = None
                if len(Y.size()) == 3:
                    if Y.size()[2] > max(self.gen_lag, self.embed_lag):
                        # we are only interested in the first predicted factor weighting here, so we grab the corresponding label 
                        # (note that Y has size batch_size x num_channels x num_recorded_timesteps)
                        Y_mask = Y[:,:self.num_supervised_factors,max(self.gen_lag, self.embed_lag)].detach().cpu().numpy() 
                    else: # this case arises in the DREAM4-based datasets, for example
                        assert Y.size()[2] == 1
                        Y_mask = Y[:,:self.num_supervised_factors,0].detach().cpu().numpy()
                elif len(Y.size()) == 2: # case for DREAM4 orig. data
                    Y_mask = Y[:,:self.num_supervised_factors].detach().cpu().numpy()
                else:
                    raise NotImplementedError("Cannot handle ground-truth labels with Y.size() == "+str(Y.size()))
                conf_mat_preds = [np.argmax(state_label_pred_mask[i,:]) for i in range(state_label_pred_mask.shape[0])]
                conf_mat_labels = [np.argmax(Y_mask[i,:]) for i in range(Y_mask.shape[0])]
                running_factor_score_confusion_matrix += confusion_matrix(conf_mat_labels, conf_mat_preds, labels=[i for i in range(self.num_supervised_factors)]) 

            del x_sims
            del state_label_preds

        if currently_pretraining_factors or currently_acclimating_factors:
            # UPDATE factor estimates
            # Prep parameters for updates
            self.factor_score_embedder.eval()
            for f in self.factors:
                f.train()
            optimizerB.zero_grad()
            # make predictions/forecast
            x_sims, _, _, state_label_preds = self.forward(X[:,:max(self.gen_lag, self.embed_lag),:], factor_weightings=None)
            # Calculate loss 
            combined_loss, _ = self.compute_loss(
                X[:,:self.embed_lag,:], 
                x_sims, 
                X[:,max(self.gen_lag, self.embed_lag):max(self.gen_lag, self.embed_lag)+(self.num_sims*output_length),:], 
                state_label_preds, 
                Y, 
                self.primary_gc_est_mode, 
                node_dag_scale=0.1,
                embedder_pretrain_loss=False, 
                factor_pretrain_loss=True
            )
            combined_loss.backward()
            optimizerB.step()
            del x_sims
            del state_label_preds
        
        if currently_training_in_combination:
            # UPDATE all parameters
            # Prep parameters for updates
            self.factor_score_embedder.train()
            for f in self.factors:
                if currently_pretraining_embedder:
                    f.train()
            optimizerA.zero_grad()
            optimizerB.zero_grad()
            # make predictions/forecast
            x_sims, _, _, state_label_preds = self.forward(X[:,:max(self.gen_lag, self.embed_lag),:])
            # Calculate loss 
            combined_loss, _ = self.compute_loss(
                X[:,:self.embed_lag,:], 
                x_sims, 
                X[:,max(self.gen_lag, self.embed_lag):max(self.gen_lag, self.embed_lag)+(self.num_sims*output_length),:], 
                state_label_preds, 
                Y, 
                self.primary_gc_est_mode, 
                node_dag_scale=0.1,
                embedder_pretrain_loss=False, 
                factor_pretrain_loss=False
            )
            combined_loss.backward()
            optimizerA.step()
            optimizerB.step()

            if running_factor_score_confusion_matrix is not None: 
                state_label_pred_mask = state_label_preds[0].detach().cpu().numpy()
                Y_mask = None
                if len(Y.size()) == 3:
                    if  Y.size()[2] > max(self.gen_lag, self.embed_lag):
                        # we are only interested in the first predicted factor weighting here, so we grab the corresponding label 
                        # (note that Y has size batch_size x num_channels x num_recorded_timesteps)
                        Y_mask = Y[:,:self.num_supervised_factors,max(self.gen_lag, self.embed_lag)].detach().cpu().numpy() 
                    else: # this case arises in the DREAM4-based datasets, for example
                        assert Y.size()[2] == 1
                        Y_mask = Y[:,:self.num_supervised_factors,0].detach().cpu().numpy()
                elif len(Y.size()) == 2: # case for DREAM4 orig. data
                    Y_mask = Y[:,:self.num_supervised_factors].detach().cpu().numpy()
                else:
                    raise NotImplementedError("Cannot handle ground-truth labels with Y.size() == "+str(Y.size()))
                conf_mat_preds = [np.argmax(state_label_pred_mask[i,:]) for i in range(state_label_pred_mask.shape[0])]
                conf_mat_labels = [np.argmax(Y_mask[i,:]) for i in range(Y_mask.shape[0])]
                running_factor_score_confusion_matrix += confusion_matrix(conf_mat_labels, conf_mat_preds, labels=[i for i in range(self.num_supervised_factors)]) 

            del x_sims
            del state_label_preds
            
        if currently_post_training_factors:
            # UPDATE factor estimates
            # Prep parameters for updates
            self.factor_score_embedder.eval()
            for f in self.factors:
                if currently_pretraining_embedder:
                    f.train()
            optimizerB.zero_grad()

            # make predictions/forecast
            x_sims, _, _, state_label_preds = self.forward(X[:,:max(self.gen_lag, self.embed_lag),:], factor_weightings=None)
            # Calculate loss 
            combined_loss, _ = self.compute_loss(
                X[:,:self.embed_lag,:], 
                x_sims, 
                X[:,max(self.gen_lag, self.embed_lag):max(self.gen_lag, self.embed_lag)+(self.num_sims*output_length),:], 
                state_label_preds, 
                Y, 
                self.primary_gc_est_mode, 
                node_dag_scale=0.1,
                embedder_pretrain_loss=False, 
                factor_pretrain_loss=currently_post_training_factors
            )
            combined_loss.backward()
            optimizerB.step()
            del x_sims
            del state_label_preds
        
        if "FreezeByBatch" in self.training_mode: # determine whether to keep changes made to each factor from current batch
            assert best_model is not None
            assert training_status_of_each_factor is not None
            need_to_change_out_factor = self.determine_which_factors_need_updates(best_model, training_status_of_each_factor)
            update_cached_gen_model_factors = False
            update_cached_factor_score_embedder = False
            for f_ind in range(self.num_factors_nK):
                if training_status_of_each_factor[f_ind]:
                    # check each factor to see if new changes will be accepted / apply the changes
                    if need_to_change_out_factor[f_ind]:
                        best_model.factors[f_ind] = deepcopy(self.factors[f_ind]) # update record of current factor
                        update_cached_gen_model_factors = True
                        update_cached_factor_score_embedder = True
                    else:
                        self.factors[f_ind] = deepcopy(best_model.factors[f_ind]) # revert to prior saved version of current factor
            if update_cached_gen_model_factors:
                best_model.gen_model[1] = best_model.factors
            if update_cached_factor_score_embedder:
                best_model.factor_score_embedder = deepcopy(self.factor_score_embedder) # we always keep the most up-to-date factor score embedder so as to avoid adverse effects on any newly-frozen factors
                best_model.gen_model[0] = best_model.factor_score_embedder

        del X
        del Y
        return best_model, running_factor_score_confusion_matrix 
        pass

    def save_checkpoint(self, save_dir, it, best_model, avg_forecasting_loss, avg_factor_loss, avg_factor_cos_sim_penalty, avg_fw_l1_penalty, 
                        avg_adj_penalty, avg_dagness_reg_loss, avg_dagness_lag_loss, avg_dagness_node_loss, avg_combo_loss, best_loss, best_it, 
                        f1score_histories, f1score_OffDiag_histories, roc_auc_histories, roc_auc_OffDiag_histories, gc_factor_l1_loss_histories, 
                        gc_factor_cosine_sim_histories, gc_factorUnsupervised_cosine_sim_histories, deltacon0_histories, 
                        deltacon0_with_directed_degrees_histories, deltaffinity_histories, path_length_mse_histories, GC, X_vis, output_length=None, 
                        num_sim_steps=None, factor_score_train_acc_history=None, factor_score_train_tpr_history=None, 
                        factor_score_train_tnr_history=None, factor_score_train_fpr_history=None, factor_score_train_fnr_history=None, 
                        factor_score_val_acc_history=None, factor_score_val_tpr_history=None, factor_score_val_tnr_history=None, 
                        factor_score_val_fpr_history=None, factor_score_val_fnr_history=None, ):
        # save summary stats
        temp_model_save_path = os.path.join(save_dir, "final_best_model.bin")
        torch.save(best_model, temp_model_save_path)
        meta_data_save_path = os.path.join(save_dir, "training_meta_data_and_hyper_parameters.pkl")
        with open(meta_data_save_path, "wb") as outfile:
            pkl.dump({
                "epoch": it, 
                "avg_forecasting_loss": avg_forecasting_loss, 
                "avg_factor_loss": avg_factor_loss, 
                "avg_factor_cos_sim_penalty": avg_factor_cos_sim_penalty, 
                "avg_fw_l1_penalty": avg_fw_l1_penalty, 
                "avg_adj_penalty": avg_adj_penalty, 
                "avg_dagness_reg_loss": avg_dagness_reg_loss, 
                "avg_dagness_lag_loss": avg_dagness_lag_loss, 
                "avg_dagness_node_loss": avg_dagness_node_loss, 
                "avg_combo_loss": avg_combo_loss,  
                "best_loss": best_loss, 
                "best_it": best_it, 
                "f1score_histories": f1score_histories, 
                "f1score_OffDiag_histories": f1score_OffDiag_histories, 
                "roc_auc_histories": roc_auc_histories, 
                "roc_auc_OffDiag_histories": roc_auc_OffDiag_histories, 
                "factor_score_train_acc_history": factor_score_train_acc_history, 
                "factor_score_train_tpr_history": factor_score_train_tpr_history, 
                "factor_score_train_tnr_history": factor_score_train_tnr_history, 
                "factor_score_train_fpr_history": factor_score_train_fpr_history, 
                "factor_score_train_fnr_history": factor_score_train_fnr_history, 
                "factor_score_val_acc_history": factor_score_val_acc_history, 
                "factor_score_val_tpr_history": factor_score_val_tpr_history, 
                "factor_score_val_tnr_history": factor_score_val_tnr_history, 
                "factor_score_val_fpr_history": factor_score_val_fpr_history, 
                "factor_score_val_fnr_history": factor_score_val_fnr_history, 
                "gc_factor_l1_loss_histories": gc_factor_l1_loss_histories, 
                "gc_factor_cosine_sim_histories": gc_factor_cosine_sim_histories, 
                "gc_factorUnsupervised_cosine_sim_histories": gc_factorUnsupervised_cosine_sim_histories, 
                "deltacon0_histories": deltacon0_histories, 
                "deltacon0_with_directed_degrees_histories": deltacon0_with_directed_degrees_histories, 
                "deltaffinity_histories": deltaffinity_histories, 
                "path_length_mse_histories": path_length_mse_histories, 
            }, outfile)
        
        # plot loss histories
        plot_curve(
            avg_forecasting_loss, "Validation Forecasting MSE Loss", "Epoch", "Average MSE Loss", 
            save_dir+os.sep+"avg_val_forecasting_mse_loss.png", domain_start=0
        )
        plot_curve(
            avg_factor_loss, "Validation Factor Score MSE Loss", "Epoch", "Average MSE Loss", 
            save_dir+os.sep+"avg_val_factor_score_mse_loss.png", domain_start=0
        )
        plot_curve(
            avg_factor_cos_sim_penalty, "Validation Factor Cosine Similarity Penalty", "Epoch", "Average Penalty", 
            save_dir+os.sep+"avg_factor_cos_sim_penalty.png", domain_start=0
        )
        plot_curve(
            avg_fw_l1_penalty, "Validation Factor Weight L1 Penalty", "Epoch", "Average L1 Penalty", 
            save_dir+os.sep+"avg_val_fw_L1_penalty.png", domain_start=0
        )
        plot_curve(
            avg_adj_penalty, "Validation Adjacency L1 Penalty", "Epoch", "Average L1 Penalty", 
            save_dir+os.sep+"avg_val_adj_L1_penalty.png", domain_start=0
        )
        plot_curve(
            avg_dagness_reg_loss, "Validation Dagness Loss", "Epoch", "Average Dagness REG Loss", 
            save_dir+os.sep+"avg_val_dagness_reg_loss.png", domain_start=0
        )
        plot_curve(
            avg_dagness_lag_loss, "Validation Dagness Loss", "Epoch", "Average Dagness LAG Loss", 
            save_dir+os.sep+"avg_val_dagness_lag_loss.png", domain_start=0
        )
        plot_curve(
            avg_dagness_node_loss, "Validation Dagness Loss", "Epoch", "Average Dagness NODE Loss", 
            save_dir+os.sep+"avg_val_dagness_node_loss.png", domain_start=0
        )
        plot_curve(
            avg_combo_loss, "Validation Combined Loss", "Epoch", "Average Combined Loss", 
            save_dir+os.sep+"avg_val_combo_loss.png", domain_start=0
        )
        for key in f1score_histories.keys():
            key_str = str(key).replace(".","-")
            plot_curve_comparisson(
                f1score_histories[key], "F1 Score History for Threshold="+key_str, "Epoch", "Granger Causal F1 Score", 
                save_dir+os.sep+"f1_score_history_"+key_str+"_visualization.png", domain_start=0, label_root="factor"
            )
            plot_curve_comparisson(
                f1score_OffDiag_histories[key], "F1 Score OffDiag History for Threshold="+key_str, "Epoch", "Granger Causal F1 Score", 
                save_dir+os.sep+"f1_score_OffDiag_history_"+key_str+"_visualization.png", domain_start=0, label_root="factor"
            )
            plot_curve_comparisson(
                roc_auc_histories[key], "ROC-AUC Score History for Threshold="+key_str, "Epoch", "Granger Causal ROC-AUC Score", 
                save_dir+os.sep+"roc_auc_score_history_"+key_str+"_visualization.png", domain_start=0, label_root="factor"
            )
            plot_curve_comparisson(
                roc_auc_OffDiag_histories[key], "ROC-AUC Score OffDiag History for Threshold="+key_str, "Epoch", "Granger Causal ROC-AUC Score", 
                save_dir+os.sep+"roc_auc_score_OffDiag_history_"+key_str+"_visualization.png", domain_start=0, label_root="factor"
            )

        if self.num_supervised_factors > 0:
            plot_curve(
                factor_score_train_acc_history, "Factor Score Training Accuracy History", "Epoch", "Accuracy", 
                save_dir+os.sep+"factor_score_train_acc_history_visualization.png", domain_start=0
            )
            plot_curve(
                factor_score_train_tpr_history, "Factor Score Training TPR History", "Epoch", "TPR", 
                save_dir+os.sep+"factor_score_train_tpr_history_visualization.png", domain_start=0
            )
            plot_curve(
                factor_score_train_tnr_history, "Factor Score Training TNR History", "Epoch", "TNR", 
                save_dir+os.sep+"factor_score_train_tnr_history_visualization.png", domain_start=0
            )
            plot_curve(
                factor_score_train_fpr_history, "Factor Score Training FPR History", "Epoch", "FPR", 
                save_dir+os.sep+"factor_score_train_fpr_history_visualization.png", domain_start=0
            )
            plot_curve(
                factor_score_train_fnr_history, "Factor Score Training FNR History", "Epoch", "FNR", 
                save_dir+os.sep+"factor_score_train_fnr_history_visualization.png", domain_start=0
            )
            plot_curve(
                factor_score_val_acc_history, "Factor Score Validation Accuracy History", "Epoch", "Accuracy", 
                save_dir+os.sep+"factor_score_val_acc_history_visualization.png", domain_start=0
            )
            plot_curve(
                factor_score_val_tpr_history, "Factor Score Validation TPR History", "Epoch", "TPR", 
                save_dir+os.sep+"factor_score_val_tpr_history_visualization.png", domain_start=0
            )
            plot_curve(
                factor_score_val_tnr_history, "Factor Score Validation TNR History", "Epoch", "TNR", 
                save_dir+os.sep+"factor_score_val_tnr_history_visualization.png", domain_start=0
            )
            plot_curve(
                factor_score_val_fpr_history, "Factor Score Validation FPR History", "Epoch", "FPR", 
                save_dir+os.sep+"factor_score_val_fpr_history_visualization.png", domain_start=0
            )
            plot_curve(
                factor_score_val_fnr_history, "Factor Score Validation FNR History", "Epoch", "FNR", 
                save_dir+os.sep+"factor_score_val_fnr_history_visualization.png", domain_start=0
            )
            plot_curve_comparisson(
                [factor_score_val_tpr_history, factor_score_val_tnr_history, factor_score_val_fpr_history, factor_score_val_fnr_history], 
                "Factor Score Confusion Matrix History", "Epoch", "Rate", save_dir+os.sep+"factor_score_val_confMatrix_history_visualization.png", 
                domain_start=0, label_root="[tpr,tnr,fpr,fnr]"
            ) 
        
        plot_curve_comparisson(
            gc_factor_l1_loss_histories, "GC L1 Loss History", "Epoch", "L1 Norm", 
            save_dir+os.sep+"gc_l1_loss_history_visualization.png", domain_start=0, label_root="factor"
        )
        plot_curve_comparisson_from_dict(
            gc_factor_cosine_sim_histories, "GC Cosine Similarity History", "Epoch", "Cosine Similarity", 
            save_dir+os.sep+"gc_factor_cosine_sim_histories_visualization.png", domain_start=0, label_root="factors"
        )
        plot_curve_comparisson_from_dict(
            gc_factorUnsupervised_cosine_sim_histories, "Unsupervised GC Cosine Similarity History", "Epoch", 
            "Cosine Similarity", save_dir+os.sep+"gc_factorUnsupervised_cosine_sim_histories_visualization.png", 
            domain_start=0, label_root="factors"
        )
        
        plot_curve_comparisson(
            deltacon0_histories, "GC DeltaCon0 Similarity", "Epoch", "DeltaCon0 Similarity", 
            save_dir+os.sep+"gc_deltacon0_similarity_history_vis.png", domain_start=0, label_root="factor"
        )
        plot_curve_comparisson(
            deltacon0_with_directed_degrees_histories, "GC DeltaCon0-with-Directed-Degrees Similarity", "Epoch", 
            "DeltaCon0-wDD Similarity", save_dir+os.sep+"gc_deltacon0_wDD_similarity_history_vis.png", domain_start=0, label_root="factor"
        )
        plot_curve_comparisson(
            deltaffinity_histories, "GC Deltaffinity Similarity", "Epoch", "Deltaffinity Similarity", 
            save_dir+os.sep+"gc_deltaffinity_similarity_history_vis.png", domain_start=0, label_root="factor"
        )
        for key in path_length_mse_histories.keys():
            plot_curve_comparisson(
                path_length_mse_histories[key], "GC MSE History for Path-Length="+str(key), "Epoch", "GC MSE", 
                save_dir+os.sep+"gc_mse_score_history_pathLen"+str(key)+"_visualization.png", domain_start=0, label_root="factor"
            )

        for batch_num, (X, _) in enumerate(X_vis):
            # Set up data.
            X = X[:self.MAX_NUM_SAMPS_FOR_GC_VIS,:,:]
            if torch.cuda.is_available():
                X = X.to(device="cuda")
            
            if self.factor_score_embedder is not None:
                self.factor_score_embedder.eval()
            for f in self.factors:
                f.eval()

            # Make Prediction(s)
            x_sims_genUpdate, _, _, state_label_preds = self.forward(X[:,:max(self.gen_lag, self.embed_lag),:])
            # make GC Estimate visualizations
            GC_est = self.GC(
                self.primary_gc_est_mode, 
                X=X[:,:max(self.gen_lag, self.embed_lag),:], 
                threshold=False, 
                ignore_lag=True, 
                combine_wavelet_representations=True, 
                rank_wavelets=False
            )
            GC_noLags = [np.sum(x, axis=2) for x in GC]
            for samp_ind in range(min(len(GC_est), self.MAX_NUM_SAMPS_FOR_GC_VIS)):
                curr_GC_est = [A.cpu().data.numpy() for A in GC_est[samp_ind]]
                plot_gc_est_comparissons_by_factor(
                    GC_noLags, curr_GC_est, save_dir+os.sep+"gc_est_noLags_results_epoch"+str(it)+"_sampInd"+str(samp_ind)+".png"
                )
            
            # make forecasting visualizations
            print("@@@ save_checkpoint: WARNING - FORECASTING VISUALIZATIONS STILL NOT IMPLEMENTED")
            del X
            del x_sims_genUpdate
            del state_label_preds
            del GC_est
            break # only run for one batch
        pass # end of save_checkpoint

    
    def determine_which_factors_need_updates(self, cached_model, training_status_of_each_factor):
        cached_model_gc_ests_NoLag = [x.detach().cpu().numpy() for x in cached_model.GC(
            "fixed_factor_exclusive", X=None, threshold=False, ignore_lag=True, combine_wavelet_representations=False, 
            rank_wavelets=False
        )[0]]
        curr_model_gc_ests_NoLag = [
            x.detach().cpu().numpy() for x in self.GC(
                "fixed_factor_exclusive", X=None, threshold=False, ignore_lag=True, combine_wavelet_representations=False, 
                rank_wavelets=False
            )[0]
        ]
        need_to_change_out_factor = [False for _ in range(self.num_factors_nK)]
        for f_ind in range(self.num_factors_nK):
            if training_status_of_each_factor[f_ind]:
                curr_cached_gcEst = cached_model_gc_ests_NoLag[f_ind] / np.max(cached_model_gc_ests_NoLag[f_ind])
                curr_new_gcEst = curr_model_gc_ests_NoLag[f_ind] / np.max(curr_model_gc_ests_NoLag[f_ind])
                if "withComboCosSimL1" in self.training_mode:
                    avg_cosSim_betw_cached_factors_and_current = 0.
                    avg_cosSim_betw_new_factors_and_current = 0.
                    for other_f_ind in range(self.num_factors_nK):
                        if other_f_ind != f_ind:
                            other_cached_gcEst = cached_model_gc_ests_NoLag[other_f_ind] / np.max(cached_model_gc_ests_NoLag[other_f_ind])
                            other_new_gcEst = curr_model_gc_ests_NoLag[other_f_ind] / np.max(curr_model_gc_ests_NoLag[other_f_ind])
                            avg_cosSim_betw_cached_factors_and_current += compute_cosine_similarity(curr_cached_gcEst, other_cached_gcEst)
                            avg_cosSim_betw_new_factors_and_current += compute_cosine_similarity(curr_new_gcEst, other_new_gcEst)
                    avg_cosSim_betw_cached_factors_and_current /= (self.num_factors_nK-1.)
                    avg_cosSim_betw_new_factors_and_current /= (self.num_factors_nK-1.)
                    curr_cached_gcEst_l1Norm = np.linalg.norm(curr_cached_gcEst, ord=1)
                    curr_new_gcEst_l1Norm = np.linalg.norm(curr_new_gcEst, ord=1)
                    if avg_cosSim_betw_new_factors_and_current*curr_new_gcEst_l1Norm < avg_cosSim_betw_cached_factors_and_current*curr_cached_gcEst_l1Norm:
                        need_to_change_out_factor[f_ind] = True
                elif "withL1" in self.training_mode:
                    curr_cached_gcEst_l1Norm = np.linalg.norm(curr_cached_gcEst, ord=1)
                    curr_new_gcEst_l1Norm = np.linalg.norm(curr_new_gcEst, ord=1)
                    if curr_new_gcEst_l1Norm < curr_cached_gcEst_l1Norm:
                        need_to_change_out_factor[f_ind] = True
                else:
                    raise NotImplementedError()
            else:
                pass # no need to change out current factor
        return need_to_change_out_factor

    
    def fit(self, save_dir, X_train, optimizerA, optimizerB, input_length, output_length, num_sim_steps, max_iter, X_val, lookback=5, 
            check_every=50, verbose=1, GC=None, deltaConEps=0.1, in_degree_coeff=1., out_degree_coeff=1., prior_factors_path=None, 
            cost_criteria="CosineSimilarity", unsupervised_start_index=0, max_factor_prior_batches=10, stopping_criteria_forecast_coeff=1., 
            stopping_criteria_factor_coeff=1., stopping_criteria_cosSim_coeff=1.):
        print("REDCLIFF_S_CMLP.fit: START", flush=True)
        print("REDCLIFF_S_CMLP.fit: WARNING - NOT USING THE input_length ARGUMENT (FROM A DECRIMENTED IMPLEMENTATION), AND INSTEAD USING gen_lag AND embed_lag ARGUMENTS PASSED TO __init__ !!!", flush=True)
        input_length = None # set to None to ensure it is not used in updated implementation (as of 04/24/2024)
        f1_thresholds = [0.0]
        training_status_of_each_factor = None
        if "Freeze" in self.training_mode:
            training_status_of_each_factor = [True for _ in range(self.num_factors_nK)]
            
        # For tracking intermediate/preliminary results
        avg_forecasting_loss = []
        avg_factor_loss = []
        avg_factor_cos_sim_penalty = []
        avg_fw_l1_penalty = []
        avg_adj_penalty = []
        avg_dagness_reg_loss = []
        avg_dagness_lag_loss = []
        avg_dagness_node_loss = []
        avg_combo_loss = []
        f1score_histories = {thresh:[[] for _ in range(self.num_supervised_factors)] for thresh in f1_thresholds}
        f1score_OffDiag_histories = {thresh:[[] for _ in range(self.num_supervised_factors)] for thresh in f1_thresholds}
        roc_auc_histories = {thresh:[[] for _ in range(self.num_supervised_factors)] for thresh in f1_thresholds}
        roc_auc_OffDiag_histories = {thresh:[[] for _ in range(self.num_supervised_factors)] for thresh in f1_thresholds}
        if self.num_supervised_factors > 0:
            factor_score_train_acc_history = []
            factor_score_train_tpr_history = []
            factor_score_train_tnr_history = []
            factor_score_train_fpr_history = []
            factor_score_train_fnr_history = []
            factor_score_val_acc_history = []
            factor_score_val_tpr_history = []
            factor_score_val_tnr_history = []
            factor_score_val_fpr_history = []
            factor_score_val_fnr_history = []
        gc_factor_l1_loss_histories = [[] for _ in range(self.num_supervised_factors)]
        gc_factor_cosine_sim_histories = {
            str(i)+"and"+str(j):[] for i in range(self.num_supervised_factors) for j in range(self.num_supervised_factors) if i < j
        }
        gc_factorUnsupervised_cosine_sim_histories = {
            str(i)+"and"+str(j):[] for i in range(self.num_supervised_factors, self.num_factors_nK) for j in range(self.num_supervised_factors, self.num_factors_nK) if i < j
        }
        deltacon0_histories = [[] for _ in range(self.num_supervised_factors)]
        deltacon0_with_directed_degrees_histories = [[] for _ in range(self.num_supervised_factors)]
        deltaffinity_histories = [[] for _ in range(self.num_supervised_factors)]
        path_length_mse_histories = {
            path_length:[[] for _ in range(self.num_supervised_factors)] for path_length in range(1,self.num_chans)
        }

        # For early stopping.
        best_it = None
        best_loss = np.inf
        best_model = None
        iter_start = 0

        RESUMING_TRAININIG = False
        try:
            best_model = deepcopy(self)
            iter_start = self.chkpt_best_it+1
            RESUMING_TRAININIG = True
            avg_forecasting_loss = self.chkpt_avg_forecasting_loss[:iter_start]
            avg_factor_loss = self.chkpt_avg_factor_loss[:iter_start]
            avg_factor_cos_sim_penalty = self.chkpt_avg_factor_cos_sim_penalty[:iter_start]
            avg_fw_l1_penalty = self.chkpt_avg_fw_l1_penalty[:iter_start]
            avg_adj_penalty = self.chkpt_avg_adj_penalty[:iter_start]
            avg_dagness_reg_loss = self.chkpt_avg_dagness_reg_loss[:iter_start]
            avg_dagness_lag_loss = self.chkpt_avg_dagness_lag_loss[:iter_start]
            avg_dagness_node_loss = self.chkpt_avg_dagness_node_loss[:iter_start]
            avg_combo_loss = self.chkpt_avg_combo_loss[:iter_start]
            best_loss = self.chkpt_best_loss
            best_it = self.chkpt_best_it
            for thresh in f1_thresholds:
                for sf in range(self.num_supervised_factors):
                    f1score_histories[thresh][sf] = self.chkpt_f1score_histories[thresh][sf][:iter_start]
                    f1score_OffDiag_histories[thresh][sf] = self.chkpt_f1score_OffDiag_histories[thresh][sf][:iter_start]
                    roc_auc_histories[thresh][sf] = self.chkpt_roc_auc_histories[thresh][sf][:iter_start]
                    roc_auc_OffDiag_histories[thresh][sf] = self.chkpt_roc_au_OffDiagc_histories[thresh][sf][:iter_start]
            if self.num_supervised_factors > 0:
                factor_score_train_acc_history = self.chkpt_factor_score_train_acc_history
                factor_score_train_tpr_history = self.chkpt_factor_score_train_tpr_history
                factor_score_train_tnr_history = self.chkpt_factor_score_train_tnr_history
                factor_score_train_fpr_history = self.chkpt_factor_score_train_fpr_history
                factor_score_train_fnr_history = self.chkpt_factor_score_train_fnr_history
                factor_score_val_acc_history = self.chkpt_factor_score_val_acc_history
                factor_score_val_tpr_history = self.chkpt_factor_score_val_tpr_history
                factor_score_val_tnr_history = self.chkpt_factor_score_val_tnr_history
                factor_score_val_fpr_history = self.chkpt_factor_score_val_fpr_history
                factor_score_val_fnr_history = self.chkpt_factor_score_val_fnr_history
            for key in gc_factor_cosine_sim_histories.keys():
                gc_factor_cosine_sim_histories[key] = self.chkpt_gc_factor_cosine_sim_histories[key][:iter_start]
            for key in gc_factorUnsupervised_cosine_sim_histories.keys():
                gc_factorUnsupervised_cosine_sim_histories[key] = self.chkpt_gc_factorUnsupervised_cosine_sim_histories[key][:iter_start]
            for sf in range(self.num_supervised_factors):
                gc_factor_l1_loss_histories[sf] = self.chkpt_gc_factor_l1_loss_histories[sf][:iter_start]
                deltacon0_histories[sf] = self.chkpt_deltacon0_histories[sf][:iter_start]
                deltacon0_with_directed_degrees_histories[sf] = self.chkpt_deltacon0_with_directed_degrees_histories[sf][:iter_start]
                deltaffinity_histories[sf] = self.chkpt_deltaffinity_histories[sf][:iter_start]
            for key in path_length_mse_histories.keys():
                for sf in range(self.num_supervised_factors):
                    path_length_mse_histories[key][sf] = self.chkpt_path_length_mse_histories[key][sf][:iter_start]

            print("REDCLIFF_S_CMLP.fit: RESUMING TRAINING WITH THE FOLLOWING CHKPT CONFIGURATION: ")
        except:
            if RESUMING_TRAININIG:
                print("REDCLIFF_S_CMLP.fit: WARNING - EXCEPTION ENCOUNTERED WHEN TRYNING TO RESUME TRAINING, WITH CHKPT PARTIALLY LOADED! THE CURRENT CHKPT CONFIGURATION IS: ")
            pass
        print("REDCLIFF_S_CMLP.fit: \t\t iter_start == ", iter_start)
        print("REDCLIFF_S_CMLP.fit: \t\t avg_forecasting_loss == ", avg_forecasting_loss)
        print("REDCLIFF_S_CMLP.fit: \t\t avg_factor_loss == ", avg_factor_loss)
        print("REDCLIFF_S_CMLP.fit: \t\t avg_factor_cos_sim_penalty == ", avg_factor_cos_sim_penalty)
        print("REDCLIFF_S_CMLP.fit: \t\t avg_fw_l1_penalty == ", avg_fw_l1_penalty)
        print("REDCLIFF_S_CMLP.fit: \t\t avg_adj_penalty == ", avg_adj_penalty)
        print("REDCLIFF_S_CMLP.fit: \t\t avg_dagness_reg_loss == ", avg_dagness_reg_loss)
        print("REDCLIFF_S_CMLP.fit: \t\t avg_dagness_lag_loss == ", avg_dagness_lag_loss)
        print("REDCLIFF_S_CMLP.fit: \t\t avg_dagness_node_loss == ", avg_dagness_node_loss)
        print("REDCLIFF_S_CMLP.fit: \t\t avg_combo_loss == ", avg_combo_loss)
        print("REDCLIFF_S_CMLP.fit: \t\t best_loss == ", best_loss)
        print("REDCLIFF_S_CMLP.fit: \t\t best_it == ", best_it)
        print("REDCLIFF_S_CMLP.fit: \t\t f1score_histories == ", f1score_histories)
        print("REDCLIFF_S_CMLP.fit: \t\t f1score_OffDiag_histories == ", f1score_OffDiag_histories)
        print("REDCLIFF_S_CMLP.fit: \t\t roc_auc_histories == ", roc_auc_histories)
        print("REDCLIFF_S_CMLP.fit: \t\t roc_auc_OffDiag_histories == ", roc_auc_OffDiag_histories)
        if self.num_supervised_factors > 0:
            print("REDCLIFF_S_CMLP.fit: \t\t factor_score_train_acc_history == ", factor_score_train_acc_history)
            print("REDCLIFF_S_CMLP.fit: \t\t factor_score_train_tpr_history == ", factor_score_train_tpr_history)
            print("REDCLIFF_S_CMLP.fit: \t\t factor_score_train_tnr_history == ", factor_score_train_tnr_history)
            print("REDCLIFF_S_CMLP.fit: \t\t factor_score_train_fpr_history == ", factor_score_train_fpr_history)
            print("REDCLIFF_S_CMLP.fit: \t\t factor_score_train_fnr_history == ", factor_score_train_fnr_history)
            print("REDCLIFF_S_CMLP.fit: \t\t factor_score_val_acc_history == ", factor_score_val_acc_history)
            print("REDCLIFF_S_CMLP.fit: \t\t factor_score_val_tpr_history == ", factor_score_val_tpr_history)
            print("REDCLIFF_S_CMLP.fit: \t\t factor_score_val_tnr_history == ", factor_score_val_tnr_history)
            print("REDCLIFF_S_CMLP.fit: \t\t factor_score_val_fpr_history == ", factor_score_val_fpr_history)
            print("REDCLIFF_S_CMLP.fit: \t\t factor_score_val_fnr_history == ", factor_score_val_fnr_history)
        print("REDCLIFF_S_CMLP.fit: \t\t gc_factor_l1_loss_histories == ", gc_factor_l1_loss_histories)
        print("REDCLIFF_S_CMLP.fit: \t\t gc_factor_cosine_sim_histories == ", gc_factor_cosine_sim_histories)
        print("REDCLIFF_S_CMLP.fit: \t\t gc_factorUnsupervised_cosine_sim_histories == ", gc_factorUnsupervised_cosine_sim_histories)
        print("REDCLIFF_S_CMLP.fit: \t\t deltacon0_histories == ", deltacon0_histories)
        print("REDCLIFF_S_CMLP.fit: \t\t deltacon0_with_directed_degrees_histories == ", deltacon0_with_directed_degrees_histories)
        print("REDCLIFF_S_CMLP.fit: \t\t deltaffinity_histories == ", deltaffinity_histories)
        print("REDCLIFF_S_CMLP.fit: \t\t path_length_mse_histories == ", path_length_mse_histories, flush=True)

        for it in range(iter_start, max_iter):
            print("REDCLIFF_S_CMLP.fit: now on epoch it == ", it, flush=True)            
            if (it == self.num_pretrain_epochs and "pretrain_factor" in self.training_mode) or (prior_factors_path is not None and it == 0): # end of pretraining
                print("REDCLIFF_S_CMLP.fit: <<< NOW USING PRIOR TO INITIALIZE/RESET FACTORS >>>", flush=True)
                self.initialize_factors_with_prior(
                    prior_factors_path=prior_factors_path, X_train=X_train, cost_criteria=cost_criteria, 
                    unsupervised_start_index=unsupervised_start_index, max_batches=max_factor_prior_batches, 
                )

            running_factor_score_confusion_matrix = None 
            if self.num_supervised_factors > 0:
                running_factor_score_confusion_matrix = np.zeros((self.num_supervised_factors, self.num_supervised_factors))
            for batch_num, (X, Y) in enumerate(X_train):
                if "FreezeByBatch" not in self.training_mode:
                    _, running_factor_score_confusion_matrix = self.batch_update(
                        it, batch_num, X, Y, optimizerA, optimizerB, output_length, best_model=None, training_status_of_each_factor=None, 
                        running_factor_score_confusion_matrix=running_factor_score_confusion_matrix 
                    )
                else:
                    best_model, running_factor_score_confusion_matrix = self.batch_update(
                        it, batch_num, X, Y, optimizerA, optimizerB, output_length, best_model=best_model, 
                        training_status_of_each_factor=training_status_of_each_factor, 
                        running_factor_score_confusion_matrix=running_factor_score_confusion_matrix
                    )
                
            if self.num_supervised_factors > 0:
                # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html and https://stackoverflow.com/questions/45053238/how-to-get-all-confusion-matrix-terminologies-tpr-fpr-tnr-fnr-for-a-multi-c
                TP = np.diag(running_factor_score_confusion_matrix)
                FP = running_factor_score_confusion_matrix.sum(axis=0) - TP
                FN = running_factor_score_confusion_matrix.sum(axis=1) - TP
                TN = running_factor_score_confusion_matrix.sum() - (FP + FN + TP)
                TP = TP.astype(float)
                FP = FP.astype(float)
                FN = FN.astype(float)
                TN = TN.astype(float)
                TPR = TP/(TP+FN) # Sensitivity, hit rate, recall, or true positive rate
                TNR = TN/(TN+FP) # Specificity or true negative rate
                FPR = FP/(FP+TN) # Fall out or false positive rate
                FNR = FN/(TP+FN) # False negative rate
                ACC = (TP+TN)/(TP+FP+FN+TN) # Overall accuracy
                factor_score_train_acc_history.append(ACC)
                factor_score_train_tpr_history.append(TPR)
                factor_score_train_tnr_history.append(TNR)
                factor_score_train_fpr_history.append(FPR)
                factor_score_train_fnr_history.append(FNR)

            # monitor/track GC development progress
            for batch_num, (X, _) in enumerate(X_val):
                # Set up data.
                X = X[:self.MAX_NUM_SAMPS_FOR_GC_PROGRESS_TRACKING,:max(self.gen_lag, self.embed_lag),:]
                if torch.cuda.is_available():
                    X = X.to(device="cuda")

                if self.factor_score_embedder is not None:
                    self.factor_score_embedder.eval()
                for f in self.factors:
                    f.eval()
                
                CURR_GC_EST = self.GC(
                    self.primary_gc_est_mode, X=X, threshold=False, ignore_lag=False, combine_wavelet_representations=False, rank_wavelets=False
                )[:self.num_supervised_factors]
                finiteness_of_each_gc_est = [torch.all(torch.isfinite(elem.data)) for gc_est in CURR_GC_EST for elem in gc_est]
                if not sum(finiteness_of_each_gc_est) == len(finiteness_of_each_gc_est):
                    print("DEBUGGING - it == ", it)
                    print("DEBUGGING - batch_num == ", batch_num)
                    print("DEBUGGING - NON-FINITE VALUE(S) DETECTED IN CURR_GC_EST")
                    print("DEBUGGING - CURR_GC_EST == ", CURR_GC_EST)
                f1score_histories, roc_auc_histories = track_receiver_operating_characteristic_stats_for_redcliff_models(
                    GC, CURR_GC_EST, f1score_histories, roc_auc_histories, remove_self_connections=False
                )
                f1score_OffDiag_histories, roc_auc_OffDiag_histories = track_receiver_operating_characteristic_stats_for_redcliff_models(
                    GC, CURR_GC_EST, f1score_OffDiag_histories, roc_auc_OffDiag_histories, remove_self_connections=True
                )
                deltacon0_histories, \
                deltacon0_with_directed_degrees_histories, \
                deltaffinity_histories, \
                path_length_mse_histories = track_deltacon0_related_stats_for_redcliff_models(
                    GC, CURR_GC_EST, self.num_chans, deltacon0_histories, deltacon0_with_directed_degrees_histories, 
                    deltaffinity_histories, path_length_mse_histories, deltaConEps=deltaConEps, in_degree_coeff=in_degree_coeff, 
                    out_degree_coeff=out_degree_coeff, remove_self_connections=False
                )
                curr_l1_loss, gc_factor_l1_loss_histories = track_l1_norm_stats_of_gc_ests_from_redcliff_models(CURR_GC_EST, gc_factor_l1_loss_histories)
                
                curr_gc_noLag_est = [
                    [x.detach().cpu().numpy() for x in sample_ests][:self.num_supervised_factors] for sample_ests in self.GC(
                        self.primary_gc_est_mode, X=X, threshold=False, ignore_lag=True, combine_wavelet_representations=True, rank_wavelets=False
                    )
                ]
                gc_factor_cosine_sim_histories = track_cosine_similarity_stats_of_gc_ests_from_redcliff_models(
                    curr_gc_noLag_est, gc_factor_cosine_sim_histories, label_offset=0
                )
                curr_unsup_gc_noLag_est = [
                    [x.detach().cpu().numpy() for x in sample_ests][self.num_supervised_factors:] for sample_ests in self.GC(
                        self.primary_gc_est_mode, X=X, threshold=False, ignore_lag=True, combine_wavelet_representations=True, rank_wavelets=False
                    )
                ]
                gc_factorUnsupervised_cosine_sim_histories = track_cosine_similarity_stats_of_gc_ests_from_redcliff_models(
                    curr_unsup_gc_noLag_est, gc_factorUnsupervised_cosine_sim_histories, label_offset=self.num_supervised_factors
                )
            
                del X
                break # only use the first batch for tracking GC progress (only applies when primary_gc_est_mode is conditional)
                
            # track validation stats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            avg_val_forecasting_loss = []
            avg_val_factor_loss = []
            avg_val_factor_cos_sim_penalty = []
            avg_val_fw_l1_penalty = []
            avg_val_adj_penalty = []
            avg_val_dagness_reg_loss = []
            avg_dagness_lag_loss = []
            avg_dagness_node_loss = []
            avg_val_combo_loss = []
            factor_score_val_acc_history = []
            factor_score_val_tpr_history = []
            factor_score_val_tnr_history = []
            factor_score_val_fpr_history = []
            factor_score_val_fnr_history = []
            if self.num_supervised_factors > 0:
                avg_val_forecasting_loss, \
                avg_val_factor_loss, \
                avg_val_factor_cos_sim_penalty, \
                avg_val_fw_l1_penalty, \
                avg_val_adj_penalty, \
                avg_val_dagness_reg_loss, \
                avg_val_dagness_lag_loss, \
                avg_val_dagness_node_loss, \
                avg_val_combo_loss, \
                factor_score_val_acc_history, \
                factor_score_val_tpr_history, \
                factor_score_val_tnr_history, \
                factor_score_val_fpr_history, \
                factor_score_val_fnr_history = self.validate_training(
                    X_val, output_length, self.num_series, 
                    factor_score_val_acc_history=factor_score_val_acc_history, 
                    factor_score_val_tpr_history=factor_score_val_tpr_history, 
                    factor_score_val_tnr_history=factor_score_val_tnr_history, 
                    factor_score_val_fpr_history=factor_score_val_fpr_history, 
                    factor_score_val_fnr_history=factor_score_val_fnr_history 
                )
            else:
                avg_val_forecasting_loss, \
                avg_val_factor_loss, \
                avg_val_factor_cos_sim_penalty, \
                avg_val_fw_l1_penalty, \
                avg_val_adj_penalty, \
                avg_val_dagness_reg_loss, \
                avg_val_dagness_lag_loss, \
                avg_val_dagness_node_loss, \
                avg_val_combo_loss = self.validate_training(X_val, output_length, self.num_series)
            avg_forecasting_loss.append(avg_val_forecasting_loss)
            avg_factor_loss.append(avg_val_factor_loss)
            avg_factor_cos_sim_penalty.append(avg_val_factor_cos_sim_penalty)
            avg_fw_l1_penalty.append(avg_val_fw_l1_penalty)
            avg_adj_penalty.append(avg_val_adj_penalty)
            avg_dagness_reg_loss.append(avg_val_dagness_reg_loss)
            avg_dagness_lag_loss.append(avg_val_dagness_lag_loss)
            avg_dagness_node_loss.append(avg_val_dagness_node_loss)
            avg_combo_loss.append(avg_val_combo_loss)
            
            # Check for early stopping.
            if it >= self.num_pretrain_epochs+self.num_acclimation_epochs:
                curr_cosSim_mean = np.mean([gc_factor_cosine_sim_histories[key][-1] for key in gc_factor_cosine_sim_histories.keys()]) # this will be reported as part of the stopping criteria evaluation(s)
                
                if "Freeze" in self.training_mode:
                    assert best_model is not None
                    need_to_change_out_factor = self.determine_which_factors_need_updates(best_model, training_status_of_each_factor)
                    print("REDCLIFF_S_CMLP.fit: need_to_change_out_factor == ", need_to_change_out_factor)
                    print("REDCLIFF_S_CMLP.fit: orig training_status_of_each_factor == ", training_status_of_each_factor)
                    
                    update_cached_gen_model_factors = False
                    update_cached_factor_score_embedder = False
                    for f_ind in range(self.num_factors_nK):
                        if "Epoch" in self.training_mode and training_status_of_each_factor[f_ind]:
                            # check each factor to see if new changes will be accepted / apply the changes, then update the training status of each factor
                            if need_to_change_out_factor[f_ind]:
                                print("REDCLIFF_S_CMLP.fit: recording improved version of factor ", f_ind)
                                best_model.factors[f_ind] = deepcopy(self.factors[f_ind]) # update record of current factor
                                update_cached_gen_model_factors = True
                                update_cached_factor_score_embedder = True
                            else:
                                print("REDCLIFF_S_CMLP.fit: reverting to previous version of factor ", f_ind)
                                self.factors[f_ind] = deepcopy(best_model.factors[f_ind]) # revert to prior saved version of current factor
                        if not need_to_change_out_factor[f_ind] and training_status_of_each_factor[f_ind]: # stop training / freeze a factor if it hasn't improved over the course of the last epoch
                            pass # training_status_of_each_factor[f_ind] = False # FOR DEBUGGING
                    if update_cached_gen_model_factors:
                        best_model.gen_model[1] = best_model.factors
                    if update_cached_factor_score_embedder:
                        best_model.factor_score_embedder = deepcopy(self.factor_score_embedder) # we always keep the most up-to-date factor score embedder so as to avoid adverse effects on any newly-frozen factors
                        best_model.gen_model[0] = best_model.factor_score_embedder
                    print("REDCLIFF_S_CMLP.fit: new training_status_of_each_factor == ", training_status_of_each_factor)
                    
                    # track Non-Frozen stopping criteria for comparisson
                    curr_criteria_val = None
                    if self.num_supervised_factors > 0:
                        if self.num_supervised_factors > 1:
                            curr_criteria_val = (stopping_criteria_factor_coeff*avg_val_factor_loss)+(stopping_criteria_forecast_coeff*avg_val_forecasting_loss)+(stopping_criteria_cosSim_coeff*curr_cosSim_mean)
                        else:
                            curr_criteria_val = (stopping_criteria_factor_coeff*avg_val_factor_loss)+(stopping_criteria_forecast_coeff*avg_val_forecasting_loss)
                    else:
                        curr_criteria_val = stopping_criteria_forecast_coeff*avg_val_forecasting_loss
                    print("REDCLIFF_S_CMLP.fit: NON-FROZEN curr_criteria_val == ", curr_criteria_val, flush=True)
                    
                    # apply "Freeze" training mode stopping criteria
                    if (sum(training_status_of_each_factor) > 0) or curr_criteria_val < best_loss:
                        best_loss = curr_criteria_val
                        best_it = it
                    else:
                        if verbose:
                            print('Stopping early')
                        break
                else:
                    print("REDCLIFF_S_CMLP.fit: avg_val_factor_loss == ", avg_val_factor_loss)
                    print("REDCLIFF_S_CMLP.fit: avg_val_forecasting_loss == ", avg_val_forecasting_loss)
                    print("REDCLIFF_S_CMLP.fit: curr_cosSim_mean == ", curr_cosSim_mean)
                    print("REDCLIFF_S_CMLP.fit: it == ", it)
                    print("REDCLIFF_S_CMLP.fit: best_it == ", best_it, flush=True)
                    curr_criteria_val = None
                    if self.num_supervised_factors > 0:
                        if self.num_supervised_factors > 1:
                            curr_criteria_val = (stopping_criteria_factor_coeff*avg_val_factor_loss)+(stopping_criteria_forecast_coeff*avg_val_forecasting_loss)+(stopping_criteria_cosSim_coeff*curr_cosSim_mean)
                        else:
                            curr_criteria_val = (stopping_criteria_factor_coeff*avg_val_factor_loss)+(stopping_criteria_forecast_coeff*avg_val_forecasting_loss)
                    else:
                        curr_criteria_val = stopping_criteria_forecast_coeff*avg_val_forecasting_loss
                    print("REDCLIFF_S_CMLP.fit: curr_criteria_val == ", curr_criteria_val, flush=True)
                    if curr_criteria_val < best_loss:
                        best_loss = curr_criteria_val
                        best_it = it
                        best_model = deepcopy(self)
                    elif (it - best_it) == lookback * check_every:
                        if verbose:
                            print('Stopping early')
                        break
            else:
                print("REDCLIFF_S_CMLP.fit: \t WRAPPING UP PRE-TRAINING EPOCH ", it, "; STOPPING CRITERION WILL NOT BE TRACKED/EVALUATED UNTIL PRE-TRAINING IS COMPLETE, BUT THE MODEL/EPOCH WILL.", flush=True)
                best_it = it
                best_model = deepcopy(self)

            # Check progress.
            if it % check_every == 0:
                if verbose > 0:
                    print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                    print('Validation Loss = %f' % avg_val_combo_loss)
                print("REDCLIFF_S_CMLP.fit: \t CHECKING")
                print("REDCLIFF_S_CMLP.fit: \t avg_forecasting_loss == ", avg_forecasting_loss)
                print("REDCLIFF_S_CMLP.fit: \t avg_factor_loss == ", avg_factor_loss)
                print("REDCLIFF_S_CMLP.fit: \t avg_factor_cos_sim_penalty == ", avg_factor_cos_sim_penalty)
                print("REDCLIFF_S_CMLP.fit: \t avg_fw_l1_penalty == ", avg_fw_l1_penalty)
                print("REDCLIFF_S_CMLP.fit: \t avg_adj_penalty == ", avg_adj_penalty)
                print("REDCLIFF_S_CMLP.fit: \t avg_dagness_reg_loss == ", avg_dagness_reg_loss)
                print("REDCLIFF_S_CMLP.fit: \t avg_dagness_lag_loss == ", avg_dagness_lag_loss)
                print("REDCLIFF_S_CMLP.fit: \t avg_dagness_node_loss == ", avg_dagness_node_loss)
                print("REDCLIFF_S_CMLP.fit: \t avg_combo_loss == ", avg_combo_loss)
                if self.num_supervised_factors > 0:
                    print("REDCLIFF_S_CMLP.fit: \t factor_score_train_acc_history == ", factor_score_train_acc_history)
                    print("REDCLIFF_S_CMLP.fit: \t factor_score_train_tpr_history == ", factor_score_train_tpr_history)
                    print("REDCLIFF_S_CMLP.fit: \t factor_score_train_tnr_history == ", factor_score_train_tnr_history)
                    print("REDCLIFF_S_CMLP.fit: \t factor_score_train_fpr_history == ", factor_score_train_fpr_history)
                    print("REDCLIFF_S_CMLP.fit: \t factor_score_train_fnr_history == ", factor_score_train_fnr_history)
                    print("REDCLIFF_S_CMLP.fit: \t factor_score_val_acc_history == ", factor_score_val_acc_history)
                    print("REDCLIFF_S_CMLP.fit: \t factor_score_val_tpr_history == ", factor_score_val_tpr_history)
                    print("REDCLIFF_S_CMLP.fit: \t factor_score_val_tnr_history == ", factor_score_val_tnr_history)
                    print("REDCLIFF_S_CMLP.fit: \t factor_score_val_fpr_history == ", factor_score_val_fpr_history)
                    print("REDCLIFF_S_CMLP.fit: \t factor_score_val_fnr_history == ", factor_score_val_fnr_history)
                    
                # save checkpoint
                if self.num_supervised_factors > 0:
                    self.save_checkpoint(
                        save_dir, it, best_model, avg_forecasting_loss, avg_factor_loss, avg_factor_cos_sim_penalty, avg_fw_l1_penalty, 
                        avg_adj_penalty, avg_dagness_reg_loss, avg_dagness_lag_loss, avg_dagness_node_loss, avg_combo_loss, best_loss, 
                        best_it, f1score_histories, f1score_OffDiag_histories, roc_auc_histories, roc_auc_OffDiag_histories, 
                        gc_factor_l1_loss_histories, gc_factor_cosine_sim_histories, gc_factorUnsupervised_cosine_sim_histories, 
                        deltacon0_histories, deltacon0_with_directed_degrees_histories, deltaffinity_histories, path_length_mse_histories, 
                        GC, X_val, output_length=None, num_sim_steps=None, factor_score_train_acc_history=factor_score_train_acc_history, 
                        factor_score_train_tpr_history=factor_score_train_tpr_history, 
                        factor_score_train_tnr_history=factor_score_train_tnr_history, 
                        factor_score_train_fpr_history=factor_score_train_fpr_history, 
                        factor_score_train_fnr_history=factor_score_train_fnr_history, 
                        factor_score_val_acc_history=factor_score_val_acc_history, 
                        factor_score_val_tpr_history=factor_score_val_tpr_history, 
                        factor_score_val_tnr_history=factor_score_val_tnr_history, 
                        factor_score_val_fpr_history=factor_score_val_fpr_history, 
                        factor_score_val_fnr_history=factor_score_val_fnr_history, 
                    )
                else:
                    self.save_checkpoint(
                        save_dir, it, best_model, avg_forecasting_loss, avg_factor_loss, avg_factor_cos_sim_penalty, avg_fw_l1_penalty, 
                        avg_adj_penalty, avg_dagness_reg_loss, avg_dagness_lag_loss, avg_dagness_node_loss, avg_combo_loss, best_loss, 
                        best_it, f1score_histories, f1score_OffDiag_histories, roc_auc_histories, roc_auc_OffDiag_histories, 
                        gc_factor_l1_loss_histories, gc_factor_cosine_sim_histories, gc_factorUnsupervised_cosine_sim_histories, 
                        deltacon0_histories, deltacon0_with_directed_degrees_histories, deltaffinity_histories, path_length_mse_histories, 
                        GC, X_val, output_length=None, num_sim_steps=None 
                    )
            pass

        # Restore best model.
        restore_parameters(self, best_model)
        final_save_path = os.path.join(save_dir, "final_best_model.bin")
        torch.save(self, final_save_path)

        # Report final Validation Score(s)
        final_mean_val_combo_loss = None
        if self.num_supervised_factors > 0:
            _, _, _, _, _, _, _, _, \
            final_mean_val_combo_loss, \
            factor_score_val_acc_history, \
            factor_score_val_tpr_history, \
            factor_score_val_tnr_history, \
            factor_score_val_fpr_history, \
            factor_score_val_fnr_history = self.validate_training( 
                X_val, output_length, self.num_series, 
                factor_score_val_acc_history=factor_score_val_acc_history, 
                factor_score_val_tpr_history=factor_score_val_tpr_history, 
                factor_score_val_tnr_history=factor_score_val_tnr_history, 
                factor_score_val_fpr_history=factor_score_val_fpr_history, 
                factor_score_val_fnr_history=factor_score_val_fnr_history 
            )
        else:
            _, _, _, _, _, _, _, _, final_mean_val_combo_loss = self.validate_training(X_val, output_length, self.num_series)
        print("FINAL BEST (STOPPING CRITERIA) LOSS == ", best_loss, flush=True)
        print("FINAL BEST (STOPPING CRITERIA) EPOCH == ", best_it, flush=True)
        print("FINAL VALIDATION COMBO LOSS == ", final_mean_val_combo_loss, flush=True)
        return final_mean_val_combo_loss

    
    def validate_training(self, X_val, output_length, num_series, factor_score_val_acc_history=None, factor_score_val_tpr_history=None, 
                          factor_score_val_tnr_history=None, factor_score_val_fpr_history=None, factor_score_val_fnr_history=None):
        # initialize vars for tracking intermediate/preliminary results
        avg_forecasting_loss = 0.
        avg_factor_loss = 0.
        avg_factor_cos_sim_penalty = 0.
        avg_fw_l1_penalty = 0.
        avg_adj_penalty = 0.
        avg_dagness_reg_loss = 0.
        avg_dagness_lag_loss = 0.
        avg_dagness_node_loss = 0.
        avg_combo_loss = 0.

        running_factor_score_confusion_matrix = None
        if self.num_supervised_factors > 0:
            running_factor_score_confusion_matrix = np.zeros((self.num_supervised_factors, self.num_supervised_factors))

        for batch_num, (X, Y) in enumerate(X_val):
            # Set up data.
            if torch.cuda.is_available():
                X = X.to(device="cuda")
                Y = Y.to(device="cuda")
            
            if self.factor_score_embedder is not None:
                self.factor_score_embedder.eval()
            for f in self.factors:
                f.eval()

            # Make Prediction(s)
            x_sims_genUpdate, _, _, state_label_preds = self.forward(X[:,:max(self.gen_lag, self.embed_lag),:])
            # Calculate loss 
            combined_loss, [forecasting_loss, factor_loss, factor_cos_sim_penalty, fw_l1_penalty, adj_l1_penalty, reg_dagness_loss] = self.compute_loss(
                X[:,:self.embed_lag,:], 
                x_sims_genUpdate, 
                X[:,max(self.gen_lag, self.embed_lag):max(self.gen_lag, self.embed_lag)+(self.num_sims*output_length),:],
                state_label_preds,
                Y, 
                self.primary_gc_est_mode, 
                node_dag_scale=0.1,
                embedder_pretrain_loss=False, 
                factor_pretrain_loss=False
            )
            forecasting_loss = forecasting_loss.cpu().detach().item()
            factor_loss = factor_loss.cpu().detach().item()
            if factor_cos_sim_penalty is not None:
                factor_cos_sim_penalty = factor_cos_sim_penalty.cpu().detach().item()
            fw_l1_penalty = fw_l1_penalty.cpu().detach().item()
            adj_l1_penalty = adj_l1_penalty.cpu().detach().item()
            reg_dagness_loss = 0.#reg_dagness_loss.cpu().detach().item() # DAGNESS LOSS REMOVED TO ENSURE NUMERICAL STABILITY, ESP. IN D4IC EXPERIMENTS - 12/20/2024
            #lagged_dagness_loss = lagged_dagness_loss.cpu().detach().item()
            #node_dagness_loss = node_dagness_loss.cpu().detach().item()
            
            if self.FORECAST_COEFF > 0.:
                forecasting_loss /= self.FORECAST_COEFF
            avg_forecasting_loss += forecasting_loss # divide out the coefficient(s) for comparisson in grid-searches
            if self.FACTOR_SCORE_COEFF > 0.:
                factor_loss /= self.FACTOR_SCORE_COEFF
            avg_factor_loss += factor_loss
            if self.FACTOR_COS_SIM_COEFF > 0.:
                if factor_cos_sim_penalty is not None:
                    factor_cos_sim_penalty /= self.FACTOR_COS_SIM_COEFF
                else:
                    factor_cos_sim_penalty = 0.
            avg_factor_cos_sim_penalty += factor_cos_sim_penalty
            if self.FACTOR_WEIGHT_L1_COEFF > 0.:
                fw_l1_penalty /= self.FACTOR_WEIGHT_L1_COEFF
            avg_fw_l1_penalty += fw_l1_penalty
            if self.ADJ_L1_REG_COEFF > 0.:
                adj_l1_penalty /= self.ADJ_L1_REG_COEFF
            avg_adj_penalty += adj_l1_penalty
            if self.DAGNESS_REG_COEFF > 0.:
                reg_dagness_loss /= self.DAGNESS_REG_COEFF
            avg_dagness_reg_loss += reg_dagness_loss
            
            avg_combo_loss += combined_loss.cpu().detach().item()

            if running_factor_score_confusion_matrix is not None: 
                state_label_pred_mask = state_label_preds[0].detach().cpu().numpy()
                Y_mask = None
                if len(Y.size()) == 3:
                    if Y.size()[2] > max(self.gen_lag, self.embed_lag): 
                        # we are only interested in the first predicted factor weighting here, so we grab the corresponding 
                        # label (note that Y has size batch_size x num_channels x num_recorded_timesteps)
                        Y_mask = Y[:,:self.num_supervised_factors,max(self.gen_lag, self.embed_lag)].detach().cpu().numpy() 
                    else: # this case arises in the DREAM4-based datasets, for example
                        assert Y.size()[2] == 1
                        Y_mask = Y[:,:self.num_supervised_factors,0].detach().cpu().numpy()
                elif len(Y.size()) == 2: # case for DREAM4 orig. data
                    Y_mask = Y[:,:self.num_supervised_factors].detach().cpu().numpy()
                else:
                    raise NotImplementedError("Cannot handle ground-truth labels with Y.size() == "+str(Y.size()))
                conf_mat_preds = [np.argmax(state_label_pred_mask[i,:]) for i in range(state_label_pred_mask.shape[0])]
                conf_mat_labels = [np.argmax(Y_mask[i,:]) for i in range(Y_mask.shape[0])]
                running_factor_score_confusion_matrix += confusion_matrix(
                    conf_mat_labels, conf_mat_preds, labels=[i for i in range(self.num_supervised_factors)]
                ) 
            
            del X
            del Y
            del x_sims_genUpdate
            del state_label_preds
        
        if self.num_supervised_factors > 0:
            # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html and https://stackoverflow.com/questions/45053238/how-to-get-all-confusion-matrix-terminologies-tpr-fpr-tnr-fnr-for-a-multi-c
            TP = np.diag(running_factor_score_confusion_matrix)
            FP = running_factor_score_confusion_matrix.sum(axis=0) - TP
            FN = running_factor_score_confusion_matrix.sum(axis=1) - TP
            TN = running_factor_score_confusion_matrix.sum() - (FP + FN + TP)
            TP = TP.astype(float)
            FP = FP.astype(float)
            FN = FN.astype(float)
            TN = TN.astype(float)
            TPR = TP/(TP+FN) # Sensitivity, hit rate, recall, or true positive rate
            TNR = TN/(TN+FP) # Specificity or true negative rate
            FPR = FP/(FP+TN) # Fall out or false positive rate
            FNR = FN/(TP+FN) # False negative rate
            ACC = (TP+TN)/(TP+FP+FN+TN) # Overall accuracy
            factor_score_val_acc_history.append(ACC)
            factor_score_val_tpr_history.append(TPR)
            factor_score_val_tnr_history.append(TNR)
            factor_score_val_fpr_history.append(FPR)
            factor_score_val_fnr_history.append(FNR)

        # track training stats
        avg_forecasting_loss = avg_forecasting_loss/len(X_val)
        avg_factor_loss = avg_factor_loss/len(X_val)
        avg_factor_cos_sim_penalty = avg_factor_cos_sim_penalty/len(X_val)
        avg_fw_l1_penalty = avg_fw_l1_penalty/len(X_val)
        avg_adj_penalty = avg_adj_penalty/len(X_val)
        avg_dagness_reg_loss = avg_dagness_reg_loss/len(X_val)
        avg_dagness_lag_loss = avg_dagness_lag_loss/len(X_val)
        avg_dagness_node_loss = avg_dagness_node_loss/len(X_val)
        avg_combo_loss = avg_combo_loss/len(X_val)

        if self.num_supervised_factors > 0:
            return avg_forecasting_loss, avg_factor_loss, avg_factor_cos_sim_penalty, avg_fw_l1_penalty, avg_adj_penalty, avg_dagness_reg_loss, avg_dagness_lag_loss, avg_dagness_node_loss, avg_combo_loss, factor_score_val_acc_history, factor_score_val_tpr_history, factor_score_val_tnr_history, factor_score_val_fpr_history, factor_score_val_fnr_history 
        return avg_forecasting_loss, avg_factor_loss, avg_factor_cos_sim_penalty, avg_fw_l1_penalty, avg_adj_penalty, avg_dagness_reg_loss, avg_dagness_lag_loss, avg_dagness_node_loss, avg_combo_loss