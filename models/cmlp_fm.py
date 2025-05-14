import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import os
import pickle as pkl
from sklearn.metrics import roc_auc_score

from models.cmlp import cMLP
from general_utils.metrics import DAGNessLoss, get_f1_score, compute_cosine_similarity
from general_utils.misc import flatten_GC_estimate_with_lags_and_gradient_tracking, unflatten_GC_estimate_with_lags
from general_utils.model_utils import restore_parameters
from general_utils.plotting import plot_curve, plot_curve_comparisson, plot_curve_comparisson_from_dict, plot_all_signal_channels, plot_gc_est_comparissons_by_factor, plot_x_simulation_comparisson



class MLPClassifier(nn.Module):
    def __init__(self, num_series, num_in_timesteps, num_out_classes, hidden_sizes, POST_CONVS_SIZE=6):
        super(MLPClassifier, self).__init__()
        self.POST_CONVS_SIZE = POST_CONVS_SIZE
        self.temporal_kernel_size = num_in_timesteps - self.POST_CONVS_SIZE + (1 - (num_in_timesteps - self.POST_CONVS_SIZE)%2)
        self.num_series = num_series
        self.num_in_timesteps = num_in_timesteps
        self.num_out_classes = num_out_classes
        self.hidden_sizes = hidden_sizes
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.final_activation = nn.ReLU()
        # Set up network
        self.series_conv1_layers = nn.Sequential(
            nn.Conv1d(num_series, hidden_sizes[0], 1), 
            nn.ReLU(), 
            nn.Conv1d(hidden_sizes[0], 1, 1),
        )
        self.temporal_conv1_layers = nn.Sequential(
            nn.Conv1d(1, hidden_sizes[1], self.temporal_kernel_size), 
        )
        final_linear_layers = []
        for f_in, f_out in zip([hidden_sizes[1]*self.POST_CONVS_SIZE]+hidden_sizes[2:], hidden_sizes[2:]+[num_out_classes]): # note: we append '1' as the final number of 'hidden' features because each class_net simply outputs a single scalar score corresponding to it's class
            final_linear_layers.append(nn.ReLU())
            final_linear_layers.append(nn.Linear(f_in, f_out))
        self.final_linear_layers = nn.Sequential(*final_linear_layers) # see https://discuss.pytorch.org/t/append-for-nn-sequential-or-directly-converting-nn-modulelist-to-nn-sequential/7104
        pass
    
    def forward(self, X, use_final_activation=True):
        curr_batch_size = X.size()[0]
        out = X.transpose(2, 1)
        out = self.relu(self.series_conv1_layers(out))
        out = self.temporal_conv1_layers(out)
        out = self.flatten(out)
        out = self.final_linear_layers(out)
        if use_final_activation:
            out = self.final_activation(out)
        return out.view(curr_batch_size, self.num_out_classes)

    

class cMLP_FM(nn.Module):
    def __init__(self, num_chans, gen_lag, gen_hidden, embed_hidden_sizes, num_in_timesteps, num_out_timesteps, coeff_dict, num_sims=1, wavelet_level=None, save_path=None):
        '''
        cMLP_FM model with num_factors cMLPs per time series.
        '''
        super(cMLP_FM, self).__init__()
        self.num_chans = num_chans # p in original cMLP implementation
        if wavelet_level is not None:
            self.num_series = num_chans*(wavelet_level+1)
        else:
            self.num_series = num_chans
        self.gen_lag = gen_lag
        self.gen_hidden = gen_hidden
        self.num_gen_hiddens = len(gen_hidden)
        self.embed_hidden_sizes = embed_hidden_sizes
        self.num_in_timesteps = num_in_timesteps
        self.num_out_timesteps = num_out_timesteps
        self.num_factors_nK = 1 # cmlp_fm is the baseline model for NCFM - the original formulation had only one factor
        self.coeff_dict = coeff_dict
        self.FORECAST_COEFF = coeff_dict["FORECAST_COEFF"]
        self.FACTOR_SCORE_COEFF = 0. # cmlp_fm is the baseline model for NCFM - the original formulation has no supervised component for factors
        self.ADJ_L1_REG_COEFF = coeff_dict["ADJ_L1_REG_COEFF"]
        self.DAGNESS_REG_COEFF = coeff_dict["DAGNESS_REG_COEFF"]
        self.DAGNESS_LAG_COEFF = coeff_dict["DAGNESS_LAG_COEFF"]
        self.DAGNESS_NODE_COEFF = coeff_dict["DAGNESS_NODE_COEFF"]
        self.num_sims = num_sims
        self.wavelet_level = wavelet_level
        self.supervised_loss_fn = nn.MSELoss(reduction='mean')
        self.dagness_loss_fn = DAGNessLoss()
        # Set up factors.
        self.factor_score_embedder = None
        self.factors = nn.ModuleList([
            cMLP(num_chans, gen_lag, gen_hidden, wavelet_level=wavelet_level, save_path=save_path) for _ in range(self.num_factors_nK)
        ])
        self.gen_model = nn.ModuleList([self.factor_score_embedder, self.factors])
        pass

    
    def forward(self, X):
        '''
        Perform forward pass.
        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        # initialize vars
        in_x = torch.zeros(*X.size())
        if torch.cuda.is_available():
            in_x = in_x.to(device="cuda")
        in_x = in_x + X

        inputs = [in_x]
        x_simulations = []
        factor_preds_over_sim = []
        factor_weighting_preds = []
            
        # make preds according to factor_weightings
        for s in range(self.num_sims):
            factor_weightings = None
            factor_preds = []
            curr_sim_start = s
            if s > 0:
                if x_simulations[-1].size() == inputs[-1].size():
                    inputs.append(x_simulations[-1])
                else:
                    inputs.append(torch.cat([inputs[-1][:,x_simulations[-1].size()[1]:,:], x_simulations[-1]],dim=1))
            
            combined_pred = None
            for i in range(self.num_factors_nK):
                # make prediction from one factor
                curr_pred = self.factors[i](inputs[s])
                # sum over factor predictions
                if i == 0:
                    combined_pred = curr_pred
                else:
                    combined_pred = combined_pred + curr_pred
                factor_preds.append(curr_pred)

            # record results of current sim step
            factor_preds_over_sim.append(factor_preds)
            factor_weighting_preds.append(factor_weightings)
            x_simulations.append(combined_pred)
            pass

        x_simulations = torch.cat(x_simulations, dim=1)
        return x_simulations, factor_preds_over_sim, factor_weighting_preds

    
    def GC(self, threshold=True, ignore_lag=True, combine_wavelet_representations=False, rank_wavelets=False):
        '''
        Extract learned Granger causality from each factor.
        Args:
          threshold: return norm of weights, or whether norm is nonzero.
        Returns:
          GCs: list of self.num_factors_nK (p x p) matrices. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        '''
        return [factor.GC(threshold=threshold, ignore_lag=ignore_lag, combine_wavelet_representations=combine_wavelet_representations, rank_wavelets=rank_wavelets) for factor in self.factors]
    
    def compute_loss(self, preds, targets, node_dag_scale=0.1):
        gc = self.GC(threshold=False, ignore_lag=True, combine_wavelet_representations=False, rank_wavelets=False)
        gc_lagged = self.GC(threshold=False, ignore_lag=False, combine_wavelet_representations=False, rank_wavelets=False)
        # get supervised loss components
        forecasting_loss = self.FORECAST_COEFF*sum([
            self.supervised_loss_fn(preds[:, :, i], targets[:, :, i]) for i in range(self.num_series)
        ])
        # get regularization penalties
        adj_l1_penalty = self.ADJ_L1_REG_COEFF*sum([torch.norm(A, 1) for A in gc])# see https://pytorch.org/docs/stable/generated/torch.norm.html
        # DAGNESS LOSS REMOVED TO ENSURE NUMERICAL STABILITY, ESP. IN D4IC EXPERIMENTS - 12/20/2024
        reg_dagness_loss = None#self.DAGNESS_REG_COEFF*sum([self.dagness_loss_fn(A) for A in gc])#self.dagness_loss_fn(gc[0])
        #lagged_dag_penalties_by_factor = []
        #node_dag_penalties_by_factor = []
        #for A in gc_lagged:
        #    A_flat = flatten_GC_estimate_with_lags_and_gradient_tracking(A)
        #    A_flat_transpose = torch.transpose(A_flat, 0, 1)
        #    lagged_dag_penalties_by_factor.append(self.dagness_loss_fn(torch.matmul(A_flat_transpose, A_flat)))
        #    node_dag_penalties_by_factor.append(self.dagness_loss_fn(node_dag_scale*torch.matmul(A_flat, A_flat_transpose)))
        lagged_dagness_loss = None#self.DAGNESS_LAG_COEFF*sum(lagged_dag_penalties_by_factor)
        node_dagness_loss = None#self.DAGNESS_NODE_COEFF*sum(node_dag_penalties_by_factor)
        # combine loss compoents and return
        combo_loss = forecasting_loss + adj_l1_penalty# + reg_dagness_loss + lagged_dagness_loss + node_dagness_loss # DAGNESS LOSS REMOVED TO ENSURE NUMERICAL STABILITY, ESP. IN D4IC EXPERIMENTS - 12/20/2024
        return combo_loss, [forecasting_loss, adj_l1_penalty, reg_dagness_loss, lagged_dagness_loss, node_dagness_loss]

    def batch_update(self, batch_num, X, gen_optim, input_length, output_length):
        # Set up data.
        if torch.cuda.is_available():
            X = X.to(device="cuda")
        # UPDATE factor models and factor score embedder
        # Prep parameters for updates
        for f in self.factors:
            f.train()
        gen_optim.zero_grad()
        # make predictions/forecast
        x_sims_genUpdate, _, _ = self.forward(X[:,:input_length,:])
        # Calculate loss 
        combined_loss, _ = self.compute_loss(
            x_sims_genUpdate, 
            X[:,input_length:input_length+(self.num_sims*output_length),:]
        )
        # update parameters
        combined_loss.backward()
        gen_optim.step()
        del X
        del x_sims_genUpdate
        pass

    
    def save_checkpoint(self, save_dir, it, best_model, avg_forecasting_loss, avg_adj_penalty, avg_dagness_reg_loss, avg_dagness_lag_loss, avg_dagness_node_loss, 
                        avg_combo_loss, best_loss, best_it, f1score_histories, roc_auc_histories, gc_factor_l1_loss_histories, GC, X_train=None, 
                        X_val=None, input_length=None, output_length=None, num_sim_steps=None):
        # save summary stats
        temp_model_save_path = os.path.join(save_dir, "final_best_model.bin")
        torch.save(best_model, temp_model_save_path)
        meta_data_save_path = os.path.join(save_dir, "training_meta_data_and_hyper_parameters.pkl")
        with open(meta_data_save_path, "wb") as outfile:
            pkl.dump({
                "epoch": it, 
                "avg_forecasting_loss": avg_forecasting_loss, 
                "avg_adj_penalty": avg_adj_penalty, 
                "avg_dagness_reg_loss": avg_dagness_reg_loss, 
                "avg_dagness_lag_loss": avg_dagness_lag_loss, 
                "avg_dagness_node_loss": avg_dagness_node_loss, 
                "avg_combo_loss": avg_combo_loss,  
                "best_loss": best_loss, 
                "best_it": best_it, 
                "f1score_histories": f1score_histories, 
                "roc_auc_histories": roc_auc_histories, 
                "gc_factor_l1_loss_histories": gc_factor_l1_loss_histories, 
            }, outfile)
        
        # plot loss histories
        plot_curve(avg_forecasting_loss, "Validation Forecasting MSE Loss", "Epoch", "Average MSE Loss", save_dir+os.sep+"avg_val_forecasting_mse_loss.png", domain_start=0)
        plot_curve(avg_adj_penalty, "Validation Adjacency L1 Penalty", "Epoch", "Average L1 Penalty", save_dir+os.sep+"avg_val_adj_L1_penalty.png", domain_start=0)
        plot_curve(avg_dagness_reg_loss, "Validation Dagness Loss", "Epoch", "Average Dagness REG Loss", save_dir+os.sep+"avg_val_dagness_reg_loss.png", domain_start=0)
        plot_curve(avg_dagness_lag_loss, "Validation Dagness Loss", "Epoch", "Average Dagness LAG Loss", save_dir+os.sep+"avg_val_dagness_lag_loss.png", domain_start=0)
        plot_curve(avg_dagness_node_loss, "Validation Dagness Loss", "Epoch", "Average Dagness NODE Loss", save_dir+os.sep+"avg_val_dagness_node_loss.png", domain_start=0)
        plot_curve(avg_combo_loss, "Validation Combined Loss", "Epoch", "Average Combined Loss", save_dir+os.sep+"avg_val_combo_loss.png", domain_start=0)
        for key in f1score_histories.keys():
            key_str = str(key).replace(".","-")
            plot_curve_comparisson(f1score_histories[key], "F1 Score History for Threshold="+key_str, "Epoch", "Granger Causal F1 Score", save_dir+os.sep+"f1_score_history_"+key_str+"_visualization.png", domain_start=0, label_root="factor")
            plot_curve_comparisson(roc_auc_histories[key], "ROC-AUC Score History for Threshold="+key_str, "Epoch", "Granger Causal ROC-AUC Score", save_dir+os.sep+"roc_auc_score_history_"+key_str+"_visualization.png", domain_start=0, label_root="factor")
        plot_curve_comparisson(gc_factor_l1_loss_histories, "GC L1 Loss History", "Epoch", "L1 Norm", save_dir+os.sep+"gc_l1_loss_history_visualization.png", domain_start=0, label_root="factor")
        
        # make GC Estimate visualizations
        GC_est = self.GC(threshold=False, ignore_lag=False, combine_wavelet_representations=True, rank_wavelets=False)
        GC_est = [A.cpu().data.numpy() for A in GC_est]
        if len(GC[0].shape) == 3:
            plot_gc_est_comparissons_by_factor(GC, GC_est, save_dir+os.sep+"gc_est_withLags_results_epoch"+str(it)+".png", include_lags=True)
            GC_est_noLags = self.GC(threshold=False, ignore_lag=True, combine_wavelet_representations=True, rank_wavelets=False)
            GC_est_noLags = [A.cpu().data.numpy() for A in GC_est_noLags]
            GC_noLags = [np.sum(x, axis=2) for x in GC]
            plot_gc_est_comparissons_by_factor(GC_noLags, GC_est_noLags, save_dir+os.sep+"gc_est_withLags_results_epoch"+str(it)+".png", include_lags=False)
        else:
            plot_gc_est_comparissons_by_factor(GC, GC_est, save_dir+os.sep+"gc_est_noLags_results_epoch"+str(it)+".png")
        if self.wavelet_level is not None:
            wGC_est = self.GC(threshold=False, ignore_lag=False, combine_wavelet_representations=False, rank_wavelets=False)
            wGC_est = [A.cpu().data.numpy() for A in wGC_est]
            plot_gc_est_comparissons_by_factor(GC, wGC_est, save_dir+os.sep+"gc_est_results_per_wavelet_epoch"+str(it)+".png", include_lags=True)
            rwGC_est = self.GC(threshold=False, ignore_lag=False, combine_wavelet_representations=False, rank_wavelets=True)
            rwGC_est = [A.cpu().data.numpy() for A in rwGC_est]
            plot_gc_est_comparissons_by_factor(GC, rwGC_est, save_dir+os.sep+"gc_est_results_per_ranked_wavelet_epoch"+str(it)+".png", include_lags=True)
        
        if X_train is not None:
            raise(NotImplementedError)
        pass

    
    def fit(self, save_dir, X_train, gen_optim, input_length, output_length, num_sim_steps, max_iter, lookback=5, check_every=50, verbose=1, GC=None, X_val=None):
        f1_thresholds = [0.0]
        # For tracking intermediate/preliminary results
        avg_forecasting_loss = []
        avg_adj_penalty = []
        avg_dagness_reg_loss = []
        avg_dagness_lag_loss = []
        avg_dagness_node_loss = []
        avg_combo_loss = []
        f1score_histories = {thresh:[[] for _ in range(len(GC))] for thresh in f1_thresholds}
        roc_auc_histories = {thresh:[[] for _ in range(len(GC))] for thresh in f1_thresholds}
        gc_factor_l1_loss_histories = [[] for _ in range(len(GC))]

        # For early stopping.
        best_it = None
        best_loss = np.inf
        best_model = None

        for it in range(max_iter):
            if it%5 == 0:
                print("cMLP_FM.fit: now on epoch it == ", it, flush=True)
            
            for batch_num, (X, _) in enumerate(X_train):
                self.batch_update(
                    batch_num, 
                    X, 
                    gen_optim, 
                    input_length, 
                    output_length
                )
            
            # monitor/track GC development progress
            curr_gc_est = self.GC(threshold=False, ignore_lag=False, combine_wavelet_representations=False, rank_wavelets=False)
            for thresh_key in f1score_histories.keys():
                for i, gc_est in enumerate(curr_gc_est):
                    curr_est = gc_est.detach().cpu().numpy()
                    curr_est = np.sum(curr_est, axis=2)
                    curr_est = curr_est / np.max(curr_est)
                    mask = curr_est > thresh_key
                    curr_est = curr_est * mask
                    for j, curr_true_gc in enumerate(GC):
                        curr_true_gc = np.sum(curr_true_gc, axis=2)
                        curr_true_gc = curr_true_gc / np.max(curr_true_gc)
                        f1score_histories[thresh_key][j].append(get_f1_score(curr_est, curr_true_gc))
                        roc_auc_labels = [int(l) for l in curr_true_gc.flatten()]
                        roc_auc_histories[thresh_key][j].append(roc_auc_score(roc_auc_labels, curr_est.flatten())) # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
            
            curr_l1_loss = None
            for est_num, gc_est in enumerate(curr_gc_est):
                curr_norm = None
                try:
                    gc_est = gc_est / np.max(gc_est)
                    curr_norm = torch.norm(torch.from_numpy(gc_est), 1)
                except:
                    gc_est = gc_est / torch.max(gc_est)
                    curr_norm = torch.norm(gc_est, 1)
                
                gc_factor_l1_loss_histories[est_num].append(curr_norm.detach().numpy())

                if est_num == 0:
                    curr_l1_loss = curr_norm
                else:
                    try:
                        curr_l1_loss += curr_norm
                    except:
                        curr_l1_loss += curr_norm
            
            # track validation stats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            avg_val_forecasting_loss, \
            avg_val_adj_penalty, \
            avg_val_dagness_reg_loss, \
            avg_val_dagness_lag_loss, \
            avg_val_dagness_node_loss, \
            avg_val_combo_loss = self.validate_training(X_val, input_length, output_length, self.num_series)
            
            avg_forecasting_loss.append(avg_val_forecasting_loss)
            avg_adj_penalty.append(avg_val_adj_penalty)
            avg_dagness_reg_loss.append(avg_val_dagness_reg_loss)
            avg_dagness_lag_loss.append(avg_val_dagness_lag_loss)
            avg_dagness_node_loss.append(avg_val_dagness_node_loss)
            avg_combo_loss.append(avg_val_combo_loss)
            
            # Check for early stopping.
            if curr_l1_loss+avg_val_forecasting_loss < best_loss: # FOR DEBUGGING PURPOSES
                best_loss = curr_l1_loss+avg_val_forecasting_loss
                best_it = it
                best_model = deepcopy(self)
            elif (it - best_it) == lookback * check_every: # FOR DEBUGGING PURPOSES
                if verbose:                                # FOR DEBUGGING PURPOSES
                    print('Stopping early')                # FOR DEBUGGING PURPOSES
                break                                      # FOR DEBUGGING PURPOSES

            # Check progress.
            if it % check_every == 0:
                if verbose > 0:
                    print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                    print('Validation Loss = %f' % avg_val_combo_loss)
                    curr_gc_ests = self.GC()
                    for gc_est_num in range(self.num_factors_nK):
                        print('Factor '+str(gc_est_num)+' Variable usage = %.2f%%' % (100 * torch.mean(curr_gc_ests[gc_est_num].float())))
                print("cMLP_FM.fit: \t CHECKING")
                print("cMLP_FM.fit: \t avg_forecasting_loss == ", avg_forecasting_loss)
                print("cMLP_FM.fit: \t avg_adj_penalty == ", avg_adj_penalty)
                print("cMLP_FM.fit: \t avg_dagness_reg_loss == ", avg_dagness_reg_loss)
                print("cMLP_FM.fit: \t avg_dagness_lag_loss == ", avg_dagness_lag_loss)
                print("cMLP_FM.fit: \t avg_dagness_node_loss == ", avg_dagness_node_loss)
                print("cMLP_FM.fit: \t avg_combo_loss == ", avg_combo_loss)
                print("cMLP_FM.fit: \t f1score_histories == ", f1score_histories)
                print("cMLP_FM.fit: \t roc_auc_histories == ", roc_auc_histories)
                print("cMLP_FM.fit: \t gc_factor_l1_loss_histories == ", gc_factor_l1_loss_histories)
                
                # save checkpoint
                self.save_checkpoint(
                    save_dir, 
                    it, 
                    best_model, 
                    avg_forecasting_loss, 
                    avg_adj_penalty, 
                    avg_dagness_reg_loss, 
                    avg_dagness_lag_loss, 
                    avg_dagness_node_loss, 
                    avg_combo_loss, 
                    best_loss, 
                    best_it, 
                    f1score_histories, 
                    roc_auc_histories, 
                    gc_factor_l1_loss_histories, 
                    GC, 
                    X_train=None, #X_train, 
                    X_val=None,# X_val, 
                    input_length=None,# input_length, 
                    output_length=None,# output_length, 
                    num_sim_steps=None#, num_sim_steps
                )
            
            pass

        # Restore best model.
        restore_parameters(self, best_model)
        final_save_path = os.path.join(save_dir, "final_best_model.bin")
        torch.save(self, final_save_path)

        # Report final Validation Score(s)
        _, _, _, _, _, final_mean_val_combo_loss = self.validate_training(
            X_val, input_length, output_length, self.num_series
        )
        print("FINAL BEST (STOPPING CRITERIA) LOSS == ", best_loss, flush=True)
        print("FINAL BEST (STOPPING CRITERIA) EPOCH == ", best_it, flush=True)
        print("FINAL VALIDATION COMBO LOSS == ", final_mean_val_combo_loss, flush=True)
        return final_mean_val_combo_loss

    
    def validate_training(self, X_val, input_length, output_length, num_series, report_normalized_loss_components=False):
        print("cMLP_FM.training_sim_eval: START")
        # initialize vars for tracking intermediate/preliminary results
        avg_forecasting_loss = 0.
        avg_adj_penalty = 0.
        avg_dagness_reg_loss = 0.
        avg_dagness_lag_loss = 0.
        avg_dagness_node_loss = 0.
        avg_combo_loss = 0.

        print("cMLP_FM.training_sim_eval: iterating over validation set")
        for batch_num, (X, _) in enumerate(X_val):
            # Set up data.
            if torch.cuda.is_available():
                X = X.to(device="cuda")
            
            for f in self.factors:
                f.eval()

            # Make Prediction(s)
            x_sims_genUpdate, _, _ = self.forward(X[:,:input_length,:])
            # Calculate loss 
            combined_loss, [forecasting_loss, adj_l1_penalty, _, _, _] = self.compute_loss(
                x_sims_genUpdate, 
                X[:,input_length:input_length+(self.num_sims*output_length),:]
            )
            
            if report_normalized_loss_components:
                if self.FORECAST_COEFF > 0.:
                    forecasting_loss /= self.FORECAST_COEFF
                if self.ADJ_L1_REG_COEFF > 0.:
                    adj_l1_penalty /= self.ADJ_L1_REG_COEFF
                # if self.DAGNESS_REG_COEFF > 0.:
                #     avg_dagness_reg_loss /= self.DAGNESS_REG_COEFF
                # if self.DAGNESS_LAG_COEFF > 0.:
                #     avg_dagness_lag_loss /= self.DAGNESS_LAG_COEFF
                # if self.DAGNESS_NODE_COEFF > 0.:
                #     avg_dagness_node_loss /= self.DAGNESS_NODE_COEFF
            avg_forecasting_loss += forecasting_loss.cpu().detach().item()
            avg_adj_penalty += adj_l1_penalty.cpu().detach().item()
            avg_dagness_reg_loss += 0.#reg_dagness_loss.cpu().detach().item() # REMOVED TO ENSURE NUMERICAL STABILITY, ESP. IN D4IC EXPERIMENTS - 12/20/2024
            avg_dagness_lag_loss += 0.#lagged_dagness_loss.cpu().detach().item()
            avg_dagness_node_loss += 0.#node_dagness_loss.cpu().detach().item()
            avg_combo_loss += combined_loss.cpu().detach().item()
            
            del X
            del x_sims_genUpdate
        
        # track training stats
        avg_forecasting_loss = avg_forecasting_loss/len(X_val)
        avg_adj_penalty = avg_adj_penalty/len(X_val)
        avg_dagness_reg_loss = avg_dagness_reg_loss/len(X_val)
        avg_dagness_lag_loss = avg_dagness_lag_loss/len(X_val)
        avg_dagness_node_loss = avg_dagness_node_loss/len(X_val)
        avg_combo_loss = avg_combo_loss/len(X_val)

        return avg_forecasting_loss, avg_adj_penalty, avg_dagness_reg_loss, avg_dagness_lag_loss, avg_dagness_node_loss, avg_combo_loss
