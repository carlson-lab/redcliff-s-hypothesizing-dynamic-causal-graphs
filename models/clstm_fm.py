import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import os
import pickle as pkl

from models.clstm import cLSTM
from models.cmlp_fm import MLPClassifier
from general_utils.metrics import DAGNessLoss
from general_utils.model_utils import restore_parameters, generate_signal_from_sequential_factor_model
from general_utils.plotting import plot_curve, plot_all_signal_channels, plot_gc_est_comparissons_by_factor, plot_x_simulation_comparisson



class cLSTM_FM(nn.Module):
    def __init__(self, num_chans, gen_hidden, embed_hidden_sizes, num_in_timesteps,  
                 coeff_dict, num_sims=1, wavelet_level=None, save_path=None):
        '''
        cLSTM_FM model with num_factors cLSTMs per time series.
        '''
        print("models.clstm_fm.cLSTM_FM.__init__: START", flush=True)
        super(cLSTM_FM, self).__init__()
        self.num_chans = num_chans # p in original cLSTM implementation
        if wavelet_level is not None:
            self.num_series = num_chans*(wavelet_level+1)
        else:
            self.num_series = num_chans
        self.gen_hidden = gen_hidden
        self.embed_hidden_sizes = embed_hidden_sizes
        self.num_in_timesteps = num_in_timesteps
        self.num_factors_nK = 1 # THIS MODEL IMPLEMENTS THE BASELINE cLSTM_FM ALGORITHM
        self.coeff_dict = coeff_dict
        self.FORECAST_COEFF = coeff_dict["FORECAST_COEFF"]
        self.ADJ_L1_REG_COEFF = coeff_dict["ADJ_L1_REG_COEFF"]
        self.DAGNESS_REG_COEFF = coeff_dict["DAGNESS_REG_COEFF"]
        self.num_sims = num_sims
        self.wavelet_level = wavelet_level
        
        self.supervised_loss_fn = nn.MSELoss(reduction='mean')
        self.dagness_loss_fn = DAGNessLoss()

        # Set up factors.
        print("models.clstm_fm.cLSTM_FM.__init__: SETTING FACTOR SCORE EMBEDDER TO NONE", flush=True)
        self.factor_score_embedder = None

        print("models.clstm_fm.cLSTM_FM.__init__: SETTING UP FACTORS", flush=True)
        self.factors = nn.ModuleList([
            cLSTM(num_chans, gen_hidden, wavelet_level=wavelet_level, save_path=save_path) for _ in range(self.num_factors_nK)
        ])
        self.gen_model = nn.ModuleList([self.factor_score_embedder, self.factors])
        print("models.clstm_fm.cLSTM_FM.__init__: STOP", flush=True)
        pass
        
        
    def forward(self, X, hiddens=None):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
          hidden: hidden states for LSTM cell.
        '''
        # initialize vars
        combined_pred = None
        factor_preds = []
        factor_hiddens = []
        if hiddens is None:
            hiddens = [None for _ in range(self.num_factors_nK)]
        
        # make preds
        for i in range(self.num_factors_nK):
            curr_pred, curr_hidden = self.factors[i](X, hiddens[i])
            if i == 0:
                combined_pred = curr_pred
            else:
                combined_pred = combined_pred + curr_pred
            factor_preds.append(curr_pred)
            factor_hiddens.append(curr_hidden)
        
        return combined_pred, factor_preds, factor_hiddens
    
    
    def GC(self, threshold=True, combine_wavelet_representations=False, rank_wavelets=False):
        '''
        Extract learned Granger causality from each factor.
        Args:
          threshold: return norm of weights, or whether norm is nonzero.
        Returns:
          GCs: list of self.num_factors_nK (p x p) matrices. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        '''
        return [factor.GC(threshold=threshold, combine_wavelet_representations=combine_wavelet_representations, rank_wavelets=rank_wavelets) for factor in self.factors]
    
    def arrange_input(self, data, context):
        '''
        Arrange a single time series into overlapping short sequences.
        Args:
        data: time series of shape (T, dim).
        context: length of short sequences.
        '''
        assert context >= 1 and isinstance(context, int)
        input = torch.zeros(len(data) - context, context, data.shape[1],
                            dtype=torch.float32, device=data.device)
        target = torch.zeros(len(data) - context, context, data.shape[1],
                            dtype=torch.float32, device=data.device)
        for i in range(context):
            start = i
            end = len(data) - context + i
            input[:, i, :] = data[start:end]
            target[:, i, :] = data[start+1:end+1]
        return input.detach(), target.detach()

    def configure_context_data_and_targets(self, X, max_input_length, context):
        if max_input_length is not None:
            X = X[:,:max_input_length,:]
        X, Y = zip(*[self.arrange_input(x, context) for x in X])
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)
        if torch.cuda.is_available():
            X, Y = X.to(device="cuda"), Y.to(device="cuda")
        return X, Y

    
    def compute_loss(self, preds, targets):
        gc = self.GC(threshold=False, combine_wavelet_representations=False, rank_wavelets=False)
        # get supervised loss components
        forecasting_loss = self.FORECAST_COEFF*sum([
            self.supervised_loss_fn(preds[:, :, i], targets[:, :, i]) for i in range(self.num_series)
        ])
        
        # get regularization penalties
        adj_l1_penalty = self.ADJ_L1_REG_COEFF*sum([torch.norm(A, 1) for A in gc])# see https://pytorch.org/docs/stable/generated/torch.norm.html
        dagness_loss = None#self.DAGNESS_REG_COEFF*self.dagness_loss_fn(gc[0]) # DAGNESS LOSS REMOVED TO ENSURE NUMERICAL STABILITY, ESP. IN D4IC EXPERIMENTS - 12/20/2024

        # combine loss compoents and return
        smooth_loss = forecasting_loss + adj_l1_penalty# + dagness_loss#+ ridge_penalty # DAGNESS LOSS REMOVED TO ENSURE NUMERICAL STABILITY, ESP. IN D4IC EXPERIMENTS - 12/20/2024
        return smooth_loss, [forecasting_loss, adj_l1_penalty, dagness_loss]

    
    def batch_update(self, batch_num, X, max_input_length, context, gen_optim, running_forecasting_loss,  
                     running_adj_penalty, running_dagness_loss, running_smooth_loss):
        # Set up data.
        X_in, X_target = self.configure_context_data_and_targets(X, max_input_length, context)
        
        # UPDATE factor models
        # Prep parameters for updates
        for f in self.factors:
            f.train()
        gen_optim.zero_grad()

        # make predictions/forecast
        x_sims, _, _ = self.forward(X_in, hiddens=None)

        # Calculate loss 
        smooth, [forecasting_loss, adj_l1, dagness_loss] = self.compute_loss(
            x_sims, 
            X_target
        )

        # update parameters
        smooth.backward()
        gen_optim.step()

        # Note: in the original cLSTM formulation, a prox update (shown below) would normally be performed - WE DO NOT PERFORM A PROX UPDATE BECAUSE WE ARE USING L1-REGULARIZATION IN THE ADAM OPTIMIZER for more direct comparisons with NCFM formulations
        # for factor in self.factors:
        #     factor.perform_prox_update_on_GC_weights(lam, glr)

        running_forecasting_loss += forecasting_loss.cpu().detach().item()
        running_adj_penalty += adj_l1.cpu().detach().item()
        running_dagness_loss += 0.#dagness_loss.cpu().detach().item()# DAGNESS LOSS REMOVED TO ENSURE NUMERICAL STABILITY, ESP. IN D4IC EXPERIMENTS - 12/20/2024
        running_smooth_loss += smooth.cpu().detach().item()
        del X
        del X_in
        del X_target
        del x_sims
        return running_forecasting_loss, running_adj_penalty, running_dagness_loss, running_smooth_loss

    
    def save_checkpoint(self, save_dir, it, best_model, avg_forecasting_loss, avg_adj_penalty, avg_dagness_loss, avg_smooth_loss, 
                        best_loss, GC, X_train, input_length, num_sim_steps, X_val=None):
        temp_model_save_path = os.path.join(save_dir, "temp_best_model_epoch"+str(it)+".bin")
        torch.save(best_model, temp_model_save_path)
        meta_data_save_path = os.path.join(save_dir, "training_meta_data_and_hyper_parameters.pkl")
        with open(meta_data_save_path, "wb") as outfile:
            pkl.dump({
                "epoch": it, 
                "avg_forecasting_loss": avg_forecasting_loss, 
                "avg_adj_penalty": avg_adj_penalty, 
                "avg_dagness_loss": avg_dagness_loss, 
                "avg_smooth_loss": avg_smooth_loss, 
                "best_loss": best_loss, 
            }, outfile)
        
        plot_curve(avg_forecasting_loss, "Training Forecasting MSE Loss", "Epoch", "Average MSE Loss", save_dir+os.sep+"avg_training_forecasting_mse_loss_epoch"+str(it)+".png", domain_start=0)
        plot_curve(avg_adj_penalty, "Training Adjacency L1 Penalty", "Epoch", "Average L1 Penalty", save_dir+os.sep+"avg_training_adj_L1_penalty_epoch"+str(it)+".png", domain_start=0)
        plot_curve(avg_dagness_loss, "Training Dagness Loss", "Epoch", "Average Dagness Loss", save_dir+os.sep+"avg_dagness_loss_epoch"+str(it)+".png", domain_start=0)
        plot_curve(avg_smooth_loss, "Training Smooth Loss", "Epoch", "Average Smooth Loss", save_dir+os.sep+"avg_training_smooth_loss_epoch"+str(it)+".png", domain_start=0)

        GC_est = self.GC(threshold=False, combine_wavelet_representations=True, rank_wavelets=False)
        GC_est = [A.cpu().data.numpy() for A in GC_est]
        plot_gc_est_comparissons_by_factor(GC, GC_est, save_dir+os.sep+"gc_est_results_epoch"+str(it)+".png")
        del GC_est
        if self.wavelet_level is not None:
            wGC_est = self.GC(threshold=False, combine_wavelet_representations=False, rank_wavelets=False)
            wGC_est = [A.cpu().data.numpy() for A in wGC_est]
            plot_gc_est_comparissons_by_factor(GC, wGC_est, save_dir+os.sep+"gc_est_results_per_wavelet_epoch"+str(it)+".png")
            del wGC_est
            rwGC_est = self.GC(threshold=False, combine_wavelet_representations=False, rank_wavelets=True)
            rwGC_est = [A.cpu().data.numpy() for A in rwGC_est]
            plot_gc_est_comparissons_by_factor(GC, rwGC_est, save_dir+os.sep+"gc_est_results_per_ranked_wavelet_epoch"+str(it)+".png")
            del rwGC_est
        
        pass
    
    
    def fit(self, save_dir, X_train, gen_optim, context, max_input_length, num_sim_steps, max_iter, lookback=5, check_every=50, verbose=1, GC=None, X_val=None):
        # For tracking intermediate/preliminary results
        avg_forecasting_loss = []
        avg_adj_penalty = []
        avg_dagness_loss = []
        avg_smooth_loss = []

        # For early stopping.
        best_it = None
        best_loss = np.inf
        best_model = None

        for it in range(max_iter):
            print("cLSTM_FM.fit: now on epoch it == ", it, flush=True)
            # initialize vars for tracking stats
            running_forecasting_loss = 0.
            running_adj_penalty = 0.
            running_dagness_loss = 0.
            running_smooth_loss = 0.

            for batch_num, (X, _) in enumerate(X_train): # drop the state labels
                running_forecasting_loss, \
                running_adj_penalty, \
                running_dagness_loss, \
                running_smooth_loss = self.batch_update(
                    batch_num, 
                    X, 
                    max_input_length, 
                    context, 
                    gen_optim, 
                    running_forecasting_loss, 
                    running_adj_penalty, 
                    running_dagness_loss, 
                    running_smooth_loss
                )
            
            # track training stats
            avg_forecasting_loss.append(running_forecasting_loss/len(X_train))
            avg_adj_penalty.append(running_adj_penalty/len(X_train))
            avg_dagness_loss.append(running_dagness_loss/len(X_train))
            avg_smooth_loss.append(running_smooth_loss/len(X_train))

            # Check progress.
            if it % check_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("cLSTM_FM.fit: \t CHECKING")
                print("cLSTM_FM.fit: \t avg_forecasting_loss == ", avg_forecasting_loss)
                print("cLSTM_FM.fit: \t avg_adj_penalty == ", avg_adj_penalty)
                print("cLSTM_FM.fit: \t avg_dagness_loss == ", avg_dagness_loss)
                print("cLSTM_FM.fit: \t avg_smooth_loss == ", avg_smooth_loss, flush=True)

                mean_val_combo_loss = self.training_sim_eval(
                    X_val, 
                    max_input_length, 
                    context, 
                    self.num_series
                )

                if verbose > 0:
                    print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1), flush=True)
                    print('Validation Loss = %f' % mean_val_combo_loss)
                    curr_gc_ests = self.GC()
                    for gc_est_num in range(self.num_factors_nK):
                        print('Factor '+str(gc_est_num)+' Variable usage = %.2f%%' % (100 * torch.mean(curr_gc_ests[gc_est_num].float())))

                # Check for early stopping.
                curr_gc_est = self.GC(threshold=False, combine_wavelet_representations=False, rank_wavelets=False)
                curr_l1_loss = None
                for est_num, gc_est in enumerate(curr_gc_est):
                    if est_num == 0:
                        try:
                            curr_l1_loss = torch.norm(torch.from_numpy(gc_est), 1)
                        except:
                            curr_l1_loss = torch.norm(gc_est, 1)
                    else:
                        try:
                            curr_l1_loss += torch.norm(torch.from_numpy(gc_est), 1)
                        except:
                            curr_l1_loss += torch.norm(gc_est, 1)

                if curr_l1_loss < best_loss: 
                    best_loss = curr_l1_loss 
                    best_it = it
                    best_model = deepcopy(self)
                elif (it - best_it) == lookback * check_every:
                    if verbose:
                        print('Stopping early', flush=True)
                    break
                
                # save checkpoint
                self.save_checkpoint(
                    save_dir, 
                    it, 
                    best_model, 
                    avg_forecasting_loss, 
                    avg_adj_penalty, 
                    avg_dagness_loss, 
                    avg_smooth_loss, 
                    best_loss, 
                    GC, 
                    X_train, 
                    context, 
                    num_sim_steps, 
                    X_val=X_val
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            pass

        # Restore best model.
        restore_parameters(self, best_model)
        final_save_path = os.path.join(save_dir, "final_best_model.bin")
        torch.save(self, final_save_path)

        # Report final Validation Score(s)
        if X_val is not None:
            final_mean_val_combo_loss = self.training_sim_eval(X_val, max_input_length, context, self.num_series)
            print("FINAL VALIDATION COMBO LOSS == ", final_mean_val_combo_loss, flush=True)

        return final_mean_val_combo_loss

    
    def training_sim_eval(self, X_val, max_input_length, context, num_series, return_loss_componentwise=False, report_normalized_loss_components=False):
        # initialize vars for tracking intermediate/preliminary results
        avg_forecasting_loss = 0.
        avg_adj_penalty = 0.
        avg_dagness_loss = 0.
        avg_smooth_loss = 0.

        for f in self.factors:
            f.eval()

        for batch_num, (X, _) in enumerate(X_val):
            # Set up data.
            X_in, X_target = self.configure_context_data_and_targets(X, max_input_length, context)
            # make predictions/forecast
            x_sims, _, _ = self.forward(X_in, hiddens=None)
            # Calculate loss 
            smooth, [forecasting_loss, adj_l1, dagness_loss] = self.compute_loss(x_sims, X_target)
            
            if report_normalized_loss_components:
                if self.FORECAST_COEFF > 0.:
                    forecasting_loss /= self.FORECAST_COEFF
                if self.ADJ_L1_REG_COEFF > 0.:
                    adj_l1 /= self.ADJ_L1_REG_COEFF
                #if self.DAGNESS_REG_COEFF > 0.:# DAGNESS LOSS REMOVED TO ENSURE NUMERICAL STABILITY, ESP. IN D4IC EXPERIMENTS - 12/20/2024
                #    avg_dagness_loss /= self.DAGNESS_REG_COEFF

            avg_forecasting_loss += forecasting_loss.cpu().detach().item()
            avg_adj_penalty += adj_l1.cpu().detach().item()
            avg_dagness_loss += 0.#dagness_loss.cpu().detach().item()# DAGNESS LOSS REMOVED TO ENSURE NUMERICAL STABILITY, ESP. IN D4IC EXPERIMENTS - 12/20/2024
            avg_smooth_loss += smooth.cpu().detach().item()

            del X
            del X_in
            del X_target
            del x_sims
        
        # track training stats
        avg_forecasting_loss = avg_forecasting_loss/len(X_val)
        avg_adj_penalty = avg_adj_penalty/len(X_val)
        avg_dagness_loss = avg_dagness_loss/len(X_val)
        avg_smooth_loss = avg_smooth_loss/len(X_val)
        avg_val_combo_loss = avg_smooth_loss

        print("cLSTM_FM.training_sim_eval: VALIDATION RESULTS: ", flush=True)
        print("cLSTM_FM.training_sim_eval: \t val avg_forecasting_loss == ", avg_forecasting_loss)
        print("cLSTM_FM.training_sim_eval: \t val avg_adj_penalty == ", avg_adj_penalty)
        print("cLSTM_FM.training_sim_eval: \t val avg_dagness_loss == ", avg_dagness_loss)
        print("cLSTM_FM.training_sim_eval: \t val avg_smooth_loss == ", avg_smooth_loss)
        print("cLSTM_FM.training_sim_eval: \t val avg_val_combo_loss == ", avg_val_combo_loss, flush=True)
        print("cLSTM_FM.training_sim_eval: \t SIM EVAL STOP", flush=True)
        if return_loss_componentwise:
            return avg_forecasting_loss, avg_adj_penalty, avg_dagness_loss, avg_smooth_loss, avg_val_combo_loss
        return avg_val_combo_loss


