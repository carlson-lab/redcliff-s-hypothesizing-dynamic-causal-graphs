import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import os
import pickle as pkl

import pandas as pd

from general_utils.plotting import plot_reconstruction_comparisson, plot_heatmap, plot_curve
from models.causalnex_dynotears import from_numpy_dynamic, _reshape_wa, dynotears_objective


class DYNOTEARS_Model():
    def __init__(self, lambda_w=0.1, lambda_a=0.1, max_iter=100, h_tol=1e-8, w_threshold=0.0, tabu_edges=None, tabu_parent_nodes=None, 
                 tabu_child_nodes=None, grad_step=1.0, wa_est=None, rho=1.0, alpha=0.0, h_value=np.inf, h_new=np.inf, wa_new=None):
        self.lambda_w = lambda_w
        self.lambda_a = lambda_a
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.w_threshold = w_threshold
        self.tabu_edges = tabu_edges
        self.tabu_parent_nodes = tabu_parent_nodes
        self.tabu_child_nodes = tabu_child_nodes
        self.grad_step = grad_step
        self.rho = rho 
        self.alpha = alpha
        self.h_value = h_value
        self.h_new = h_new
        self.wa_new = wa_new
        self.wa_est = wa_est
        self.w_est = None
        self.a_est = None
        self.d_vars, self.p_orders, self.n = None, None, None
        pass
   
    def GC(self,):
        assert self.d_vars is not None
        assert self.p_orders is not None
        w_mat, a_mat = _reshape_wa(self.wa_est, self.d_vars, self.p_orders)
        return a_mat
    
    def save_checkpoint(self, save_dir, it, val_avg_loss_history, best_loss, best_it, best_model):
        # save summary stats
        temp_model_save_path = os.path.join(save_dir, "final_best_model.pkl")
        torch.save(best_model, temp_model_save_path)
        meta_data_save_path = os.path.join(save_dir, "training_meta_data_and_hyper_parameters.pkl")
        with open(meta_data_save_path, "wb") as outfile:
            pkl.dump({
                "epoch": it, 
                "val_avg_loss_history": val_avg_loss_history, 
                "best_loss": best_loss, 
                "best_it": best_it, 
            }, outfile)
        # plot loss histories
        plot_curve(val_avg_loss_history, "Validation Loss", "Epoch", "Average Loss", save_dir+os.sep+"avg_val_loss.png", domain_start=0)
        # make GC Estimate visualizations
        GC_est = [self.GC()]
        plot_heatmap(GC_est[0], save_dir+os.sep+"gc_est_noLags_results_epoch"+str(it)+".png", "Estimated Causal Graph", "", "")
        pass

    
    def fit(self, save_path, max_data_iter, X_train, X_val, iter_start=0, lag_size=1, num_iters_prior_to_stop=10, reuse_rho=False, 
            reuse_alpha=False, reuse_h_val=False, reuse_h_new=False, GC_orig=None, check_every=5, reuse_wa_new=False):
        val_avg_loss_history = []
        best_it = None
        best_loss = np.inf
        best_model = None
        if GC_orig is not None:
            gc_combo = None
            for i, gc in enumerate(GC_orig):
                if i == 0:
                    gc_combo = np.zeros(gc.reshape((gc.shape[0], -1)).shape)
                plot_heatmap(gc.reshape((gc.shape[0], -1)), save_path+os.sep+"eval_gc_factor"+str(i)+"_orig_visualization.png", "Flattened True Causal Graph", "", "")
                gc_combo = gc_combo + gc.reshape((gc.shape[0], -1))
            plot_heatmap(gc_combo, save_path+os.sep+"eval_gc_factorSum_orig_visualization.png", "Flattened True Causal Graph", "", "")

        for it in range(iter_start, max_data_iter):
            if it%5 == 0:
                print("DYNOTEARS_Model.fit: now on epoch it == ", it, flush=True)

            for batch_num, (X, Y) in enumerate(X_train):
                X_in = X[:,:-1*lag_size,:].detach().numpy()
                X_lag = X[:,lag_size:,:].detach().numpy()
                for b in range(X_in.shape[0]):
                    curr_x = X_in[b,:,:]
                    curr_x_lag = X_lag[b,:,:]
                    _, self.w_est, self.a_est, self.wa_est, rho, alpha, h_value, h_new, wa_new, self.n, self.d_vars, self.p_orders = from_numpy_dynamic(
                        curr_x, curr_x_lag, lambda_w=self.lambda_w, lambda_a=self.lambda_a, max_iter=self.max_iter, h_tol=self.h_tol, 
                        w_threshold=self.w_threshold, tabu_edges=self.tabu_edges, tabu_parent_nodes=self.tabu_parent_nodes, 
                        tabu_child_nodes=self.tabu_child_nodes, grad_step=self.grad_step, wa_est=self.wa_est, rho=self.rho, alpha=self.alpha, 
                        h_value=self.h_value, h_new=self.h_new, wa_new=self.wa_new
                    )
                    if reuse_rho:
                        self.rho = rho
                    if reuse_alpha:
                        self.alpha = alpha
                    if reuse_h_val:
                        self.h_value = h_value
                    if reuse_h_new:
                        self.h_new = h_new
                    if reuse_wa_new:
                        self.wa_new = wa_new
            
            # perform model validation
            curr_avg_val_loss = 0.
            sample_counter = 0.
            for batch_num, (X, Y) in enumerate(X_val):
                X_in = X[:,:-1*lag_size,:].detach().numpy()
                X_lag = X[:,lag_size:,:].detach().numpy()
                for b in range(X_in.shape[0]):
                    curr_x = X_in[b,:,:]
                    curr_x_lag = X_lag[b,:,:]
                    curr_avg_val_loss += dynotears_objective(
                        curr_x, curr_x_lag, self.wa_est, self.rho, self.alpha, self.d_vars, self.p_orders, self.lambda_a, self.lambda_w, self.n
                    )
                    sample_counter += 1.

            curr_avg_val_loss /= sample_counter
            val_avg_loss_history.append(curr_avg_val_loss)

            # check stopping criteria
            if curr_avg_val_loss < best_loss: # FOR DEBUGGING PURPOSES
                best_loss = curr_avg_val_loss
                best_it = it
                best_model = deepcopy(self)
            elif (it - best_it) == num_iters_prior_to_stop: # FOR DEBUGGING PURPOSES
                print('Stopping early')                # FOR DEBUGGING PURPOSES
                break    
            
            # Check progress.
            if it % check_every == 0:
                print("NCFM_WITH_CMLP_FACTORS.fit: \t CHECKING")
                print("NCFM_WITH_CMLP_FACTORS.fit: \t val_avg_loss_history == ", val_avg_loss_history)
                # save checkpoint
                self.save_checkpoint(
                    save_path, it, val_avg_loss_history, best_loss, best_it, best_model
                )
            pass

        # Restore best model.
        final_save_path = os.path.join(save_path, "final_best_model.pkl")
        torch.save(best_model, final_save_path)
        # Report final Validation Score(s)
        final_avg_val_loss = best_model.evaluate(X_val, save_path, lag_size=lag_size)
        print("FINAL BEST (STOPPING CRITERIA) LOSS == ", best_loss, flush=True)
        print("FINAL BEST (STOPPING CRITERIA) EPOCH == ", best_it, flush=True)
        print("FINAL VALIDATION COMBO LOSS == ", final_avg_val_loss, flush=True)
        return final_avg_val_loss

    
    def evaluate(self, X_orig, save_path, lag_size=1):
        GC_est = self.GC()
        plot_heatmap(GC_est, save_path+os.sep+"eval_gc_est_visualization.png", "Estimated Causal Graph", "", "")
        curr_avg_val_loss = 0.
        sample_counter = 0.
        for batch_num, (X, Y) in enumerate(X_orig):
            X_in = X[:,:-1*lag_size,:].detach().numpy()
            X_lag = X[:,lag_size:,:].detach().numpy()
            for b in range(X_in.shape[0]):
                curr_x = X_in[b,:,:]
                curr_x_lag = X_lag[b,:,:]
                curr_avg_val_loss += dynotears_objective(
                    curr_x, curr_x_lag, self.wa_est, self.rho, self.alpha, self.d_vars, self.p_orders, self.lambda_a, self.lambda_w, self.n
                )
                sample_counter += 1.
        curr_avg_val_loss /= sample_counter
        return curr_avg_val_loss
