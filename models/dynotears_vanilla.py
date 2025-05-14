import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import os
import pickle as pkl

import pandas as pd

from general_utils.plotting import plot_heatmap
from models.causalnex_dynotears_vanilla import from_numpy_dynamic


class DYNOTEARS_Model():
    def __init__(self, lambda_w=0.1, lambda_a=0.1, max_iter=100, h_tol=1e-8, w_threshold=0.0, tabu_edges=None, tabu_parent_nodes=None, tabu_child_nodes=None):
        self.lambda_w = lambda_w
        self.lambda_a = lambda_a
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.w_threshold = w_threshold
        self.tabu_edges = tabu_edges
        self.tabu_parent_nodes = tabu_parent_nodes
        self.tabu_child_nodes = tabu_child_nodes
        self.a_est = None
        pass
   
    def GC(self,):
        return self.a_est
    
    def save_checkpoint(self, save_dir):
        # save summary stats
        temp_model_save_path = os.path.join(save_dir, "final_best_model.pkl")
        torch.save(self, temp_model_save_path)
        # make GC Estimate visualizations
        GC_est = [self.GC()]
        plot_heatmap(GC_est[0], save_dir+os.sep+"gc_est_noLags_results_epoch"+str(it)+".png", "Estimated Causal Graph", "", "")
        pass

    
    def fit(self, save_path, X_train, X_val, lag_size=1, GC_orig=None, save_a_est=True):
        gc_combo = None
        if GC_orig is not None:
            for i, gc in enumerate(GC_orig):
                if i == 0:
                    gc_combo = np.zeros(gc.reshape((gc.shape[0], -1)).shape)
                plot_heatmap(gc.reshape((gc.shape[0], -1)), save_path+os.sep+"eval_gc_factor"+str(i)+"_orig_visualization.png", "Flattened True Causal Graph", "", "")
                gc_combo = gc_combo + gc.reshape((gc.shape[0], -1))
            plot_heatmap(gc_combo, save_path+os.sep+"eval_gc_factorSum_orig_visualization.png", "Flattened True Causal Graph", "", "")
        
        num_samps = X_train.shape[0]
        num_nodes_in_graph = X_train.shape[-1]
        final_a_est = np.zeros((num_nodes_in_graph, num_nodes_in_graph))
        for s in range(num_samps):
            X_in = X_train[s,:-1*lag_size,:]
            X_lag = X_train[s,lag_size:,:]
            _, _, curr_a_est = from_numpy_dynamic( 
                X_in, X_lag, self.lambda_w, self.lambda_a, self.max_iter, self.h_tol, self.w_threshold, self.tabu_edges, self.tabu_parent_nodes, self.tabu_child_nodes
            )
            final_a_est = final_a_est + curr_a_est
        final_a_est = final_a_est / (1.0 * num_nodes_in_graph)

        if save_a_est:
            self.a_est = final_a_est

        final_avg_val_loss = None
        if gc_combo is not None:
            final_avg_val_loss = self.evaluate(X_val, save_path, GC_orig, lag_size=lag_size)
        final_save_path = os.path.join(save_path, "final_best_model.pkl")
        torch.save(self, final_save_path)
        return final_avg_val_loss

    
    def evaluate(self, X, save_path, GC_orig, lag_size=1):
        print("dynotears_vanilla.DYNOTEARS_Model.evaluate: WARNING - evaluate function NOT implemented, returning None!!!!", flush=True)
        return None
