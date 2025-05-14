# based on https://torcheeg.readthedocs.io/en/latest/generated/torcheeg.models.DGCNN.html and https://github.com/xueyunlong12589/DGCNN/tree/main
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import os
import pickle as pkl

from torcheeg.models import DGCNN as baseDGCNN
from general_utils.model_utils import restore_parameters
from general_utils.plotting import plot_curve, plot_gc_est_comparissons_by_factor



class DGCNN_Model(nn.Module):
    def __init__(self, num_channels, num_wavelets_per_chan, num_features_per_node, num_graph_conv_layers, num_hidden_nodes, num_classes):
        '''
        DGCNN_Model model with helper functions.
        '''
        print("models.dgcnn.DGCNN_Model.__init__: START", flush=True)
        super(DGCNN_Model, self).__init__()
        self.num_channels = num_channels
        self.num_wavelets_per_chan = num_wavelets_per_chan
        self.num_nodes = num_channels * num_wavelets_per_chan
        self.num_features_per_node = num_features_per_node
        self.num_graph_conv_layers = num_graph_conv_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.num_classes = num_classes
        self.supervised_loss_fn = nn.MSELoss(reduction='mean')
        print("models.dgcnn.DGCNN_Model.__init__: num_channels == ", num_channels)
        print("models.dgcnn.DGCNN_Model.__init__: num_wavelets_per_chan == ", num_wavelets_per_chan)
        print("models.dgcnn.DGCNN_Model.__init__: num_features_per_node == ", num_features_per_node)
        print("models.dgcnn.DGCNN_Model.__init__: num_graph_conv_layers == ", num_graph_conv_layers)
        print("models.dgcnn.DGCNN_Model.__init__: num_hidden_nodes == ", num_hidden_nodes)
        print("models.dgcnn.DGCNN_Model.__init__: num_classes == ", num_classes)
        print("models.dgcnn.DGCNN_Model.__init__: self.num_nodes == ", self.num_nodes)
        self.dgcnn = baseDGCNN(
            num_features_per_node, 
            self.num_nodes, 
            num_graph_conv_layers, 
            num_hidden_nodes, 
            num_classes
        )
        print("models.dgcnn.DGCNN_Model.__init__: STOP", flush=True)
        pass
    
    def GC(self, threshold=True, combine_node_feature_edges=False):
        GC = self.dgcnn.A
        if combine_node_feature_edges:
            condensed_GC = torch.zeros((self.num_channels, self.num_channels))
            if torch.cuda.is_available():
                condensed_GC = condensed_GC.to("cuda")
            for i in range(self.num_channels):
                for j in range(self.num_channels):
                    condensed_GC[i,j] = condensed_GC[i,j] + torch.norm(GC[i*self.num_wavelets_per_chan:(i+1)*self.num_wavelets_per_chan,j*self.num_wavelets_per_chan:(j+1)*self.num_wavelets_per_chan])
            GC = condensed_GC
        # UPDATED 03/27/2024 - TRANSPOSE OPERATION INCLUDED AFTER EXPERIMENTS SUGGESTED THE ESTIMATED GC MATRIX OBTAINED HIGHER COSINE-SIMILARITY SCORES WITH GROUND TRUTH AFTER THE TRANSPOSE OPERATION
        GC = GC.T
        if threshold:
            return (GC > 0).int()
        return GC
    
    def forward(self, X):
        return self.dgcnn.forward(X)

    
    def batch_update(self, batch_num, X, Y, gen_optim, running_factor_loss):   
        # Set up data.
        if torch.cuda.is_available():
            X = X.to(device="cuda")
            Y = Y.to(device="cuda")
        
        # UPDATE factor models and factor score embedder
        # Prep parameters for updates
        self.dgcnn.train()
        gen_optim.zero_grad()
        # make predictions/forecast
        Y_pred = self.forward(torch.transpose(X[:,:self.num_features_per_node,:], 1,2))

        # Calculate loss 
        factor_loss = None
        if len(Y.size()) == 3:
            if Y.size()[2] > self.num_features_per_node:
                factor_loss = self.supervised_loss_fn(Y_pred, Y[:,:,self.num_features_per_node])
            else: # case reached on certain datasets, e.g. DREAM4-based multi-state simulations
                assert Y.size()[2] == 1
                factor_loss = self.supervised_loss_fn(Y_pred, Y[:,:,0])
        elif len(Y.size()) == 2: # case for DREAM4 orig. data
            factor_loss = self.supervised_loss_fn(Y_pred, Y)
        else:
            raise NotImplementedError("Cannot handle ground-truth labels with Y.size() == "+str(Y.size()))

        # update parameters
        factor_loss.backward()
        gen_optim.step()

        running_factor_loss += factor_loss.cpu().detach().item()
        del X
        del Y
        del Y_pred
        return running_factor_loss

    
    def save_checkpoint(self, save_dir, it, best_model, avg_factor_loss, best_loss, GC):
        temp_model_save_path = os.path.join(save_dir, "temp_best_model_epoch"+str(it)+".bin")
        torch.save(best_model, temp_model_save_path)
        meta_data_save_path = os.path.join(save_dir, "training_meta_data_and_hyper_parameters.pkl")
        with open(meta_data_save_path, "wb") as outfile:
            pkl.dump({
                "epoch": it, 
                "avg_factor_loss": avg_factor_loss, 
                "best_loss": best_loss, 
            }, outfile)
        
        plot_curve(avg_factor_loss, "Training Loss", "Epoch", "Average Loss", save_dir+os.sep+"avg_training_loss_epoch"+str(it)+".png", domain_start=0)
        GC_est = self.GC(threshold=False)
        GC_est = [GC_est.cpu().data.numpy()]
        plot_gc_est_comparissons_by_factor(GC, GC_est, save_dir+os.sep+"gc_est_results_epoch"+str(it)+".png")
        pass
    
    
    def fit(self, save_dir, train_loader, gen_optim, max_iter, lookback, check_every, verbose, GC, val_loader): # <><><>
        # For tracking intermediate/preliminary results
        avg_factor_loss = []
        # For early stopping.
        best_it = None
        best_loss = np.inf
        best_model = None

        for it in range(max_iter):
            print("DGCNN_Model.fit: now on epoch it == ", it, flush=True)
            
            # initialize vars for tracking stats
            running_factor_loss = 0.
            for batch_num, (X, Y) in enumerate(train_loader): 
                running_factor_loss = self.batch_update(
                    batch_num, X, Y, gen_optim, running_factor_loss, 
                )
            
            # track training stats
            avg_factor_loss.append(running_factor_loss/len(train_loader))

            # Check progress.
            if it % check_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("DGCNN_Model.fit: \t CHECKING")
                print("DGCNN_Model.fit: \t avg_factor_loss == ", avg_factor_loss)

                mean_val_loss = self.training_eval(val_loader) #<><><>

                if verbose > 0:
                    print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1), flush=True)
                    print('Validation Loss = %f' % mean_val_loss)
                    curr_gc_ests = self.GC()
                    for gc_est_num in range(1):
                        print('Factor '+str(gc_est_num)+' Variable usage = %.2f%%' % (100 * torch.mean(curr_gc_ests[gc_est_num].float())))

                # Check for early stopping.
                curr_gc_est = self.GC(threshold=False, combine_node_feature_edges=False)
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
                    save_dir, it, best_model, avg_factor_loss, best_loss, GC
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            pass

        # Restore best model.
        restore_parameters(self, best_model)
        final_save_path = os.path.join(save_dir, "final_best_model.bin")
        torch.save(self, final_save_path)
        # Report final Validation Score(s)
        final_mean_val_loss = self.training_eval(val_loader)
        print("FINAL VALIDATION LOSS == ", final_mean_val_loss, flush=True)
        return final_mean_val_loss

    
    def training_eval(self, val_loader):
        # initialize vars for tracking intermediate/preliminary results
        avg_factor_loss = 0.
        self.dgcnn.eval()
        for batch_num, (X, Y) in enumerate(val_loader):
            # Set up data.
            if torch.cuda.is_available():
                X = X.to(device="cuda")
                Y = Y.to(device="cuda")

            # make predictions/forecast
            Y_pred = self.forward(torch.transpose(X[:,:self.num_features_per_node,:], 1,2))

            # Calculate loss 
            factor_loss = None
            if len(Y.size()) == 3: 
                if Y.size()[2] > self.num_features_per_node:
                    factor_loss = self.supervised_loss_fn(Y_pred, Y[:,:,self.num_features_per_node])
                else: # case reached on certain datasets, e.g. DREAM4-based
                    assert Y.size()[2] == 1
                    factor_loss = self.supervised_loss_fn(Y_pred, Y[:,:,0])
            elif len(Y.size()) == 2: # case for DREAM4 orig. data
                factor_loss = self.supervised_loss_fn(Y_pred, Y)
            else:
                raise NotImplementedError("Cannot handle ground-truth labels with Y.size() == "+str(Y.size()))

            avg_factor_loss += factor_loss.cpu().detach().item()
            del X
            del Y
            del Y_pred
        
        # track training stats
        avg_factor_loss = avg_factor_loss/len(val_loader)
        print("DGCNN_Model.training_eval: VALIDATION RESULTS: ", flush=True)
        print("DGCNN_Model.training_eval: \t val avg_factor_loss == ", avg_factor_loss)
        return avg_factor_loss


