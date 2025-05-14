import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import os
import pickle as pkl
# import copy

from general_utils.plotting import plot_heatmap


class MLP(nn.Module):
    def __init__(self, num_series, lag, hidden):
        super(MLP, self).__init__()
        self.activation = torch.nn.ReLU()
        self.lag = lag
        # Set up network.
        hidden = hidden+[1]
        layer = nn.Conv1d(num_series, hidden[0], lag) # see https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        nn.init.xavier_uniform_(layer.weight) # see https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/7
        modules = [layer]
        for d_in, d_out in zip(hidden[:-1], hidden[1:]):
            layer = nn.Conv1d(d_in, d_out, 1)
            modules.append(layer)
        # Register parameters.
        self.layers = nn.ModuleList(modules)
        pass

    def forward(self, X):
        out = X.transpose(2, 1)
        for i, fc in enumerate(self.layers):
            if i != 0:
                out = self.activation(out)
            out = fc(out)
        return out.transpose(2, 1)

    def apply_causal_filter(self, X):
        out = X.transpose(2, 1)
        out = self.activation(self.layers[0](out))
        return out.transpose(2, 1)


    
class cMLP(nn.Module):
    def __init__(self, num_chans, lag, hidden, wavelet_level=None, save_path=None):
        '''
        cMLP model with one MLP per time series.
        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
        '''
        super(cMLP, self).__init__()
        self.num_chans = num_chans
        self.wavelet_level = wavelet_level
        self.lag = lag
        if wavelet_level is None:
            self.num_series = num_chans
            self.wavelet_mask = None
        else:
            self.num_series = int(num_chans*(wavelet_level+1))
            # create a wavelet mask for ranking relationships between wavelets
            wavelet_mask = torch.ones(self.num_series, self.num_series)
            wavelets_per_chan = int(self.num_series / self.num_chans)
            assert wavelets_per_chan == 4 # currently only implemented for wavelets_per_chan==4 scenarios (rank_factor would need to be tuned in other cases)
            rank_factor = wavelets_per_chan//4
            # initialize wavelet_submask
            wavelet_submask = torch.ones(wavelets_per_chan, wavelets_per_chan)
            for i in range(wavelets_per_chan):
                wavelet_submask[i,:] = wavelet_submask[i,:]*(1.3**(2.*(rank_factor-1.*i)))
            for i in range(wavelets_per_chan):
                wavelet_submask[:,i] = wavelet_submask[:,i]*(1.3**(2.*(rank_factor-1.*i)))
            # update wavelet_mask with wavelet_submask
            for i in range(self.num_series//wavelets_per_chan):
                for j in range(self.num_series//wavelets_per_chan):
                    wavelet_mask[wavelets_per_chan*i:wavelets_per_chan*(i+1),wavelets_per_chan*j:wavelets_per_chan*(j+1)] = wavelet_submask*wavelet_mask[wavelets_per_chan*i:wavelets_per_chan*(i+1),wavelets_per_chan*j:wavelets_per_chan*(j+1)]
            
            self.wavelet_mask = wavelet_mask
            if torch.cuda.is_available():
                self.wavelet_mask = self.wavelet_mask.to(device="cuda")
            if save_path is not None:
                plot_heatmap(self.wavelet_mask.cpu().data.numpy(), save_path+os.sep+"wavelet_ranking_mask_visualization.png", "Wavelet Ranking Mask", "Affected Channel-Wavelet", "Causal Channel-Wavelet")

        self.activation = torch.nn.ReLU()
        # Set up networks.
        self.networks = nn.ModuleList([MLP(self.num_series, lag, hidden) for _ in range(self.num_series)])
        pass

    
    def forward(self, X):
        '''
        Perform forward pass.
        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        outs = []
        for network in self.networks:
            curr_out = network(X)
            outs.append(curr_out)
        final_out = torch.cat(outs, dim=2)
        return final_out

    def apply_causal_filter(self, X):
        '''
        Perform forward pass.
        Args:
          X: torch tensor of shape (batch, T, p).
        Output: torch tensor of shape (batch, h0, p)
        '''
        outs = []
        for n, network in enumerate(self.networks):
            curr_out = network.apply_causal_filter(X)
            outs.append(curr_out)
        final_out = torch.cat(outs, dim=1).transpose(2,1)
        return final_out
    
    def perform_prox_update_on_GC_weights(self, lam, lr, penalty): # code from original cMLP implementation (see baselines folder)
        '''
        Perform in place proximal update on first layer (GC) weight matrix.
        Args:
        lam: regularization parameter.
        lr: learning rate.
        penalty: one of GL (group lasso), GSGL (group sparse group lasso),
            H (hierarchical).
        '''
        for network in self.networks:
            W = network.layers[0].weight
            hidden, n_series, lag = W.shape
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
        pass

    
    def GC(self, threshold=True, ignore_lag=True, combine_wavelet_representations=False, 
           rank_wavelets=False):
        '''
        Extract learned Granger causality.
        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.
        Returns:
          GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
            indicates whether variable j is Granger causal of variable i. In
            second case, entry (i, j, k) indicates whether it's Granger causal
            at lag k.
        '''
        GC = None
        if ignore_lag:
            GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                  for net in self.networks]
        else:
            GC = [torch.norm(net.layers[0].weight, dim=0)
                  for net in self.networks]
        GC = torch.stack(GC)

        if rank_wavelets:
            assert self.wavelet_mask is not None
            # rank wavelet relationships in GC matrix/tensor
            if ignore_lag:
                GC = self.wavelet_mask*GC
            else:
                assert GC.shape == (self.num_series, self.num_series, self.lag)
                for l in range(self.lag):
                    GC[:,:,l] = self.wavelet_mask*GC[:,:,l]
        
        if self.wavelet_level is not None and combine_wavelet_representations:
            if not ignore_lag:
                assert not ignore_lag # current implementation does not support ignore_lag==True
                assert len(GC.size()) == 3
                condensed_GC = torch.zeros((self.num_chans, self.num_chans, self.lag))
                if torch.cuda.is_available():
                    condensed_GC = condensed_GC.to("cuda")
                for row_c in range(self.num_chans):
                    for col_c in range(self.num_chans):
                        curr_gc_slice = GC[row_c*self.wavelet_level:(row_c+1)*self.wavelet_level, col_c*self.wavelet_level:(col_c+1)*self.wavelet_level, :]
                        condensed_GC[row_c,col_c,:] = condensed_GC[row_c,col_c,:] + torch.sum(torch.sum(curr_gc_slice, 0, keepdim=True), 1, keepdim=True)[0,0,:]
                GC = condensed_GC
            else:
                condensed_GC = torch.zeros((self.num_chans, self.num_chans))
                if torch.cuda.is_available():
                    condensed_GC = condensed_GC.to("cuda")
                for row_c in range(self.num_chans):
                    for col_c in range(self.num_chans):
                        curr_gc_slice = GC[row_c*self.wavelet_level:(row_c+1)*self.wavelet_level, col_c*self.wavelet_level:(col_c+1)*self.wavelet_level]
                        condensed_GC[row_c,col_c] = condensed_GC[row_c,col_c] + torch.sum(curr_gc_slice)
                GC = condensed_GC

        if threshold:
            return (GC > 0).int()
        return GC
    