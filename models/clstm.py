import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import os
import pickle as pkl

from general_utils.plotting import plot_heatmap



class LSTM(nn.Module):
    def __init__(self, num_series, hidden):
        '''
        LSTM model with output layer to generate predictions.
        Args:
          num_series: number of input time series.
          hidden: number of hidden units.
        '''
        super(LSTM, self).__init__()
        self.num_series = num_series
        self.hidden = hidden
        # Set up network.
        self.lstm = nn.LSTM(num_series, hidden, batch_first=True)
        self.lstm.flatten_parameters()
        self.linear = nn.Conv1d(hidden, 1, 1)

    def init_hidden(self, batch):
        '''Initialize hidden states for LSTM cell.'''
        device = self.lstm.weight_ih_l0.device
        return (torch.zeros(1, batch, self.hidden, device=device),
                torch.zeros(1, batch, self.hidden, device=device))

    def forward(self, X, hidden=None):
        # Set up hidden state.
        if hidden is None:
            hidden = self.init_hidden(X.shape[0])
        # Apply LSTM.
        X, hidden = self.lstm(X, hidden)
        # Calculate predictions using output layer.
        X = X.transpose(2, 1)
        X = self.linear(X)
        return X.transpose(2, 1), hidden


class cLSTM(nn.Module):
    def __init__(self, num_chans, hidden, wavelet_level=None, save_path=None):
        '''
        cLSTM model with one LSTM per time series.
        Args:
          num_series: dimensionality of multivariate time series.
          hidden: number of units in LSTM cell.
        '''
        super(cLSTM, self).__init__()
        self.num_chans = num_chans
        self.wavelet_level = wavelet_level
        print("models.clstm.cLSTM.__init__: self.num_chans == ", self.num_chans)
        print("models.clstm.cLSTM.__init__: self.wavelet_level == ", self.wavelet_level)

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
       
        print("models.clstm.cLSTM.__init__: self.num_series == ", self.num_series)
        print("models.clstm.cLSTM.__init__: self.wavelet_mask == ", self.wavelet_mask)

        # Set up networks.
        self.networks = nn.ModuleList([
            LSTM(self.num_series, hidden) for _ in range(self.num_series)
        ])
        pass

    
    def forward(self, X, hidden=None):
        '''
        Perform forward pass.
        Args:
          X: torch tensor of shape (batch, T, p).
          hidden: hidden states for LSTM cell.
        '''
        if hidden is None:
            hidden = [None for _ in range(self.num_series)]
        pred = [self.networks[i](X, hidden[i]) for i in range(self.num_series)]
        pred, hidden = zip(*pred)
        pred = torch.cat(pred, dim=2)
        return pred, hidden
    
    def perform_prox_update_on_GC_weights(self, lam, lr): # code from original cMLP implementation
        '''
        Perform in place proximal update on first layer (GC) weight matrix.
        '''
        for network in self.networks:
            W = network.lstm.weight_ih_l0
            norm = torch.norm(W, dim=0, keepdim=True)
            W.data = ((W / torch.clamp(norm, min=(lam * lr))) * torch.clamp(norm - (lr * lam), min=0.0))
            network.lstm.flatten_parameters()
        pass

    
    def GC(self, threshold=True, combine_wavelet_representations=False, rank_wavelets=False):
        '''
        Extract learned Granger causality.
        Args:
          threshold: return norm of weights, or whether norm is nonzero.
        Returns:
          GC: (p x p) matrix. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        '''
        GC = [torch.norm(net.lstm.weight_ih_l0, dim=0) for net in self.networks]
        GC = torch.stack(GC)

        if rank_wavelets:
            assert self.wavelet_mask is not None
            # rank wavelet relationships in GC matrix/tensor
            GC = self.wavelet_mask*GC
        
        if self.wavelet_level is not None and combine_wavelet_representations:
            assert len(GC.size()) == 2
            condensed_GC = torch.zeros((self.num_chans, self.num_chans))
            if torch.cuda.is_available():
                condensed_GC = condensed_GC.to("cuda")
            for row_c in range(self.num_chans):
                for col_c in range(self.num_chans):
                    curr_gc_slice = GC[row_c*self.wavelet_level:(row_c+1)*self.wavelet_level, col_c*self.wavelet_level:(col_c+1)*self.wavelet_level]
                    condensed_GC[row_c,col_c] = condensed_GC[row_c,col_c] + torch.sum(torch.sum(curr_gc_slice, 0, keepdim=True), 1, keepdim=True)[0,0]
            GC = condensed_GC
        
        if threshold:
            return (GC > 0).int()
        return GC