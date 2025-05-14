import torch
import torch.nn as nn
import numpy as np

from models.cmlp import MLP
from models.dgcnn import DGCNN_Model
from models.ts_transformer import TSTransformerEncoderClassiregressor



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
        self.temporal_conv1_layers = nn.Sequential(nn.Conv1d(1, hidden_sizes[1], self.temporal_kernel_size), )
        final_linear_layers = []
        # note: we append '1' as the final number of 'hidden' features because each class_net simply outputs a single scalar score corresponding to it's class
        for f_in, f_out in zip([hidden_sizes[1]*self.POST_CONVS_SIZE]+hidden_sizes[2:], hidden_sizes[2:]+[num_out_classes]): 
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
        return out.view(curr_batch_size, self.num_out_classes), None # do not return separate label prediciton

    
    # V6 (single) FACTOR SCORE EMBEDDER.  # IMPLEMENTED TO HANDLE COMPLEX CLASSIFICATION TASKS *AND* TO RESTRICT GENERATIVE CAPACITY OF WEIGHTING MODULE *AND* ISOLATE SUPERVISED FACTORS
class MLPClassifierForSingleObjective(nn.Module):
    def __init__(self, num_series, num_in_timesteps, num_factor_scores, hidden_sizes, use_sigmoid_restriction, sigmoid_eccentricity_coeff=10.):
        super(MLPClassifierForSingleObjective, self).__init__()
        self.num_series = num_series
        self.num_in_timesteps = num_in_timesteps
        self.num_factor_scores = num_factor_scores
        assert len(hidden_sizes) == 1
        self.hidden_sizes = hidden_sizes
        self.flatten = nn.Flatten()
        self.use_sigmoid_restriction = use_sigmoid_restriction
        if use_sigmoid_restriction: 
            print("MLPClassifierForSingleObjective.__init__: NOTICE/WARNING - APPLYING SIGMOID TO FACTOR WEIGHT PREDICTIONS")
            self.sigmoid = nn.Sigmoid()
            self.sigmoid_eccentricity_coeff = sigmoid_eccentricity_coeff
        else:
            self.sigmoid = None
            self.sigmoid_eccentricity_coeff = None
        temporal_kernel_width = num_in_timesteps - ((num_in_timesteps-1) % 2)
        # Set up network
        print("MLPClassifierForSingleObjective.__init__: WARNING - REMOVING BIAS FROM ALL NETWORK PARAMETERS TO MITIGATE NUMERICAL INSTABILITY")
        self.series_embedding_layers = nn.Sequential( 
            nn.Conv2d(1, hidden_sizes[0], (num_series, temporal_kernel_width), stride=1, padding=(0,temporal_kernel_width//2), dilation=1, bias=False), 
            nn.ReLU(), 
            nn.Conv2d(hidden_sizes[0], hidden_sizes[0], (1, num_in_timesteps), stride=1, padding=0, dilation=1, bias=False), 
            nn.ReLU(), 
        )
        self.unsup_factor_weighting_layer = nn.Linear(hidden_sizes[0], num_factor_scores, bias=False)
        pass
    
    def forward(self, X, use_final_activation=True):
        """
        Args:
            X: (batch_size, seq_length, feat_dim)
        Returns:
            output: [(batch_size, num_factor_scores), None]
        """
        curr_batch_size = X.size()[0]
        X = torch.transpose(X, 1, 2)
        X = X.view(curr_batch_size, 1, self.num_series, self.num_in_timesteps) # reshape in prep for 2-d convs
        embedded_recording = self.series_embedding_layers(X)
        embedded_recording = embedded_recording.view(curr_batch_size,-1)
        # generate factor weightings
        out_scores = self.unsup_factor_weighting_layer(embedded_recording)
        out_scores = out_scores.view(curr_batch_size, self.num_factor_scores)
        out_scores = out_scores.view(curr_batch_size, self.num_factor_scores)
        if self.sigmoid is not None:
            # the sigmoid is ment to restrict generative capabilities of the weighting module, with the eccentricity coefficient 
            # encouraging gradients to push the parameter values away from activations in the range of -1 to 1
            out_scores = self.sigmoid(self.sigmoid_eccentricity_coeff*out_scores) 
        return out_scores, None # return separate factor score / predicted labels 


# V10 (UPDATED) FACTOR SCORE EMBEDDER.  # IMPLEMENTED TO HANDLE COMPLEX CLASSIFICATION TASKS *AND* TO RESTRICT GENERATIVE CAPACITY OF WEIGHTING MODULE *AND* ISOLATE SUPERVISED FACTORS
class MLPClassifierForMultipleObjectives(nn.Module):
    def __init__(self, num_series, num_in_timesteps, num_factor_scores, num_out_classes, hidden_sizes, use_sigmoid_restriction, sigmoid_eccentricity_coeff=10.):
        print("MLPClassifierForMultipleObjectives.__init__: START")
        print("MLPClassifierForMultipleObjectives.__init__: num_series == ", num_series)
        print("MLPClassifierForMultipleObjectives.__init__: num_in_timesteps == ", num_in_timesteps)
        print("MLPClassifierForMultipleObjectives.__init__: num_factor_scores == ", num_factor_scores)
        print("MLPClassifierForMultipleObjectives.__init__: num_out_classes == ", num_out_classes)
        print("MLPClassifierForMultipleObjectives.__init__: hidden_sizes == ", hidden_sizes)
        print("MLPClassifierForMultipleObjectives.__init__: use_sigmoid_restriction == ", use_sigmoid_restriction)
        print("MLPClassifierForMultipleObjectives.__init__: sigmoid_eccentricity_coeff == ", sigmoid_eccentricity_coeff)
        super(MLPClassifierForMultipleObjectives, self).__init__()
        self.num_series = num_series
        self.num_in_timesteps = num_in_timesteps
        self.num_factor_scores = num_factor_scores
        self.num_out_classes = num_out_classes
        assert len(hidden_sizes) == 1
        self.hidden_sizes = hidden_sizes
        self.flatten = nn.Flatten()
        self.use_sigmoid_restriction = use_sigmoid_restriction
        if use_sigmoid_restriction: 
            print("MLPClassifierForMultipleObjectives.__init__: NOTICE/WARNING - APPLYING SIGMOID TO BOTH CLASSIFIER AND FACTOR WEIGHT PREDICTIONS")
            self.sigmoid = nn.Sigmoid()
            self.sigmoid_eccentricity_coeff = sigmoid_eccentricity_coeff
        else:
            self.sigmoid = None
            self.sigmoid_eccentricity_coeff = None
        
        # Set up network
        temporal_kernel_width = num_in_timesteps - ((num_in_timesteps-1) % 2)
        print("MLPClassifierForMultipleObjectives.__init__: WARNING - REMOVING BIAS FROM ALL NETWORK PARAMETERS TO MITIGATE NUMERICAL INSTABILITY")
        self.series_embedding_layers = nn.Sequential( 
            nn.Conv2d(1, hidden_sizes[0], (num_series, temporal_kernel_width), stride=1, padding=(0,temporal_kernel_width//2), dilation=1, bias=False), 
            nn.ReLU(), 
            nn.Conv2d(hidden_sizes[0], hidden_sizes[0], (1, num_in_timesteps), stride=1, padding=0, dilation=1, bias=False), 
            nn.ReLU(), 
        )
        if num_factor_scores-num_out_classes > 0:
            self.unsup_factor_weighting_layer = nn.Linear(hidden_sizes[0]-num_out_classes, num_factor_scores-num_out_classes, bias=False)
        else:
            self.unsup_factor_weighting_layer = None
        pass
    
    
    def forward(self, X, use_final_activation=True):
        """
        Args:
            X: (batch_size, seq_length, feat_dim)
        Returns:
            output: [(batch_size, num_factor_scores), (batch_size, num_out_classes)]
        """
        curr_batch_size = X.size()[0]
        X = torch.transpose(X, 1, 2)
        X = X.view(curr_batch_size, 1, self.num_series, self.num_in_timesteps) # reshape in prep for 2-d convs
        embedded_recording = self.series_embedding_layers(X)
        embedded_recording = embedded_recording.view(curr_batch_size,-1)
        
        # generate factor weightings
        sup_out_scores = embedded_recording[:,:self.num_out_classes]
        if self.num_factor_scores-self.num_out_classes > 0:
            unsup_out_scores = self.unsup_factor_weighting_layer(embedded_recording[:,self.num_out_classes:])
            unsup_out_scores = unsup_out_scores.view(curr_batch_size, self.num_factor_scores-self.num_out_classes)
            out_scores = torch.cat((sup_out_scores,unsup_out_scores), 1)
        else:
            out_scores = sup_out_scores
        out_scores = out_scores.view(curr_batch_size, self.num_factor_scores)
        if self.sigmoid is not None: 
            # the sigmoid is ment to restrict generative capabilities of the weighting module, with the eccentricity coefficient 
            # encouraging gradients to push the parameter values away from activations in the range of -1 to 1
            out_scores = self.sigmoid(self.sigmoid_eccentricity_coeff*out_scores) 
            
        # generate class predictions
        out_class_logits = embedded_recording[:,:self.num_out_classes]
        if use_final_activation and self.sigmoid is not None:
            out_class_logits = self.sigmoid(out_class_logits)
            
        return out_scores, out_class_logits # return separate factor score / predicted labels 


    
class cEmbedder(nn.Module):
    def __init__(self, num_chans, num_class_preds, num_factor_preds, use_sigmoid_restriction, sigmoid_eccentricity_coeff, lag, hidden, wavelet_level=None, save_path=None):
        '''
        Modified cMLP model with one MLP per factor pred.
        Arg Notes:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
        '''
        super(cEmbedder, self).__init__()
        self.num_chans = num_chans
        self.num_class_preds = num_class_preds
        self.num_factor_preds = num_factor_preds
        self.use_sigmoid_restriction = use_sigmoid_restriction
        self.sigmoid_eccentricity_coeff = sigmoid_eccentricity_coeff
        self.lag = lag
        self.hidden = hidden
        self.wavelet_level = wavelet_level
        self.save_path = save_path
                          
        if wavelet_level is None:
            self.num_series = num_chans
            self.wavelet_mask = None
        else:
            self.num_series = int(num_chans*(wavelet_level+1))
            # create a wavelet mask for ranking relationships between wavelets
            wavelet_mask = torch.ones(self.num_factor_preds, self.num_series)
            wavelets_per_chan = int(self.num_series / self.num_chans)
            assert wavelets_per_chan == 4 # currently only implemented for wavelets_per_chan==4 scenarios (rank_factor would need to be tuned in other cases)
            rank_factor = wavelets_per_chan//4
            # initialize wavelet_submask
            wavelet_submask = torch.ones(1, wavelets_per_chan)
            for i in range(1):
                wavelet_submask[i,:] = wavelet_submask[i,:]*(1.3**(2.*(rank_factor-1.*i)))
            for i in range(wavelets_per_chan):
                wavelet_submask[:,i] = wavelet_submask[:,i]*(1.3**(2.*(rank_factor-1.*i)))
            # update wavelet_mask with wavelet_submask
            for i in range(self.num_factor_preds):
                for j in range(self.num_series//wavelets_per_chan):
                    wavelet_mask[i:i+1,wavelets_per_chan*j:wavelets_per_chan*(j+1)] = wavelet_submask*wavelet_mask[i:i+1,wavelets_per_chan*j:wavelets_per_chan*(j+1)]
            
            self.wavelet_mask = wavelet_mask
            if torch.cuda.is_available():
                self.wavelet_mask = self.wavelet_mask.to(device="cuda")
            if save_path is not None:
                plot_heatmap(self.wavelet_mask.cpu().data.numpy(), save_path+os.sep+"wavelet_ranking_mask_visualization.png", "Wavelet Ranking Mask", "Affected Factor", "Causal Channel-Wavelet")

        self.activation = torch.nn.ReLU()
        if use_sigmoid_restriction: 
            print("cEmbedder.__init__: NOTICE/WARNING - APPLYING SIGMOID TO BOTH CLASSIFIER AND FACTOR WEIGHT PREDICTIONS")
            self.sigmoid = nn.Sigmoid()
            self.sigmoid_eccentricity_coeff = sigmoid_eccentricity_coeff
        else:
            self.sigmoid = None
            self.sigmoid_eccentricity_coeff = None
                          
        # Set up networks.
        self.networks = nn.ModuleList([MLP(self.num_series, lag, hidden) for _ in range(self.num_factor_preds)])
        pass

    
    def forward(self, X, use_final_activation=True):
        '''
        Perform forward pass.
        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        curr_batch_size = X.size()[0]
                          
        # generate factor weightings
        weightings = []
        for network in self.networks:
            curr_w = network(X)
            weightings.append(curr_w)
        factor_weightings = torch.cat(weightings, dim=2).view(curr_batch_size,self.num_factor_preds)
        
        # generate class predictions
        out_class_logits = None
        if self.num_class_preds > 0:
            out_class_logits = factor_weightings[:,:self.num_class_preds]
            if use_final_activation and self.sigmoid is not None:
                # don't apply eccentricity coefficient to class predictions, as it is intended to flatten multi-step factor weighting predictions
                out_class_logits = self.sigmoid(out_class_logits) 
                          
        if self.sigmoid is not None:
            # the sigmoid is ment to restrict generative capabilities of the weighting module, with the eccentricity coefficient 
            # encouraging gradients to push the parameter values away from activations in the range of -1 to 1
            factor_weightings = self.sigmoid(self.sigmoid_eccentricity_coeff*factor_weightings) 
                          
        return factor_weightings, out_class_logits

    
    def GC(self, threshold=True, ignore_lag=True, combine_wavelet_representations=False, 
           rank_wavelets=False):
        '''
        Extract learned Granger causality.
        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.
        Returns:
          GC: (num_factor_preds x p) or (num_factor_preds x p x lag) matrix. In 
            first case, entry (i, j) indicates whether variable j is Granger 
            causal of factor weight i. In second case, entry (i, j, k) 
            indicates the strength of the granger causal relationship at lag k.
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
    
    
                          
class DGCNN_Embedder(nn.Module):
    def __init__(self, num_channels, num_wavelets_per_chan, num_features_per_node, num_graph_conv_layers, num_hidden_nodes, sigmoid_eccentricity_coeff, 
                 use_sigmoid_restriction, num_factors, num_classes):
        super(DGCNN_Embedder, self).__init__()
        self.dgcnn = DGCNN_Model(num_channels, num_wavelets_per_chan, num_features_per_node, num_graph_conv_layers, num_hidden_nodes, num_factors)
        print("DGCNN_Embedder.__init__: num_channels == ", num_channels)
        print("DGCNN_Embedder.__init__: num_wavelets_per_chan == ", num_wavelets_per_chan)
        print("DGCNN_Embedder.__init__: num_features_per_node == ", num_features_per_node)
        print("DGCNN_Embedder.__init__: num_graph_conv_layers == ", num_graph_conv_layers)
        print("DGCNN_Embedder.__init__: num_hidden_nodes == ", num_hidden_nodes)
        print("DGCNN_Embedder.__init__: num_factors == ", num_factors)
        print("DGCNN_Embedder.__init__: num_classes == ", num_classes)
        self.num_channels = num_channels
        self.num_wavelets_per_chan = num_wavelets_per_chan
        self.num_features_per_node = num_features_per_node
        self.num_graph_conv_layers = num_graph_conv_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.num_factors = num_factors
        self.num_classes = num_classes
        self.use_sigmoid_restriction = use_sigmoid_restriction
        self.sigmoid_eccentricity_coeff = sigmoid_eccentricity_coeff
        if use_sigmoid_restriction: 
            print("DGCNN_Embedder.__init__: NOTICE/WARNING - APPLYING SIGMOID TO BOTH CLASSIFIER AND FACTOR WEIGHT PREDICTIONS")
            self.sigmoid = nn.Sigmoid()
            self.sigmoid_eccentricity_coeff = sigmoid_eccentricity_coeff
        else:
            self.sigmoid = None
            self.sigmoid_eccentricity_coeff = None
        pass
    
    
    def forward(self, X, use_final_activation=True):
        # generate factor weightings
        assert len(X.size()) == 3
        if X.size()[2] != self.num_features_per_node:
            assert X.size()[1] == self.num_features_per_node
            X = torch.transpose(X, 1, 2)
        factor_weightings = self.dgcnn(X)
        #factor_weightings = torch.ones(factor_weightings.size(), requires_grad=False).to(X.device) # FOR ABLATION PURPOSES
                          
        # generate class predictions
        out_class_logits = None
        if self.num_classes > 0:
            out_class_logits = factor_weightings[:,:self.num_classes]
            if use_final_activation and self.sigmoid is not None:
                # don't apply eccentricity coefficient to class predictions, as it is intended to flatten multi-step factor weighting predictions
                out_class_logits = self.sigmoid(out_class_logits) 
                          
        if self.sigmoid is not None:
            # the sigmoid is ment to restrict generative capabilities of the weighting module, with the eccentricity coefficient encouraging 
            # gradients to push the parameter values away from activations in the range of -1 to 1
            factor_weightings = self.sigmoid(self.sigmoid_eccentricity_coeff*factor_weightings) 
                          
        return factor_weightings, out_class_logits
    
    
    def GC(self, threshold=True, combine_node_feature_edges=False):
        return self.dgcnn.GC(threshold=threshold, combine_node_feature_edges=combine_node_feature_edges)