import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import NMF

try:
    from torchbd.loss import BetaDivLoss
    TORCHBD_INSTALLED = True
except ModuleNotFoundError:
    TORCHBD_INSTALLED = False

from scipy.stats import mannwhitneyu

import os
import pickle as pkl
from general_utils.plotting import plot_gc_est_comparissons_by_factor, plot_reconstruction_comparisson



class NmfBase(nn.Module):
    """
    Base class for DcsfaNmf models

    Raises
    ------
    * ``ValueError``: if fixed corr values are not in {"positive","negative","n/a"}

    Parameters
    ----------
    n_components : int
        number of networks to learn or latent dimensionality
    device : str, optional
        torch device in {"cuda","cpu","auto"}. Defaults to 'auto'.
    n_sup_networks : int, optional
        Number of networks that will be supervised
        ``(0 < n_sup_networks < n_components)``. Defaults to ``1``.
    fixed_corr : list of str, optional
        List the same length as n_sup_networks indicating correlation constraints
        for the network. Defaults to None.
        "positive" - constrains a supervised network to have a positive correlation
        between score and label
        "negative" - constrains a supervised network to have a negative correlation
        between score and label
        "n/a" - no constraint is applied and the supervised network can be positive
        or negatively correlated.
    recon_loss : str, optional
        Reconstruction loss function in ``{"IS","MSE"}``. Defaults to ``'MSE'``.
    recon_weight : float, optional
        Importance weight for the reconstruction. Defaults to 1.0.
    sup_recon_type : str, optional
        Which supervised component reconstruction loss to use in {"Residual","All"}.
        Defaults to ``"Residual"``.
        "Residual" - Estimates network scores optimal for reconstruction and
        penalizes deviation of the real scores from those values
        "All" - Evaluates the recon_loss of the supervised network reconstruction
        against all features.
    sup_recon_weight : float, optional
        Importance weight for the reconstruction of the supervised component.
        Defaults to ``1.0``.
    sup_smoothness_weight : float, optional
        Encourages smoothness for the supervised network. Defaults to ``1.0``.
    feature_groups : list of int, optional
        Indices of the divisions of feature types. Defaults to None.
    group_weights : list of floats, optional
        Weights for each of the feature types. Defaults to None.
    verbose : bool, optional
        Activates or deactivates print statements globally. Defaults to False.
    """

    def __init__(self, n_components, device="auto", n_sup_networks=1, fixed_corr=None, recon_loss="MSE", recon_weight=1.0, 
                 sup_recon_type="Residual", sup_recon_weight=1.0, sup_smoothness_weight=1, feature_groups=None, group_weights=None, 
                 verbose=False,):
        super(NmfBase, self).__init__()
        self.n_components = n_components
        self.n_sup_networks = n_sup_networks
        self.recon_loss = recon_loss
        self.recon_weight = recon_weight
        self.sup_recon_type = sup_recon_type
        self.sup_smoothness_weight = sup_smoothness_weight
        self.sup_recon_weight = sup_recon_weight
        self.verbose = verbose
        self.recon_loss_f = self.get_recon_loss(recon_loss)
        # Set correlation constraints
        if fixed_corr == None:
            self.fixed_corr = ["n/a" for sup_net in range(self.n_sup_networks)]
        elif type(fixed_corr) != list:
            if fixed_corr.lower() == "positive":
                self.fixed_corr = ["positive"]
            elif fixed_corr.lower() == "negative":
                self.fixed_corr = ["negative"]
            elif fixed_corr.lower() == "n/a":
                self.fixed_corr = ["n/a"]
            else:
                raise ValueError("fixed corr must be a list or in {`positive`,`negative`,`n/a`}")
        else:
            assert len(fixed_corr) == len(range(self.n_sup_networks))
            self.fixed_corr = fixed_corr
        self.feature_groups = feature_groups
        if feature_groups is not None and group_weights is None:
            group_weights = []
            for (lb, ub) in feature_groups:
                group_weights.append((feature_groups[-1][-1] - feature_groups[0][0]) / (ub - lb))
            self.group_weights = group_weights
        else:
            self.group_weights = group_weights

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

            
    def _initialize_NMF(self, dim_in):
        """
        Instantiates the NMF decoder and moves the NmfBase instance to self.device

        Parameters
        ----------
        dim_in : int
            Number of total features
        """
        self.W_nmf = nn.Parameter(torch.rand(self.n_components, dim_in))
        self.to(self.device)

        
    @staticmethod
    def inverse_softplus(x, eps=1e-5):
        """
        Gets the inverse softplus for sklearn model pretraining
        """
        # Calculate inverse softplus
        x_inv_softplus = np.log(np.exp(x + eps) - (1.0 - eps))
        # Return inverse softplus
        return x_inv_softplus

    def get_W_nmf(self):
        """
        Passes the W_nmf parameter through a softplus function to make it non-negative
        """
        return nn.Softplus()(self.W_nmf)

    
    @torch.no_grad()
    def get_recon_loss(self, recon_loss):
        """
        Returns the reconstruction loss function

        Parameters
        ----------
        recon_loss : str in {"MSE","IS"}
            Identifies which loss function to use
        """
        if recon_loss == "MSE":
            return nn.MSELoss()
        elif recon_loss == "IS":
            assert TORCHBD_INSTALLED, "torchbd needs to be installed!"
            return BetaDivLoss(beta=0, reduction="mean")
        else:
            raise ValueError(f"{recon_loss} is not supported")

            
    @torch.no_grad()
    def get_optim(self, optim_name):
        """
        returns a torch optimizer based on text input from the user
        """
        if optim_name == "AdamW":
            return torch.optim.AdamW
        elif optim_name == "Adam":
            return torch.optim.Adam
        elif optim_name == "SGD":
            return torch.optim.SGD
        else:
            raise ValueError(f"{optim_name} is not supported")

            
    @torch.no_grad()
    def pretrain_NMF(self, X, y, nmf_max_iter=100):
        """
        Trains an unsupervised NMF model and sorts components by predictiveness for corresponding tasks.
        Saved NMF components are stored as a member variable self.W_nmf. Sklearn NMF model is saved as NMF_init.

        Parameters
        ----------
        X : numpy.ndarray
            Input features
            Shape: ``[n_samps,n_features]``
        y : numpy.ndarray
            Input labels
            Shape: ``[n_samps,n_sup_networks]``
        nmf_max_iter : int
            Maximum iterations for convergence using sklearn.decomposition.NMF
        """
        if self.verbose:
            print("Pretraining NMF...")
        # Initialize the model - solver corresponds to defined recon loss
        if self.recon_loss == "IS":
            self.NMF_init = NMF(
                n_components=self.n_components,
                solver="mu",
                beta_loss="itakura-saito",
                init="nndsvda",
                max_iter=nmf_max_iter,
            )
        else:
            self.NMF_init = NMF(
                n_components=self.n_components, max_iter=nmf_max_iter, init="nndsvd"
            )
        # Fit the model
        s_NMF = self.NMF_init.fit_transform(X)
        # define arrays for storing model predictive information
        selected_networks = []
        selected_aucs = []
        final_network_order = []
        # Find the most predictive component for each task - the first task gets priority
        for sup_net in range(self.n_sup_networks):
            # Prep labels and set auc storage
            class_auc_list = []
            # Create progress bar if verbose
            if self.verbose:
                print("Identifying predictive components for supervised network {}".format(sup_net))
                component_iter = tqdm(range(self.n_components))
            else:
                component_iter = range(self.n_components)
            # Find each components AUC for the current task
            for component in component_iter:
                s_pos = s_NMF[y[:, sup_net] >= 0.6, component].reshape(-1, 1)
                s_neg = s_NMF[y[:, sup_net] < 0.6, component].reshape(-1, 1)
                U, _ = mannwhitneyu(s_pos, s_neg)
                U = U.squeeze()
                auc = U / (len(s_pos) * len(s_neg))
                class_auc_list.append(auc)
            class_auc_list = np.array(class_auc_list)
            # Sort AUC predictions based on correlations
            predictive_order = np.argsort(np.abs(class_auc_list - 0.5))[::-1]
            positive_predictive_order = np.argsort(class_auc_list)[::-1]
            negative_predictive_order = np.argsort(1 - class_auc_list)[::-1]
            # Ignore components that have been used for previous supervised tasks
            if len(selected_networks) > 0:
                for taken_network in selected_networks:
                    predictive_order = predictive_order[predictive_order != taken_network]
                    positive_predictive_order = positive_predictive_order[positive_predictive_order != taken_network]
                    negative_predictive_order = negative_predictive_order[negative_predictive_order != taken_network]
            # Select the component based on predictive correlation
            if self.fixed_corr[sup_net] == "n/a":
                current_net = predictive_order[0]
            elif self.fixed_corr[sup_net].lower() == "positive":
                current_net = positive_predictive_order[0]
            elif self.fixed_corr[sup_net].lower() == "negative":
                current_net = negative_predictive_order[0]
            current_auc = class_auc_list[current_net.astype(int)]
            # Declare the selected network and save the network and chosen AUCs
            if self.verbose:
                print("Selecting network: {} with auc {} for sup net {} using constraint {} correlation".format(current_net, current_auc, sup_net, self.fixed_corr[sup_net]))
            selected_networks.append(current_net)
            selected_aucs.append(selected_aucs)
            predictive_order = predictive_order[predictive_order != current_net]
            positive_predictive_order = positive_predictive_order[positive_predictive_order != current_net]
            negative_predictive_order = negative_predictive_order[negative_predictive_order != current_net]
        # Save the selected networks and corresponding AUCs
        self.skl_pretrain_networks_ = selected_networks
        self.skl_pretrain_aucs_ = selected_aucs
        final_network_order = selected_networks
        # Get the final sorting of the networks for predictiveness
        for idx, _ in enumerate(predictive_order):
            final_network_order.append(predictive_order[idx])
        # Save the final sorted components to the W_nmf parameter
        sorted_NMF = self.NMF_init.components_[final_network_order]
        self.W_nmf.data = torch.from_numpy(self.inverse_softplus(sorted_NMF.astype(np.float32))).to(self.device)

        
    def get_sup_recon(self, s):
        """
        Returns the reconstruction of all of the supervised networks

        Parameters
        ----------
        s : torch.Tensor.float()
            Factor activation scores
            Shape: ``[n_samps,n_components]``

        Returns
        -------
        X_sup_recon : torch.Tensor.float()
            Reconstruction using only supervised components
            Shape: ``[n_samps,n_features]``
        """
        X_sup_recon = s[:, : self.n_sup_networks].view(-1, self.n_sup_networks) @ self.get_W_nmf()[: self.n_sup_networks, :].view(self.n_sup_networks, -1)
        return X_sup_recon

    
    def get_residual_scores(self, X, s):
        """
        Returns the supervised score values that would maximize reconstruction performance based on the residual reconstruction.
        s_h = (X - s_unsup @ W_unsup) @ w_sup.T @ (w_sup @ w_sup.T)^(-1)

        Parameters
        ----------
        X : torch.Tensor
            Ground truth features
            Shape: ``[n_samples,n_features]``
        s : torch.Tensor
            Factor activation scores
            Shape: ``[n_samples,n_components]``

        Returns
        -------
        NOTE: HERE
        """
        resid = (X - s[:, self.n_sup_networks :] @ self.get_W_nmf()[self.n_sup_networks :, :])
        w_sup = self.get_W_nmf()[: self.n_sup_networks, :].view(self.n_sup_networks, -1)
        s_h = resid @ w_sup.T @ torch.inverse(w_sup @ w_sup.T)
        return s_h

    
    def residual_loss_f(self, s, s_h):
        """
        Loss function between supervised factor scores and the maximal values for reconstruction. Factors scores are encouraged to be non-zero by the smoothness weight.
        f(s,s_h) = ||s_sup - s_h||^2 / (1 - smoothness_weight * exp(-||s_h||^2))

        Parameters
        ----------
        s : torch.Tensor
            Factor activation scores
            Shape: ``[n_samples,n_components]``
        s_h : torch.Tensor
            Factor scores that would minimize the reconstruction loss
            Shape: ``[n_samples,n_components]``

        Returns
        ---------
        res_loss : torch.Tensor
            Residual scores loss
        """
        res_loss = torch.norm(s[:, : self.n_sup_networks].view(-1, self.n_sup_networks) - s_h) / (1 - self.sup_smoothness_weight * torch.exp(-torch.norm(s_h)))
        return res_loss

    
    def get_weighted_recon_loss_f(self, X_pred, X_true):
        """
        Model training often involves multiple feature types such as Power and Directed Spectrum that have vastly different feature counts
        ``(power: n_roi*n_freq, ds: n_roi*(n_roi-1)*n_freq)``.

        This loss reweights the reconstruction of each feature group proportionally to the number of features such that each feature type has roughly equal importance
        to the reconstruction.

        Parameters
        ----------
        X_pred : torch.Tensor
            Reconstructed features
            Shape: ``[n_samps,n_features]``
        X_true : torch.Tensor
            Ground Truth Features
            Shape: ``[n_samps,n_features]``

        Returns
        ---------
        recon_loss : torch.Tensor
            Weighted reconstruction loss for each feature
        """
        recon_loss = 0.0
        for weight, (lb, ub) in zip(self.group_weights, self.feature_groups):
            recon_loss += weight * self.recon_loss_f(X_pred[:, lb:ub], X_true[:, lb:ub])
        return recon_loss

    
    def eval_recon_loss(self, X_pred, X_true):
        """
        If using feature groups, returns weighted recon loss
        Else, returns unweighted recon loss

        Parameters
        ----------
        X_pred : torch.Tensor
            Reconstructed features
            Shape: ``[n_samps,n_features]``
        X_true : torch.Tensor
            Ground Truth Features
            Shape: ``[n_samps,n_features]``

        Returns
        ---------
        recon_loss : torch.Tensor
            Weighted or Unweighted recon loss
        """
        if self.feature_groups is None:
            recon_loss = self.recon_loss_f(X_pred, X_true)
        else:
            recon_loss = self.get_weighted_recon_loss_f(X_pred, X_true)
        return recon_loss

    
    def NMF_decoder_forward(self, X, s):
        """
        NMF Decoder forward pass

        Parameters
        ----------
        X : torch.Tensor
            Input Features
            Shape: ``[n_samps,n_features]``
        s : torch.Tensor
            Encoder embeddings / factor score activations
            Shape: ``[n_samps,n_components]``

        Returns
        -------
        recon_loss : torch.Tensor
            Whole data recon loss + supervised recon loss
        """
        recon_loss = 0.0
        X_recon = s @ self.get_W_nmf()
        recon_loss += self.recon_weight * self.eval_recon_loss(X_recon, X)

        if self.sup_recon_type == "Residual":
            s_h = self.get_residual_scores(X, s)
            sup_recon_loss = self.residual_loss_f(s, s_h)
        elif self.sup_recon_type == "All":
            X_recon = self.get_sup_recon(s)
            sup_recon_loss = self.recon_loss_f(X_recon, X)

        recon_loss += self.sup_recon_weight * sup_recon_loss
        return recon_loss
    

    @torch.no_grad()
    def get_comp_recon(self, s, component):
        """
        Gets the reconstruction for a specific component

        Parameters
        ----------
        s : torch.Tensor
            Encoder embeddings / factor score activations
            Shape: ``[n_samps,n_components]``
        component : int, 0 <= component < n_components
            Component to use for the reconstruction

        Returns
        ---------
        X_recon : numpy.ndarray
            Reconstruction using a specific component
        """
        X_recon = s[:, component].view(-1, 1) @ self.get_W_nmf()[component, :].view(1, -1)
        return X_recon.detach().cpu().numpy()

    
    @torch.no_grad()
    def get_all_comp_recon(self, s):
        """
        Gets the reconstruction for all components

        Parameters
        ----------
        s : torch.Tensor
            Encoder embeddings / factor score activations
            Shape: ``[n_samps,n_components]``

        Returns
        ---------
        X_recon : numpy.ndarray
            Reconstruction using a specific component
        """
        X_recon = s @ self.get_W_nmf()
        return X_recon

    
    @torch.no_grad()
    def get_factor(self, component):
        """
        Returns the numpy array for the corresponding factor

        Parameters
        ----------
        component : int, 0 <= component < n_components
            Component to use for the reconstruction

        Returns
        ---------
        factor : np.ndarray
            Factor from W_nmf
        """
        return self.get_W_nmf()[component, :].detach().cpu().numpy()


    
class DcsfaNmf(NmfBase):
    """
    dCSFA-NMF model

    Parameters
    ----------
    n_components : int, optional
        number of networks to learn or latent dimensionality
        NOTE: the sklearn convention is for all parameters to have default
        values.
    device : ``{'cuda','cpu','auto'}``, optional
        Torch device. Defaults to ``'auto'``.
    n_intercepts : int, optional
        Number of unique intercepts for logistic regression
        (sometimes mouse specific intercept is helpful). Defaults to ``1``.
        NOTE: can this be detected automatically, given a groups argument?
    n_sup_networks : int, optional
        Number of networks that will be supervised
        ``0 < n_sup_networks < n_components``. Defaults to ``1``.
    optim_name : ``{'SGD','Adam','AdamW'}``, optional
        torch.optim algorithm to use in . Defaults to ``'AdamW'``.
    recon_loss : ``{'IS', 'MSE'}``, optional
        Reconstruction loss function. Defaults to ``'MSE'``.
    recon_weight : float, optional
        Importance weight for the reconstruction. Defaults to ``1.0``.
    sup_weight : float, optional
        Importance weight for the supervision. Defaults to ``1.0``.
    sup_recon_weight : float, optional
        Importance weight for the reconstruction of the supervised component.
        Defaults to ``1.0``.
    use_deep_encoder : bool, optional
        NOTE: I've changed the name to use underscores
        Whether to use a deep or linear encoder. Defaults to ``True``.
    h : int, optional
        Hidden layer size for the deep encoder. Defaults to ``256``.
    sup_recon_type : ``{'Residual', 'All'}``, optional
        Which supervised component reconstruction loss to use. Defaults to
        ``'Residual'``. ``'Residual'`` estimates network scores optimal for
        reconstruction and penalizes deviation of the real scores from those
        values. ``'All'`` evaluates the reconstruction loss of the supervised
        network reconstruction against all features.
    feature_groups : ``None`` or list of int, optional
        Indices of the divisions of feature types. Defaults to ``None``.
    group_weights : ``None`` or list of floats, optional
        Weights for each of the feature types. Defaults to ``None``.
    fixed_corr : ``None`` or list of str, optional
        List the same length as ``n_sup_networks`` indicating correlation
        constraints for the network. Defaults to ``None``. ``'positive'``
        constrains a supervised network to have a positive correlation between
        score and label. ``'negative'`` constrains a supervised network to have
        a negative correlation between score and label. ``'n/a'`` applies no
        constraint, meaning the supervised network can be positive or
        negatively correlated.
    momentum : float, optional
        Momentum value if optimizer is ``'SGD'``. Defaults to ``0.9``.
    lr : float, optional
        Learning rate for the optimizer. Defaults to ``1e-3``.
    sup_smoothness_weight : float, optional
        Encourages smoothness for the supervised network. Defaults to ``1.0``.
    save_folder : str, optional
        Location to save the best pytorch model parameters. Defaults to
        ``'~'``. NOTE: this seems like a bad default. Maybe a ``save_filename``
        parameter would be better.
    verbose : bool, optional
        Activates or deactivates print statements. Defaults to ``False``.
    """

    def __init__(self, n_components=32, device="auto", n_intercepts=1, n_sup_networks=1, optim_name="AdamW", recon_loss="MSE", 
                 recon_weight=1.0, sup_weight=1.0, sup_recon_weight=1.0, use_deep_encoder=True, h=256, sup_recon_type="Residual", 
                 feature_groups=None, group_weights=None, fixed_corr=None, momentum=0.9, lr=1e-3, sup_smoothness_weight=1.0, 
                 save_folder="~", verbose=False,):
        super(DcsfaNmf, self).__init__(
            n_components, device, n_sup_networks, fixed_corr, recon_loss, recon_weight, sup_recon_type, sup_recon_weight,
            sup_smoothness_weight, feature_groups, group_weights, verbose,
        )
        self.n_intercepts = n_intercepts
        self.optim_name = optim_name
        self.optim_alg = self.get_optim(optim_name)
        self.pred_loss_f = nn.BCELoss
        self.recon_weight = recon_weight
        self.sup_weight = sup_weight
        self.use_deep_encoder = use_deep_encoder
        self.h = h
        self.momentum = momentum
        self.lr = lr
        self.sup_smoothness_weight = sup_smoothness_weight
        self.save_folder = save_folder

        
    def _initialize(self, dim_in):
        """
        Initializes encoder and NMF parameters using the input dimensionality

        Parameters
        ----------
        dim_in : int
            Total number of features
        """
        # NOTE: the sklearn convention is for attributes assigned after the
        # __init__ function to have trailing underscores. It doesn't look like
        # this attribute is every used, though, so maybe it should be removed.
        self.dim_in_ = dim_in
        self._initialize_NMF(dim_in)
        if self.use_deep_encoder:
            self.encoder = nn.Sequential(
                nn.Linear(dim_in, self.h),
                nn.BatchNorm1d(self.h),
                nn.LeakyReLU(),
                nn.Linear(self.h, self.n_components),
                nn.Softplus(),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(dim_in, self.n_components),
                nn.Softplus(),
            )
        # Initialize logistic regression parameters.
        self.phi_list = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(self.n_sup_networks)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.randn(self.n_intercepts, 1)) for _ in range(self.n_sup_networks)])
        self.to(self.device)

        
    def instantiate_optimizer(self):
        """
        Create an optimizer.

        Returns
        -------
        optimizer : torch.optim.Optimizer
            Torch optimizer
        """
        if self.optim_name == "SGD":
            optimizer = self.optim_alg(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
            )
        else:
            optimizer = self.optim_alg(self.parameters(), lr=self.lr)
        return optimizer
    

    def get_all_class_predictions(self, X, s, intercept_mask, avg_intercept):
        """
        Get predictions for every supervised network.

        Parameters
        ----------
        X : torch.Tensor
            Input features
            Shape: ``[batch_size,n_features]``
        s : torch.Tensor
            latent embeddings
            Shape: ``[batch_size,n_components]``
        intercept_mask : torch.Tensor
            One-hot encoded mask for logistic regression intercept terms
            Shape: ``[batch_size,n_intercepts]``
        avg_intercept : bool
            Whether to average all intercepts together.

        Returns
        -------
        y_pred_proba : torch.Tensor
            Predictions
            Shape: ``[batch_size, n_sup_networks]``
        """
        if intercept_mask is None and avg_intercept is False:
            warnings.warn(
                f"Intercept mask cannot be none and avg_intercept False... "
                f"Averaging Intercepts",
            )
            avg_intercept = True
        # Get predictions for each class.
        y_pred_list = []
        for sup_net in range(self.n_sup_networks):
            if self.n_intercepts == 1:
                # NOTE: use torch.sigmoid instead of torch.nn.Sigmoid
                y_pred_proba = torch.sigmoid(
                    s[:, sup_net].view(-1, 1) * self.get_phi(sup_net)
                    + self.beta_list[sup_net],
                ).squeeze()
            elif self.n_intercepts > 1 and not avg_intercept:
                y_pred_proba = torch.sigmoid(
                    s[:, sup_net].view(-1, 1) * self.get_phi(sup_net)
                    + intercept_mask @ self.beta_list[sup_net],
                ).squeeze()
            else:
                intercept_mask = (
                    torch.ones(X.shape[0], self.n_intercepts).to(self.device)
                    / self.n_intercepts
                )
                y_pred_proba = torch.sigmoid(
                    s[:, sup_net].view(-1, 1) * self.get_phi(sup_net)
                    + intercept_mask @ self.beta_list[sup_net]
                ).squeeze()
            y_pred_list.append(y_pred_proba.view(-1, 1))
        # Concatenate predictions into a single matrix [n_samples,n_tasks]
        y_pred_proba = torch.cat(y_pred_list, dim=1)
        return y_pred_proba

    
    def get_embedding(self, X):
        """
        Map features to latents

        Parameters
        ----------
        X : torch.Tensor
            Input features
            Shape: ``[batch_size,n_features]``

        Returns
        -------
        s : torch.Tensor
            Latent embeddings (scores)
            Shape: ``[batch_size,n_components]``
        """
        return self.encoder(X)
    
    
    def get_phi(self, sup_net=0):
        """
        Return the logistic regression coefficient correspoding to ``sup_net``.
        # NOTE: Raises before Parameters and Returns

        Raises
        ------
        * ``ValueError`` if ``self.fixed_corr[sup_net]`` is not in
          ``{'n/a', 'positive', 'negative'}``. This should be caught at
          initialization. # NOTE: it's better to catch this a initialization.

        Parameters
        ----------
        sup_net : int, optional
            Which supervised network you would like to get a coefficient for.
            Defaults to ``0``.

        Returns
        -------
        phi : torch.Tensor
            The coefficient that has either been returned raw, or through a
            positive or negative softplus(phi).
        """
        fixed_corr_str = self.fixed_corr[sup_net].lower()
        if fixed_corr_str == "n/a":
            return self.phi_list[sup_net]
        elif fixed_corr_str == "positive":
            # NOTE: use nn.functional instead of nn here.
            return F.softplus(self.phi_list[sup_net])
        elif fixed_corr_str == "negative":
            return -1 * F.softplus(self.phi_list[sup_net])
        else:
            raise ValueError(f"Unsupported fixed_corr value: {fixed_corr_str}")

            
    def forward(self, X, y, task_mask, pred_weight, intercept_mask=None, avg_intercept=False):
        """
        dCSFA-NMF forward pass

        Parameters
        ----------
        X : torch.Tensor
            Input Features
            Shape: ``[batch_size,n_features]``
        y : torch.Tensor
            Ground truth labels
            Shape: ``[batch_size,n_sup_networks]``
        task_mask : torch.Tensor
            Per window mask for whether or not predictions should be counted
            Shape: ``[batch_size,n_sup_networks]``
        pred_weight : torch.Tensor
            Per window classification importance weighting
            Shape: ``[batch_size,1]``
        intercept_mask : ``None`` or torch.Tensor, optional
            Window specific intercept mask. Defaults to ``None``.
            Shape: ``[batch_size,n_intercepts]``
        avg_intercept : bool, optional
            Whether or not to average intercepts. This is used in evaluation.
            Defaults to ``False``.

        Returns
        -------
        recon_loss (torch.Tensor):
            ``recon_weight*full_recon_loss + sup_recon_weight*sup_recon_loss``
        pred_loss (torch.Tensor):
            ``sup_weight * BCELoss()``
        """
        X = X.to(self.device)
        y = y.to(self.device)
        task_mask = task_mask.to(self.device)
        pred_weight = pred_weight.to(self.device)
        if intercept_mask is not None:
            intercept_mask = intercept_mask.to(self.device)

        # Get the scores from the encoder
        s = self.get_embedding(X)
        # Get the reconstruction losses
        recon_loss = self.NMF_decoder_forward(X, s)
        # Get predictions
        y_pred = self.get_all_class_predictions(X, s, intercept_mask, avg_intercept,)
        pred_loss_f = self.pred_loss_f(weight=pred_weight)
        pred_loss = self.sup_weight * pred_loss_f(y_pred * task_mask, y * task_mask,)

        # recon loss and pred loss are left seperate so only recon_loss can be applied for encoder pretraining
        return recon_loss, pred_loss

    
    @torch.no_grad()
    def transform(self, X, intercept_mask=None, avg_intercept=True, return_npy=True):
        """
        Transform method to return reconstruction, predictions, and projections

        Parameters
        ----------
        X (torch.Tensor): Input Features
            Shape: ``[batch_size,n_features]``
        intercept_mask (torch.Tensor, optional): window specific intercept
            mask. Defaults to None.
            Shape: ``[batch_size,n_intercepts]``
        avg_intercept (bool, optional): Whether or not to average intercepts -
            used in evaluation. Defaults to False.
        return_npy (bool, optional): Whether or not to convert to numpy arrays.
            Defaults to True.

        Returns
        -------
        X_recon (torch.Tensor) : Full reconstruction of input features
            Shape: ``[n_samples,n_features]``
        y_pred (torch.Tensor) : All task predictions
            Shape: ``[n_samples,n_sup_networks]``
        s (torch.Tensor) : Network activation scores
            Shape: ``[n_samples,n_components]``
        """
        if not torch.is_tensor(X):
            X = torch.Tensor(X).float().to(self.device)
        else:
            X = X.to(self.device)

        if intercept_mask is not None:
            intercept_mask = torch.Tensor(intercept_mask).to(self.device)

        s = self.get_embedding(X)
        X_recon = self.get_all_comp_recon(s)
        y_pred = self.get_all_class_predictions(X, s, intercept_mask, avg_intercept,)
        if return_npy:
            s = s.detach().cpu().numpy()
            X_recon = X_recon.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
        return X_recon, y_pred, s

    
    def pretrain_encoder(self, X, y, y_pred_weights, task_mask, intercept_mask, sample_weights, n_pre_epochs=100, batch_size=128,):
        """
        Pretrain the encoder.

        Parameters
        ----------
        X (torch.Tensor): Input Features
            Shape: ``[n_samples,n_features]``
        y (torch.Tensor): ground truth labels
            Shape: ``[n_samples,n_sup_networks]``
        task_mask (torch.Tensor):
            per window mask for whether or not predictions should be counted
            Shape: ``[n_samples,n_sup_networks]``
        y_pred_weight (torch.Tensor):
            per window classification importance weighting
            Shape: ``[n_samples,1]``
        intercept_mask (torch.Tensor, optional):
            window specific intercept mask. Defaults to None.
            Shape: ``[n_samples,n_intercepts]``
        sample_weights (torch.Tensor):
            Gradient Descent sampling weights.
            Shape: ``[n_samples,1]
        n_pre_epochs (int,optional):
            number of epochs for pretraining the encoder. Defaults to 100
        batch_size (int,optional):
            batch size for pretraining. Defaults to 128
        """
        # Freeze the decoder
        self.W_nmf.requires_grad = False
        # Load arguments onto device
        X = torch.Tensor(X).float().to("cpu")
        y = torch.Tensor(y).float().to("cpu")
        y_pred_weights = torch.Tensor(y_pred_weights).float().to("cpu")
        task_mask = torch.Tensor(task_mask).long().to("cpu")
        intercept_mask = torch.Tensor(intercept_mask).to("cpu")
        sample_weights = torch.Tensor(sample_weights).to("cpu")
        # Create a Dataset.
        dset = TensorDataset(X, y, task_mask, y_pred_weights, intercept_mask)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        loader = DataLoader(dset, batch_size=batch_size, sampler=sampler)
        # Instantiate Optimizer
        optimizer = self.instantiate_optimizer()
        # Define iterator
        if self.verbose:
            epoch_iter = tqdm(range(n_pre_epochs))
        else:
            epoch_iter = range(n_pre_epochs)
        for epoch in epoch_iter:
            r_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                recon_loss, _ = self.forward(*batch)
                recon_loss.backward()
                optimizer.step()
                r_loss += recon_loss.item()
            if self.verbose:
                avg_r = r_loss / len(loader)
                epoch_iter.set_description(f"Encoder Pretrain Epoch: {epoch}, Recon Loss: {avg_r:.6}",)
        # Unfreeze the NMF decoder.
        self.W_nmf.requires_grad = True
        

    def fit(self, X, y, y_pred_weights=None, task_mask=None, intercept_mask=None, y_sample_groups=None, n_epochs=100, n_pre_epochs=100, 
            nmf_max_iter=100, batch_size=128, lr=1e-3, pretrain=True, verbose=False, X_val=None, y_val=None, y_pred_weights_val=None, 
            task_mask_val=None, best_model_name="dCSFA-NMF-best-model.pt",):
        """
        Fit the model.
        NOTE: isn't ``lr`` a model parameter? epochs and pre epochs could be too.

        Parameters
        ----------
        X (np.ndarray): Input Features
            Shape: ``[n_samples, n_features]``
        y (np.ndarray): ground truth labels
            Shape: ``[n_samples, n_sup_networks]``
        y_pred_weights (np.ndarray, optional):
            supervision window specific importance weights. Defaults to
            ``None``.
            Shape: ``[n_samples,1]``
        task_mask (np.ndarray, optional):
            identifies which windows should be trained on which tasks. Defaults
            to ``None``.
            Shape: ``[n_samples,n_sup_networks]``
        intercept_mask (np.ndarray, optional):
            One-hot Mask for group specific intercepts in the logistic
            regression model. Defaults to None.
            Shape: ``[n_samples,n_intercepts]``
        y_sample_groups (_type_, optional):
            groups for creating sample weights - each group will be sampled
            evenly. Defaults to None.
            Shape: ``[n_samples,1]``
        n_epochs (int, optional):
            number of training epochs. Defaults to 100.
        n_pre_epochs (int, optional):
            number of pretraining epochs. Defaults to 100.
        nmf_max_iter (int, optional):
            max iterations for NMF pretraining solver. Defaults to 100.
        batch_size (int, optional):
            batch size for gradient descent. Defaults to 128.
        lr (_type_, optional):
            learning rate for gradient descent. Defaults to 1e-3.
        pretrain (bool, optional):
            whether or not to pretrain the generative model. Defaults to True.
        verbose (bool, optional):
            activate or deactivate print statements. Defaults to False.
        X_val (np.ndarray, optional):
            Validation Features for checkpointing. Defaults to None.
            Shape: ``[n_val_samples,n_features]``
        y_val (np.ndarray, optional):
            Validation Labels for checkpointing. Defaults to None.
            Shape: ``[n_val_samples,n_sup_networks]``
        y_pred_weights_val (np.ndarray, optional):
            window specific classification weights. Defaults to None.
            Shape: ``[n_val_samples,1]``
        task_mask_val (np.ndarray, optional):
            validation task relevant window masking. Defaults to None.
            Shape: ``[n_val_samples,n_sup_networks]``
        best_model_name (str, optional):
            save file name for the best model. Must end in ".pt". Defaults to
            ``'dCSFA-NMF-best-model.pt'``.

        Returns
        -------
        self : dCSFA_NMF
            The fitted model. NOTE: sklearn convention
        """
        # Initialize model parameters.
        self._initialize(X.shape[1])

        # Establish loss histories.
        self.training_hist = []  # tracks average overall loss
        self.recon_hist = []  # tracks training data mse
        self.pred_hist = []  # tracks training data aucs

        # Globaly activate/deactivate print statements.
        self.verbose = verbose
        self.lr = lr

        # Fill default values
        if intercept_mask is None:
            intercept_mask = np.ones((X.shape[0], self.n_intercepts))
        if task_mask is None:
            task_mask = np.ones(y.shape)
        if y_pred_weights is None:
            y_pred_weights = np.ones((y.shape[0], 1))

        # Fill sampler parameters.
        if y_sample_groups is None:
            y_sample_groups = np.ones((y.shape[0]))
            samples_weights = y_sample_groups
        else:
            class_sample_counts = np.array([np.sum(y_sample_groups == group) for group in np.unique(y_sample_groups)],)
            weight = 1.0 / class_sample_counts
            samples_weights = np.array([weight[t] for t in y_sample_groups.astype(int)],).squeeze()
            samples_weights = torch.Tensor(samples_weights)

        # Pretrain the model.
        if pretrain:
            self.pretrain_NMF(X, y, nmf_max_iter)
            self.pretrain_encoder(X, y, y_pred_weights, task_mask, intercept_mask, samples_weights, n_pre_epochs, batch_size,)

        # Send training arguments to Tensors.
        X = torch.Tensor(X).float().to("cpu")
        y = torch.Tensor(y).float().to("cpu")
        y_pred_weights = torch.Tensor(y_pred_weights).float().to("cpu")
        task_mask = torch.Tensor(task_mask).long().to("cpu")
        intercept_mask = torch.Tensor(intercept_mask).to("cpu")
        samples_weights = torch.Tensor(samples_weights).to("cpu")

        # If validation data is provided, set up the tensors.
        if X_val is not None and y_val is not None:
            assert best_model_name.split(".")[-1] == "pt", (
                f"Save file `{self.save_folder + best_model_name}` must be "
                f"of type .pt"
            )
            self.best_model_name = best_model_name
            self.best_performance = 1e8
            self.best_val_recon = 1e8
            self.best_val_avg_auc = 0.0
            self.val_recon_hist = []
            self.val_pred_hist = []

            if task_mask_val is None:
                task_mask_val = np.ones(y_val.shape)

            if y_pred_weights_val is None:
                y_pred_weights_val = np.ones((y_val[:, 0].shape[0], 1))

            X_val = torch.Tensor(X_val).float().to("cpu")
            y_val = torch.Tensor(y_val).float().to("cpu")
            task_mask_val = torch.Tensor(task_mask_val).long().to("cpu")
            y_pred_weights_val = (torch.Tensor(y_pred_weights_val,).float().to("cpu"))

        # Instantiate the dataloader and optimizer.
        dset = TensorDataset(X, y, task_mask, y_pred_weights, intercept_mask)
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
        loader = DataLoader(dset, batch_size=batch_size, sampler=sampler)
        optimizer = self.instantiate_optimizer()

        # Define the training iterator.
        if self.verbose:
            print("Beginning Training")
            epoch_iter = tqdm(range(n_epochs))
        else:
            epoch_iter = range(n_epochs)

        # Training loop.
        for epoch in epoch_iter:
            epoch_loss = 0.0
            recon_e_loss = 0.0
            pred_e_loss = 0.0

            for batch in loader:
                self.train()
                optimizer.zero_grad()
                recon_loss, pred_loss = self.forward(*batch)
                # Weighting happens inside of the forward call
                loss = recon_loss + pred_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                recon_e_loss += recon_loss.item()
                pred_e_loss += pred_loss.item()
            self.training_hist.append(epoch_loss / len(loader))
            with torch.no_grad():
                self.eval()
                X_recon, y_pred, _ = self.transform(
                    X,
                    intercept_mask,
                    avg_intercept=False,
                    return_npy=True,
                )
                training_mse_loss = np.mean((X.detach().numpy() - X_recon) ** 2)
                training_auc_list = []
                for sup_net in range(self.n_sup_networks):
                    temp_mask = task_mask[:, sup_net].detach().numpy()
                    auc = roc_auc_score(
                        y.detach().numpy()[temp_mask == 1, sup_net] >= 0.6, 
                        y_pred[temp_mask == 1, sup_net] >= 0.6, 
                    )
                    training_auc_list.append(auc)
                self.recon_hist.append(training_mse_loss)
                self.pred_hist.append(training_auc_list)

                # If validation data is present, collect performance metrics.
                if X_val is not None and y_val is not None:
                    X_recon_val, y_pred_val, _ = self.transform(
                        X_val,
                        return_npy=True,
                    )
                    validation_mse_loss = np.mean(
                        (X_val.detach().numpy() - X_recon_val) ** 2
                    )
                    validation_auc_list = []
                    for sup_net in range(self.n_sup_networks):
                        temp_mask = task_mask_val[:, sup_net].detach().numpy()
                        auc = roc_auc_score(
                            y_val.detach().numpy()[temp_mask == 1, sup_net] >= 0.6, 
                            y_pred_val[temp_mask == 1, sup_net] >= 0.6, 
                        )
                        validation_auc_list.append(auc)

                    self.val_recon_hist.append(validation_mse_loss)
                    self.val_pred_hist.append(validation_auc_list)

                    mse_var_rat = validation_mse_loss / torch.std(X_val) ** 2
                    auc_err = 1 - np.mean(validation_auc_list)

                    if mse_var_rat + auc_err < self.best_performance:
                        self.best_epoch = epoch
                        self.best_performance = mse_var_rat + auc_err
                        self.best_val_avg_auc = np.mean(validation_auc_list)
                        self.best_val_recon = validation_mse_loss
                        self.best_val_aucs = validation_auc_list
                        torch.save(
                            self.state_dict(),
                            self.save_folder +os.sep+ self.best_model_name,
                        )

                    if self.verbose:
                        epoch_iter.set_description(
                            "Epoch: {}, Best Epoch: {}, Best Val MSE: {:.6}, Best Val by Window ROC-AUC {}, current MSE: {:.6}, current AUC: {}".format(
                                epoch, self.best_epoch, self.best_val_recon, self.best_val_aucs, validation_mse_loss, validation_auc_list,
                            )
                        )
                else:
                    if self.verbose:
                        epoch_iter.set_description("Epoch: {}, Current Training MSE: {:.6}, Current Training by Window ROC-AUC: {}".format(epoch, training_mse_loss, training_auc_list))

        if self.verbose:
            print("Saving the last epoch with training MSE: {:.6} and AUCs:{}".format(training_mse_loss, training_auc_list))
            
        # Last epoch is saved in case the saved 'best model' doesn't make sense
        torch.save(self, self.save_folder +os.sep+ "dCSFA-NMF-last-epoch.pt",)

        if X_val is not None and y_val is not None:
            if self.verbose:
                print("Loaded the best model from Epoch: {} with MSE: {:.6} and AUCs: {}".format(self.best_epoch, self.best_val_recon, self.best_val_aucs))
            try:
                self.load_state_dict(torch.load(self.save_folder + self.best_model_name))
            except:
                print("WARNING: COULD NOT FIND MODEL AT "+str(self.save_folder + self.best_model_name)+"; ATTEMPTING TO MODIFY LAST PORTION WITH ADDITIONAL os.sep CALL")
                self.load_state_dict(torch.load(self.save_folder + os.sep + self.best_model_name))
                print("WARNING: UPDATE - MODEL SUCCESSFULLY LOADED FROM "+str(self.save_folder + os.sep + self.best_model_name))
        return self

    
    def load_last_epoch(self):
        """
        Loads model parameters of the last epoch in case the last epoch is preferable to the early stop checkpoint conditions. This will be visible in the training printout if self.verbose=True
        """
        self.load_state_dict(torch.load(self.save_folder + "dCSFA-NMF-last-epoch.pt"),)
        

    def reconstruct(self, X, component=None):
        """
        Gets full or partial reconstruction.

        Parameters
        ----------
        X : numpy.ndarray
            Input Features
            Shape: ``[n_samples,n_features]``
        component : int, optional
            identifies which component to use for reconstruction

        Returns
        -------
        X_recon : numpy.ndarray
            Full recon if component=None, else, recon for network[comopnent]
        """
        X_recon, _, s = self.transform(X)
        if component is not None:
            X_recon = self.get_comp_recon(s, component)
        return X_recon
    

    def predict_proba(self, X, return_scores=False):
        """
        Returns prediction probabilities

        Parameters
        ----------
        X : numpy.ndarray
            Input Features
            Shape: ``[n_samples,n_features]``
        return_scores (bool, optional):
            Whether or not to include the projections. Defaults to False.

        Returns
        -------
        y_pred_proba (numpy.ndarray): predictions
            Shape: ``[n_samples,n_sup_networks]``
        s (numpy.ndarray): supervised network activation scores
            Shape: ``[n_samples,n_components]
        """
        _, y_pred, s = self.transform(X)
        if return_scores:
            return y_pred, s
        else:
            return y_pred
        

    def predict(self, X, return_scores=False):
        """
        Return binned predictions.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input Features
        return_scores : bool, optional
            Whether or not to include the projections. Defaults to ``False``.

        Returns
        -------
        y_pred_proba : numpy.ndarray
            Predictions in {0,1}
            Shape: ``[n_samples,n_sup_networks]``
        s : numpy.ndarray
            supervised network activation scores
            Shape: ``[n_samples,n_components]
        """
        _, y_pred, s = self.transform(X)
        if return_scores:
            return y_pred > 0.5, s
        else:
            return y_pred > 0.5

        
    def project(self, X):
        """
        Get projections

        Parameters
        ----------
        X : numpy.ndarray
            Input Features
            Shape: ``[n_samples,n_features]
        return_scores : bool, optional
            Whether or not to include the projections. Defaults to False.

        Returns
        -------
        s : numpy.ndarray
            supervised network activation scores
            Shape: ``[n_samples,n_components]
        """
        _, _, s = self.transform(X)
        return s
    

    def score(self, X, y, groups=None, return_dict=False):
        """
        Gets a list of task AUCs either by group or for all samples. Can return
        a dictionary with AUCs for each group with each group label as a key.

        Parameters
        ----------
        X : numpy.ndarray
            Input Features
            Shape: ``[n_samples,n_features]
        y : numpy.ndarray
            Ground Truth Labels
            Shape: ``[n_samples,n_sup_networks]``
        groups : numpy.ndarray, optional
            per window group assignment labels. Defaults to None.
            Shape: ``[n_samples,1]``
        return_dict : bool, optional
            Whether or not to return a dictionary with values for each group. Defaults to False.

        Returns
        -------
        score_results: numpy.ndarray or dict
            Array or dictionary of results either as the mean performance of all groups,
            the performance of all samples, or a dictionary of results for each group.
        """
        _, y_pred, _ = self.transform(X)
        if groups is not None:
            auc_dict = {}
            for group in np.unique(groups):
                auc_list = []
                for sup_net in range(self.n_sup_networks):
                    auc = roc_auc_score(y[:, sup_net], y_pred[:, sup_net])
                    auc_list.append(auc)
                auc_dict[group] = auc_list
            if return_dict:
                score_results = auc_dict
            else:
                auc_array = np.vstack([auc_dict[key] for key in np.unique(groups)])
                score_results = np.mean(auc_array, axis=0)
        else:
            auc_list = []
            for sup_net in range(self.n_sup_networks):
                auc = roc_auc_score(y[:, sup_net], y_pred[:, sup_net])
                auc_list.append(auc)
            score_results = np.array(auc_list)
        return score_results


    
# <><><> ##############################################################################################################
class FullDCSFAModel(DcsfaNmf):
    
    def __init__(self, num_nodes=5, num_high_level_node_features=25, n_components=4, n_sup_networks=4, h=100, save_folder="", 
        momentum=0.9, lr=1e-3, device="auto", n_intercepts=1, optim_name="AdamW", recon_loss="MSE", recon_weight=1.0, 
        sup_weight=1.0, sup_recon_weight=1.0, use_deep_encoder=True, sup_recon_type="Residual", feature_groups=None, 
        group_weights=None, fixed_corr=None, sup_smoothness_weight=1.0, verbose=False,):
        super(FullDCSFAModel, self).__init__(
            n_components=n_components, device=device, n_intercepts=n_intercepts, n_sup_networks=n_sup_networks, 
            optim_name=optim_name, recon_loss=recon_loss, recon_weight=recon_weight, sup_weight=sup_weight, 
            sup_recon_weight=sup_recon_weight, use_deep_encoder=use_deep_encoder, h=h, sup_recon_type=sup_recon_type, 
            feature_groups=feature_groups, group_weights=group_weights, fixed_corr=fixed_corr, momentum=momentum, lr=lr, 
            sup_smoothness_weight=sup_smoothness_weight, save_folder=save_folder, verbose=verbose,
        )
        self.num_nodes = num_nodes
        self.num_high_level_node_features = num_high_level_node_features
        pass   

    
    def get_factor_GC(self, factor, threshold=True, ignore_features=True):
        raw_adjacency_tensor = np.reshape(factor, (self.num_nodes, self.num_nodes, self.num_high_level_node_features))
        GC = raw_adjacency_tensor*raw_adjacency_tensor # get the l-2 norm of each element
        if ignore_features:
            GC = np.sum(GC, axis=2)

        if threshold:
            return (GC > 0).int()
        return GC
    
    
    def GC(self, threshold=True, ignore_features=True):
        '''
        Extract learned Granger causality from each factor.
        Args:
          threshold: return norm of weights, or whether norm is nonzero.
        Returns:
          GCs: list of self.num_factors_nK (p x p) matrices. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        '''
        W_nmf = self.get_W_nmf().detach().cpu().numpy()
        assert len(W_nmf.shape) == 2
        return [self.get_factor_GC(W_nmf[i,:].reshape(1,-1), threshold=threshold, ignore_features=ignore_features) for i in range(W_nmf.shape[0])]
    
    
    def evaluate(self, X_orig, y_orig, GC_orig, save_path, threshold=False, ignore_features=True):
        GC_est = self.GC(threshold=threshold, ignore_features=ignore_features)
        gc_mse = [(i,j,torch.nn.functional.mse_loss(torch.from_numpy(gc_est), torch.from_numpy(gc_orig))) for i,gc_est in enumerate(GC_est) for j,gc_orig in enumerate(GC_orig)]
        print("models.dcsfa_nmf_vanillaDirSpec.FullDCSFAModel.evaluate: gc_mse == ", gc_mse)
        plot_gc_est_comparissons_by_factor(GC_orig, GC_est, save_path+os.sep+"eval_gc_est_comparisson_visualization.png", include_lags=False)

        X_hat = self.reconstruct(X_orig)
        y_hat, _ = self.predict_proba(X_orig, return_scores=True)
        recon_mse = torch.nn.functional.mse_loss(torch.from_numpy(X_hat), torch.from_numpy(X_orig))
        score_mse = torch.nn.functional.mse_loss(torch.from_numpy(y_hat), torch.from_numpy(y_orig))
        print("models.dcsfa_nmf_vanillaDirSpec.FullDCSFAModel.evaluate: recon_mse == ", recon_mse)
        print("models.dcsfa_nmf_vanillaDirSpec.FullDCSFAModel.evaluate: score_mse == ", score_mse)
        avg_recon_mse = torch.mean(recon_mse)
        avg_score_mse = torch.mean(score_mse)
        print("models.dcsfa_nmf_vanillaDirSpec.FullDCSFAModel.evaluate: avg_recon_mse == ", avg_recon_mse)
        print("models.dcsfa_nmf_vanillaDirSpec.FullDCSFAModel.evaluate: avg_score_mse == ", avg_score_mse)

        plot_reconstruction_comparisson(X_orig.reshape((-1)), X_hat.reshape((-1)), save_path+os.sep+"eval_reconstruction_comparisson_visualization.png")

        with open(save_path+os.sep+"eval_summary.pkl", 'wb') as outfile:
            pkl.dump({
                "gc_mse": gc_mse, 
                "recon_mse": recon_mse, 
                "score_mse": score_mse, 
                "avg_recon_mse": avg_recon_mse, 
                "avg_score_mse": avg_score_mse, 
            }, outfile) 
        return recon_mse, avg_recon_mse, score_mse, avg_score_mse, gc_mse



if __name__ == "__main__":
    pass
