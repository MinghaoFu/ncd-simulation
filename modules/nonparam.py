"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from .components.beta import BetaVAE_MLP, BetaTVAE_MLP, BetaStateVAE_MLP
from .components.transition import NPChangeTransitionPrior, NPTransitionPrior, NPStatePrior, NPMaskChangeTransitionxrior, NPInstantaneousTransitionPrior
from .components.mlp import MLPEncoder, MLPDecoder, Inference
from .components.flow import ComponentWiseCondSpline
from .metrics.correlation import compute_mcc
import matplotlib.pyplot as plt
from io import BytesIO
import networkx as nx
import os
from PIL import Image

from Caulimate.Utils.Tools import check_tensor, check_array, bin_mat
from Caulimate.Utils.Lego import PartiallyPeriodicMLP
from Caulimate.Utils.Visualization import plot_sparsity_matrix
from Caulimate.Utils.Metrics.block import scipy_kl_estimator
import wandb
import ipdb as pdb

def normalize(data):
    array = check_array(data)
    # Normalize each feature (column) independently
    min_val = np.min(array, axis=0, keepdims=True)
    max_val = np.max(array, axis=0, keepdims=True)
    
    # Avoid division by zero in case of constant features
    normalized_array = (array - min_val) / (max_val - min_val + 1e-8)
    return normalized_array

def sample_mask(logits, tau=0.3):
    num_vars = len(logits)
    mask = gumbel_sigmoid(logits, tau=tau)
    non_diagonal_mask = torch.ones(num_vars, num_vars) - torch.eye(num_vars)
    # Set diagonal entries to 0
    mask = mask * non_diagonal_mask
    return mask

def sample_logistic(shape, out=None):
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return torch.log(U) - torch.log(1-U)

def gumbel_sigmoid(logits, tau=1):
    dims = logits.dim()
    logistic_noise = sample_logistic(logits.size(), out=logits.data.new())
    y = logits + logistic_noise
    return torch.sigmoid(y / tau)

class ModularShifts(pl.LightningModule):

    def __init__(
        self, 
        input_dim,
        length,
        obs_dim,
        dyn_dim, 
        lag,
        nclass,
        hidden_dim=128,
        dyn_embedding_dim=2,
        obs_embedding_dim=2,
        trans_prior='INS', # NP, INS
        lr=1e-4,
        infer_mode='F',
        bound=5,
        count_bins=8,
        order='linear',
        beta=0.0025,
        gamma=0.0075,
        sigma=0.0025,
        decoder_dist='gaussian',
        correlation='Pearson',
        masks=None,
        B=None,
        instantaneous=False):
        '''Nonlinear ICA for general causal processes with modualar distribution shifts'''
        super().__init__()
        # Transition prior must be L (Linear), NP (Nonparametric)
        self.save_hyperparameters() # save hyperparameters to checkpoint
        
        assert trans_prior in ('L', 'NP')
        self.obs_dim = obs_dim
        self.dyn_dim = dyn_dim
        self.obs_embedding_dim = obs_embedding_dim
        self.dyn_embedding_dim = dyn_embedding_dim
        self.z_dim = obs_dim + dyn_dim
        self.lag = lag
        self.input_dim = input_dim
        self.lr = lr
        self.lag = lag
        self.length = length
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.infer_mode = infer_mode
        # Domain embeddings (dynamics)
        self.dyn_embed_func = nn.Embedding(nclass, dyn_embedding_dim)
        self.obs_embed_func = nn.Embedding(nclass, obs_embedding_dim)
        # Flow for nonstationary regimes
        self.flow = ComponentWiseCondSpline(input_dim=self.obs_dim,
                                            context_dim=obs_embedding_dim,
                                            bound=bound,
                                            count_bins=count_bins,
                                            order=order)
        # Factorized inference
        self.t_embed_dim = 2
        # self.net = BetaTVAE_MLP(input_dim=input_dim, 
        #                         z_dim=self.z_dim, 
        #                         t_embed_dim=self.t_embed_dim, 
        #                         hidden_dim=hidden_dim)
        
        # self.xs_net = BetaTVAE_MLP(input_dim=input_dim, 
        #                         z_dim=input_dim, 
        #                         t_embed_dim=self.t_embed_dim,
        #                         hidden_dim=hidden_dim)

        self.net = BetaVAE_MLP(input_dim=input_dim, 
                                z_dim=self.z_dim, 
                                hidden_dim=hidden_dim)
        
        if masks is not None:
            self.B_mask = check_tensor(masks)
            self.masks = check_tensor(bin_mat(torch.inverse(check_tensor(torch.eye(self.input_dim)) - check_tensor(self.B_mask))))
        else:
            self.B_mask = torch.ones(self.input_dim, self.input_dim) - torch.eye(self.input_dim)
            self.masks = None #check_tensor(torch.ones(self.input_dim, self.input_dim))

        if B is not None:
            self.B = check_tensor(B)
            
        self.xs_net = BetaStateVAE_MLP(input_dim=input_dim, 
                                z_dim=input_dim, 
                                hidden_dim=hidden_dim,
                                masks=self.masks)

        # Initialize transition prior
        if trans_prior == 'L':
            pass
            # self.transition_prior = MBDTransitionPrior(lags=lag, 
            #                                            latent_size=self.dyn_dim, 
            #                                            bias=False)
        elif trans_prior == 'NP':
            self.transition_prior = NPChangeTransitionPrior(lags=lag, 
                                                            latent_size=self.dyn_dim,
                                                            embedding_dim=dyn_embedding_dim,
                                                            num_layers=4, 
                                                            hidden_dim=hidden_dim)
        elif trans_prior == 'INS':  
            self.transition_prior = NPMaskChangeTransitionPrior(lags=lag, 
                                                      latent_size=self.dyn_dim,
                                                      embedding_dim=dyn_embedding_dim,
                                                      num_layers=4, 
                                                      hidden_dim=hidden_dim)
            
        # self.np_gen = NPChangeTransitionPrior(lags=1,
        #                                         latent_size=self.obs_dim + self.dyn_dim, # change to observed size
        #                                         embedding_dim=dyn_embedding_dim,
        #                                         num_layers=4,
        #                                         hidden_dim=hidden_dim)
        self.t_embedding = PartiallyPeriodicMLP(1, 4, self.t_embed_dim, 500)
        self.state_prior = NPStatePrior(lags=1,
                                             latent_size=self.z_dim,
                                             input_dim=input_dim,
                                             num_layers=4,
                                             hidden_dim=hidden_dim)
        self.zt_true_accum = []
        self.zt_recon_accum = []

        # instantaneous
        self.transition_prior_fix = NPInstantaneousTransitionPrior(lags=lag, 
                                                      latent_size=self.z_dim, 
                                                      num_layers=3, 
                                                      hidden_dim=hidden_dim)

        # base distribution for calculation of log prob under the model
        self.register_buffer('dyn_base_dist_mean', torch.zeros(self.dyn_dim))
        self.register_buffer('dyn_base_dist_var', torch.eye(self.dyn_dim))
        self.register_buffer('obs_base_dist_mean', torch.zeros(self.obs_dim))
        self.register_buffer('obs_base_dist_var', torch.eye(self.obs_dim))
        self.register_buffer('sta_base_dist_mean', torch.zeros(self.input_dim))
        self.register_buffer('sta_base_dist_var', torch.eye(self.input_dim))
        self.register_buffer('ind_base_dist_mean', torch.zeros(self.input_dim + self.obs_dim + self.dyn_dim))
        self.register_buffer('ind_base_dist_var', torch.eye(self.input_dim + self.obs_dim + self.dyn_dim))

    @property
    def dyn_base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.dyn_base_dist_mean, self.dyn_base_dist_var)

    @property
    def obs_base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.obs_base_dist_mean, self.obs_base_dist_var)
    
    @property
    def sta_base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.sta_base_dist_mean, self.sta_base_dist_var)
    
    @property
    def ind_base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.ind_base_dist_mean, self.ind_base_dist_var)
    
    def preprocess(self, B):
        return B - B.diagonal().diag()
    
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def reconstruction_loss(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(
                x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'gaussian':
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'sigmoid_gaussian':
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        return recon_loss

    def DAG_loss(self, B):
        if len(B.shape) == 2:
            B = B.unsqueeze(0)  
        matrix_exp = torch.exp(B * B)
        traces = torch.sum(torch.diagonal(matrix_exp, dim1=-2, dim2=-1)) - B.shape[0] * B.shape[1]
        return traces
    
    def forward(self, batch):
        x, z, c, h = batch['xt'], batch['zt'], batch['ct'], batch['ht']
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape

        t_embedding = self.t_embedding(h / 100000).unsqueeze(1).repeat(1, length, 1)    
        x_flat = x.view(-1, self.input_dim)
        
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size,1,self.obs_embedding_dim).repeat(1,length,1)
        # Inference
        x_recon, mus, logvars, zs = self.net(x_flat)

        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)

        #x_recon[:, :, 1] = x_recon[:, :, 1] + x_recon[:, :, 0] * self.t_embedding(h)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)
        ### Dynamics parts ###
        # Past KLD <=> N(0,1) #
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag,:self.dyn_dim]), torch.ones_like(logvars[:,:self.lag, :self.dyn_dim]))
        log_pz_past = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag,:self.dyn_dim]),dim=-1),dim=-1)
        log_qz_past = torch.sum(torch.sum(log_qz[:,:self.lag,:self.dyn_dim],dim=-1),dim=-1)
        past_kld_dyn = log_qz_past - log_pz_past
        past_kld_dyn = past_kld_dyn.mean()
        # Future KLD #
        log_qz_future = log_qz[:,self.lag:]
        residuals, logabsdet = self.transition_prior(zs[:,:,:self.dyn_dim], dyn_embeddings)
        log_pz_future = torch.sum(self.dyn_base_dist.log_prob(residuals), dim=1) + logabsdet
        future_kld_dyn = (torch.sum(torch.sum(log_qz_future,dim=-1),dim=-1) - log_pz_future) / (length-self.lag)
        future_kld_dyn = future_kld_dyn.mean()

        ### Observation parts ###
        p_dist_obs = D.Normal(obs_embeddings[:,:,0].reshape(batch_size, length, 1), 
                              torch.exp(obs_embeddings[:,:,1].reshape(batch_size, length, 1) / 2) )
        log_pz_obs = torch.sum(torch.sum(p_dist_obs.log_prob(zs[:,:,self.dyn_dim:]), dim=1),dim=-1)
        log_qz_obs = torch.sum(torch.sum(log_qz[:,:self.lag,self.dyn_dim:],dim=-1),dim=-1)
        kld_obs = log_qz_obs - log_pz_obs
        kld_obs = kld_obs.mean()      

        # masks for mixing matrix
        # B_mask = torch.zeros(self.input_dim, self.input_dim)
        # B_mask[:2, :1] = 1; B_mask.diagonal(dim1=-2, dim2=-1).fill_(1) # must be sparse => invertible
        # masks = check_tensor(bin_mat(torch.inverse(B_mask)))
        # state transition
        jac_mat, x_recon_s, s_mus, s_logvars, xs = self.xs_net(x_flat, standardize=False)    

        ## if without mixing
        # std = s_logvars.div(2).exp()
        # eps = Variable(std.data.new(std.size()).normal_())
        # x_recon_s = s_mus + std*eps
        jac_mat = jac_mat.view(batch_size, length, self.input_dim, self.input_dim)  
        x_recon_s = x_recon_s.view(batch_size, length, self.input_dim)
        s_mus = s_mus.reshape(batch_size, length, self.input_dim)
        s_logvars  = s_logvars.reshape(batch_size, length, self.input_dim)
        xs = xs.reshape(batch_size, length, self.input_dim)
        
        # state prior & KL with posterior
        s_q_dist = D.Normal(s_mus, torch.exp(s_logvars / 2))
        log_qs = s_q_dist.log_prob(xs)
        s_residuals, s_logabsdet = self.state_prior(zs, xs)
        log_ps = torch.sum(self.sta_base_dist.log_prob(s_residuals), dim=1) + s_logabsdet
        s_kld_dyn = torch.sum(torch.sum(log_qs, dim=-1), dim=-1) - log_ps
        s_kld_dyn = s_kld_dyn.mean()

        # independence between e_x and e_z
        sz_residuals = torch.cat([residuals, s_residuals[:, -1:, :]], dim=-1)
        ind_loss = - torch.sum(self.ind_base_dist.log_prob(sz_residuals)) / (batch_size * (length - self.lag))

        # psz_dist = D.Normal(torch.zeros_like(sz_residuals), torch.ones_like(sz_residuals))
        # log_psz = torch.sum(torch.sum(self.log_prob(sz_residuals),dim=-1),dim=-1)
        # log_qsz = torch.sum(torch.sum(log_qsz,dim=-1),dim=-1)
        # ind_kld_dyn = log_qsz - log_psz
        # ind_kld_dyn = ind_kld_dyn.mean()

        # p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag,:self.dyn_dim]), torch.ones_like(logvars[:,:self.lag, :self.dyn_dim]))
        # log_pz_past = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag,:self.dyn_dim]),dim=-1),dim=-1)
        # log_qz_past = torch.sum(torch.sum(log_qz[:,:self.lag,:self.dyn_dim],dim=-1),dim=-1)
        # past_kld_dyn = log_qz_past - log_pz_past
        # past_kld_dyn = past_kld_dyn.mean()
        
        # MSCL sparsity
        # ins_spa = torch.norm(self.transition_prior_fix.u, p=1)

        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
        
        s_recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon_s[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon_s[:,self.lag:], self.decoder_dist))/(length-self.lag)
        
        # sparsity in offdiagonal xs -> x
        M_inv_mean = torch.mean(torch.abs(torch.inverse(jac_mat)), dim=(0,1))
        est_B = M_inv_mean.fill_diagonal_(0) # M_inv_mean[~torch.eye(M_inv_mean.shape[0], dtype=bool)]
        spr_loss = torch.norm(est_B, p=1)  # * (1 - self.B_mask)
        
        # DAG constraint
        DAG_loss = self.DAG_loss(M_inv_mean)

        # VAE training
        loss = s_recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + \
            self.sigma * kld_obs + 1e-2 * s_kld_dyn + 1e-3 * spr_loss + 1e-3 * ind_loss
        if self.current_epoch > 10:
            loss += 1e-6 * DAG_loss
        # Compute Mean Correlation Coefficient (MCC)
        # zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()
        # zt_true = batch["zt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        # mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        zt_recon = mus[:, -1:].view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = batch["zt"][:, -1:].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        zt_recon = normalize(zt_recon.T) # (batch_size, z_dim)  
        zt_true = normalize(zt_true.T)

        kl_step = scipy_kl_estimator(zt_recon, zt_true, 1)
        self.log('kl_step', kl_step)

        # Compute MCC of state variables
        st_recon = s_mus.view(-1, self.input_dim).T.detach().cpu().numpy()
        st_true = batch["st"].view(-1, self.input_dim).T.detach().cpu().numpy()
        s_mcc = compute_mcc(st_recon, st_true, self.correlation)

        return mcc, s_mcc, loss, recon_loss, s_recon_loss, past_kld_dyn, future_kld_dyn, kld_obs, \
            s_kld_dyn, spr_loss, DAG_loss, ind_loss, est_B, zt_true, zt_recon

    def training_step(self, batch, batch_idx):
        mcc, s_mcc, loss, recon_loss, s_recon_loss, past_kld_dyn, future_kld_dyn, kld_obs, \
            s_kld_dyn, x_spr, DAG_loss, ind_loss, B, zt_true, zt_recon = self.forward(batch)
        
        
        self.log("train_mcc", mcc)
        self.log("train_s_mcc", s_mcc)
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_s_recon_loss", s_recon_loss)
        self.log("train_kld_normal", past_kld_dyn)
        self.log("train_kld_dynamics", future_kld_dyn)
        self.log("train_kld_observation", kld_obs)
        self.log("train_kld_state", s_kld_dyn)
        self.log("train_spr_loss", x_spr)
        self.log("train_DAG_loss", DAG_loss)
        self.log("train_ind_loss", ind_loss)
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        mcc, s_mcc, loss, recon_loss, s_recon_loss, past_kld_dyn, future_kld_dyn, kld_obs, \
            s_kld_dyn, x_spr, DAG_loss, ind_loss, B, zt_true, zt_recon = self.forward(batch)
        
        
        self.zt_recon_accum.append(zt_recon)
        self.zt_true_accum.append(zt_true)
        
        if os.path.exists(self.logger.save_dir):
            est_B_path = os.path.join(self.logger.save_dir, 'est_B.png')
            mask_B_path = os.path.join(self.logger.save_dir, 'mask_B.png')
            gt_B_path = os.path.join(self.logger.save_dir, 'gt_B.png')

            fig = plot_sparsity_matrix(B.detach().cpu().numpy(), 'Estimated B')
            fig.savefig(est_B_path, format='png')

            fig = plot_sparsity_matrix(self.B_mask.detach().cpu().numpy(), 'B mask')
            fig.savefig(mask_B_path, format='png')

            fig = plot_sparsity_matrix(self.B.detach().cpu().numpy(), 'True B')
            fig.savefig(gt_B_path, format='png')

        # self.logger.experiment.log({
        #     "mask_B": wandb.Image(np.array(Image.open(mask_B_path))), 
        #     "est_B": wandb.Image(np.array(Image.open(est_B_path))),
        #     "gt_B": wandb.Image(np.array(Image.open(gt_B_path))),    
        # })
        
        self.log("val_mcc", mcc) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_normal", past_kld_dyn)
        self.log("val_kld_dynamics", future_kld_dyn)
        self.log("val_kld_observation", kld_obs)
        self.log("val_kld_observation", s_kld_dyn)
        # self.log("val_spr_loss", x_spr)
        # self.log("val_DAG_loss", DAG_loss)
        
        return loss

    def on_train_epoch_end(self):
        zt_true_all = np.concatenate(self.zt_true_accum, axis=1)
        zt_recon_all = np.concatenate(self.zt_recon_accum, axis=1)
        
        kl_epoch = scipy_kl_estimator(zt_recon_all, zt_true_all, 1)
        self.log('val_kl_epoch', kl_epoch)
        
        # Reset accumulators for the next epoch
        self.zt_true_accum = []
        self.zt_recon_accum = []
    
    def sample(self, n=64):
        with torch.no_grad():
            e = torch.randn(n, self.z_dim, device=self.device)
            eps, _ = self.spline.inverse(e)
        return eps

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []

    def loss_function(self, x, x_recon, mus, logvars, zs, embeddings):
        '''
        VAE ELBO loss: recon_loss + kld_loss (past: N(0,1), future: N(0,1) after flow) + sparsity_loss
        '''
        batch_size, length, _ = x.shape

        # Sparsity loss
        sparsity_loss = 0
        # fix
        # if self.z_dim_fix>0:
        #     unmasked_alphas_fix = self.alphas_fix()
        #     mask_fix = (unmasked_alphas_fix > 0.1).float()
        #     alphas_fix = unmasked_alphas_fix * mask_fix

        # Recon loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag]), torch.ones_like(logvars[:,:self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag]),dim=-1),dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:,:self.lag],dim=-1),dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()

        # Future KLD
        kld_future = []
        # fix
        if self.z_dim_fix>0:
            log_qz_laplace = log_qz[:,self.lag:,:self.z_dim_fix]
            residuals, logabsdet, hist_jac = self.transition_prior_fix.forward(zs[:,:,:self.z_dim_fix])
            log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet
            kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
            kld_future.append(kld_laplace)
            # sparsity on history
            # sparsity_loss = F.l1_loss(hist_jac, torch.zeros_like(hist_jac))

            # import pdb
            # pdb.set_trace()

            trans_show = []
            inst_show = []
            numm = 0
            p_list = [2, 7, 11]
            p2_list = [0.0, 1.0, 4.0]
            cnt = 0

            
            for jac in hist_jac:
                sparsity_loss = sparsity_loss + self.w_inst[cnt] * F.l1_loss(jac[:,0,self.lag*self.z_dim_fix:], torch.zeros_like(jac[:,0,self.lag*self.z_dim_fix:]), reduction='sum')
                sparsity_loss = sparsity_loss + self.w_hist[cnt] * F.l1_loss(jac[:,0,:self.lag*self.z_dim_fix], torch.zeros_like(jac[:,0,:self.lag*self.z_dim_fix]), reduction='sum')
                cnt += 1
                numm = numm + jac.numel()
                trans_show.append(jac[:,0,:self.lag*self.z_dim_fix].detach().cpu())
                inst_cur = jac[:,0,self.lag*self.z_dim_fix:].detach().cpu()
                inst_cur = torch.nn.functional.pad(inst_cur, (0,self.z_dim_fix-inst_cur.shape[1],0,0), mode='constant', value=0)
                inst_show.append(inst_cur)
            trans_show = torch.stack(trans_show, dim=1).abs().mean(dim=0)
            inst_show = torch.stack(inst_show, dim=1).abs().mean(dim=0)
            sparsity_loss = sparsity_loss / numm


            self.trans_show = trans_show
            self.inst_show = inst_show
        # change
        if self.z_dim_change>0:
            assert(0)
            # log_qz_laplace = log_qz[:,self.lag:,self.z_dim_fix:]
            # residuals, logabsdet = self.transition_prior_change.forward(zs[:,:,self.z_dim_fix:], embeddings, alphas_change)
            # log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet
            # kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
            # kld_future.append(kld_laplace)
        kld_future = torch.cat(kld_future, dim=-1)
        kld_future = kld_future.mean()

        return sparsity_loss ,recon_loss, kld_normal, kld_future
