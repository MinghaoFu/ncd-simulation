"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from .components.beta import BetaVAE_MLP, BetaTVAE_MLP, BetaStateVAE_MLP
from .components.transition import NPChangeTransitionPrior, NPTransitionPrior, NPStatePrior
from .components.mlp import MLPEncoder, MLPDecoder, Inference
from .components.flow import ComponentWiseCondSpline
from .metrics.correlation import compute_mcc
import matplotlib.pyplot as plt
from io import BytesIO
import networkx as nx
from PIL import Image

from Caulimate.Utils.Tools import check_tensor, bin_mat
from Caulimate.Utils.Lego import PartiallyPeriodicMLP
import wandb
import ipdb as pdb

def plot_sparsity_matrix(alphas, title):
    mat_a = torch.tensor(alphas).numpy()
    mat_a_values = np.round(mat_a, decimals=2)
    fig, ax = plt.subplots()
    fig_alpha = ax.imshow(mat_a, cmap='Greens', vmin=0, vmax=1)
    for i in range(mat_a.shape[0]):
        for j in range(mat_a.shape[1]):
            ax.text(j, i, mat_a_values[i, j],
                    ha='center', va='center', color='black')
    cbar = fig.colorbar(fig_alpha)

    ax.set_xticks(np.arange(mat_a.shape[1]))
    ax.set_yticks(np.arange(mat_a.shape[0]))

    ax.set_xticklabels(np.arange(mat_a.shape[1])+1)
    ax.set_yticklabels(np.arange(mat_a.shape[0])+1)

    cbar.set_label('Alpha')

    ax.set_title(title)
    ax.set_xlabel('From')
    ax.set_ylabel('To')

    return fig

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
        trans_prior='NP',
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
        masks=None):
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
            self.masks = check_tensor(bin_mat(torch.inverse(check_tensor(torch.eye(self.input_dim)) - check_tensor(masks))))
        
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


        # base distribution for calculation of log prob under the model
        self.register_buffer('dyn_base_dist_mean', torch.zeros(self.dyn_dim))
        self.register_buffer('dyn_base_dist_var', torch.eye(self.dyn_dim))
        self.register_buffer('obs_base_dist_mean', torch.zeros(self.obs_dim))
        self.register_buffer('obs_base_dist_var', torch.eye(self.obs_dim))
        self.register_buffer('sta_base_dist_mean', torch.zeros(self.input_dim))
        self.register_buffer('sta_base_dist_var', torch.eye(self.input_dim))

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
        residuals, logabsdet = self.state_prior(zs, xs)
        log_ps = torch.sum(self.sta_base_dist.log_prob(residuals), dim=1) + logabsdet
        s_kld_dyn = torch.sum(torch.sum(log_qs, dim=-1), dim=-1) - log_ps
        s_kld_dyn = s_kld_dyn.mean()

        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
        
        s_recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon_s[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon_s[:,self.lag:], self.decoder_dist))/(length-self.lag)
        
        # sparsity in offdiagonal xs -> x
        M_inv_mean = torch.mean(torch.abs(torch.inverse(jac_mat)), dim=(0,1))
        est_B = M_inv_mean.fill_diagonal_(0) # M_inv_mean[~torch.eye(M_inv_mean.shape[0], dtype=bool)]
        spr_loss = torch.norm(est_B, p=1)  # * (1 - self.B_mask)
        #x_spy += self.diversity_loss(torch.mean(torch.abs(jac_mat), dim=(0,1)))
        #x_spy -= mean_jac[0,1] - mean_jac[1,0]
        
        # DAG constraint
        DAG_loss = self.DAG_loss(M_inv_mean)

        # VAE training
        loss = s_recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + \
            self.sigma * kld_obs + 1e-2 * s_kld_dyn + 1e-3 * spr_loss
        if self.current_epoch > 10:
            loss += 1e-6 * DAG_loss
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()

        zt_true = batch["zt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        return mcc, loss, recon_loss, s_recon_loss, past_kld_dyn, future_kld_dyn, kld_obs, \
            s_kld_dyn, spr_loss, DAG_loss, est_B


        # B = M_inv_mean.detach().cpu().numpy()
        
        # plt.savefig(buf, format='png')
        
        # self.logger.experiment.log({"train_B": wandb.Image(image_np)})

        # M_inv_mean_arr = wandb.Image(M_inv_mean.detach().cpu().numpy() // 10, caption='M Inv Mean')
        # wandb.log({"train_M_inv_mean": M_inv_mean_arr})

    def training_step(self, batch, batch_idx):
        mcc, loss, recon_loss, s_recon_loss, past_kld_dyn, future_kld_dyn, kld_obs, \
            s_kld_dyn, x_spr, DAG_loss, B = self.forward(batch)
        
        
        self.log("train_mcc", mcc)
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_s_recon_loss", s_recon_loss)
        self.log("train_kld_normal", past_kld_dyn)
        self.log("train_kld_dynamics", future_kld_dyn)
        self.log("train_kld_observation", kld_obs)
        self.log("train_kld_state", s_kld_dyn)
        self.log("train_spr_loss", x_spr)
        self.log("train_DAG_loss", DAG_loss)
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        mcc, loss, recon_loss, s_recon_loss, past_kld_dyn, future_kld_dyn, kld_obs, \
            s_kld_dyn, x_spr, DAG_loss, B = self.forward(batch)
        
        fig = plot_sparsity_matrix(B.detach().cpu().numpy(), 'Estimated B')
        est_B_path = './est_B.png'
        gt_B_path = './gt_B.png'
        fig.savefig(est_B_path, format='png')

        fig = plot_sparsity_matrix(self.B_mask.detach().cpu().numpy(), 'B mask')
        fig.savefig(gt_B_path, format='png')
        self.logger.experiment.log({
            "B_mask": wandb.Image(np.array(Image.open(est_B_path))), 
             "train_B": wandb.Image(np.array(Image.open(gt_B_path))),
                                             })

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
    
    def sample(self, n=64):
        with torch.no_grad():
            e = torch.randn(n, self.z_dim, device=self.device)
            eps, _ = self.spline.inverse(e)
        return eps

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []
