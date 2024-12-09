"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
import torch.optim as optim
from torch.nn import functional as F
from .components.beta import BetaVAE_MLP
from .components.transition import (MBDTransitionPrior, 
                                    NPChangeTransitionPrior)
from .components.mlp import MLPEncoder, MLPDecoder, Inference, NLayerLeakyMLP
from .components.flow import ComponentWiseCondSpline
from .metrics.correlation import compute_mcc
from ..minghao_utils import check_tensor, check_array, threshold_till_dag, count_accuracy, bin_mat, postprocess, top_k_abs_tensor

import ipdb as pdb

class linear_causal_state(pl.LightningModule):
    def __init__(
        self,
        output_dim, 
        u_dim,
        activation=None,
    ):
        super().__init__()
        self.net = NLayerLeakyMLP(u_dim, output_dim**2, num_layers=4)
        self.output_dim = output_dim    
        self.activation = activation
        self.theta_B = nn.Parameter(torch.randn(output_dim, output_dim))
        self.start_epoch = 0
        self.Bs = None
        
    def pretrain_sparse(self, train_loader, pretrain_epochs):
        optimizer = optim.SGD(self.net.parameters(), lr=0.01)  # Choose an appropriate optimizer and learning rate

        for epoch in range(pretrain_epochs):
            for batch_data in train_loader:
                optimizer.zero_grad()
                output = self.net(batch_data['ht'])  # Forward pass through the network
                loss = torch.mean(output ** 2)  # Define the loss as the mean squared output
                
                loss.backward()  # Backward pass to compute gradients
                optimizer.step()  # Update the network's parameters based on the gradients
        print('Pretrained sparse model loss: {}'.format(loss))
        
    def disentangle(self, x, u, Bs=None):
        '''
            x: [batch_size, output_dim]
            u: [batch_size, u_dim]
        '''
        flat_batch_size = x.shape[0]
        batch_size = u.shape[0]
        length = flat_batch_size // batch_size
        if Bs is not None:
            mat_Bs = Bs
        else:
            # get causal hidden state
            flat_Bs = self.net(u)
            if self.activation is not None:
                flat_Bs = self.activation(flat_Bs)
            mat_Bs = flat_Bs.reshape(-1, self.output_dim, self.output_dim)
            
            mat_Bs = self.theta_B.repeat(batch_size, 1, 1)# + mat_Bs
            for i in range(self.output_dim):
                mat_Bs[:, i, i] = 0
        # store I and B for reconstruct
        self.batch_Bs = mat_Bs #mat_Bs.repeat(length, 1, 1) 
        # reshape to lags+length
        if self.current_epoch < self.start_epoch and self.training:
            self.Bs = check_tensor(torch.zeros((flat_batch_size, self.output_dim, self.output_dim)), astype=x) 
        else:
            self.Bs = mat_Bs.repeat(length, 1, 1) #check_tensor(torch.zeros((flat_batch_size, self.output_dim, self.output_dim)), astype=x) 
        self.Is = check_tensor(torch.eye(self.output_dim).unsqueeze(0).repeat(flat_batch_size, 1, 1), astype=x)
        
        x = torch.bmm((self.Is - self.Bs), x.unsqueeze(-1)).squeeze(-1)
        
        return x
        
    def reconstruct(self, x):
        # recover causal hidden state to original
        x_recon = torch.bmm(torch.inverse(self.Is - self.Bs), x.unsqueeze(-1)).squeeze(-1)
        return x_recon
    
class ModularShiftsVaryB(pl.LightningModule):
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
        B_sparsity=0.0025,
        decoder_dist='gaussian',
        correlation='Pearson'):
        '''Nonlinear ICA for general causal processes with modualar distribution shifts'''
        super().__init__()
        # Transition prior must be L (Linear), NP (Nonparametric)
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
        self.B_sparsity = B_sparsity
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
        self.net = BetaVAE_MLP(input_dim=input_dim, 
                                z_dim=self.z_dim, 
                                hidden_dim=hidden_dim)

        # Initialize transition prior
        if trans_prior == 'L':
            self.transition_prior = MBDTransitionPrior(lags=lag, 
                                                       latent_size=self.dyn_dim, 
                                                       bias=False)
        elif trans_prior == 'NP':
            self.transition_prior = NPChangeTransitionPrior(lags=lag, 
                                                            latent_size=self.dyn_dim,
                                                            embedding_dim=dyn_embedding_dim,
                                                            num_layers=4, 
                                                            hidden_dim=hidden_dim)
            
        # Initialize causal adjacency matrix in observed variables
        u_dim = 1
        self.causal_state = linear_causal_state(output_dim=self.input_dim, 
                                                       u_dim=u_dim,
                                                       activation=None)

        # base distribution for calculation of log prob under the model
        self.register_buffer('dyn_base_dist_mean', torch.zeros(self.dyn_dim))
        self.register_buffer('dyn_base_dist_var', torch.eye(self.dyn_dim))
        self.register_buffer('obs_base_dist_mean', torch.zeros(self.obs_dim))
        self.register_buffer('obs_base_dist_var', torch.eye(self.obs_dim))
        
    @property
    def dyn_base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.dyn_base_dist_mean, self.dyn_base_dist_var)

    @property
    def obs_base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.obs_base_dist_mean, self.obs_base_dist_var)
    
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

    def forward(self, batch):
        x, y, c, h, b = batch['xt'], batch['yt'], batch['ct'], batch['ht'], batch['bt']
        batch_size, length, _ = x.shape
        u = h#torch.cat([c, h], dim=-1)
        
        x_flat = x.view(-1, self.input_dim)
        # x to (I-B)x
        x_flat = self.causal_state.disentangle(x_flat, u)
        _, mus, logvars, zs = self.net(x_flat)
        return zs, mus, logvars       

    def training_step(self, batch, batch_idx):
        x, y, c, h, b = batch['xt'], batch['yt'], batch['ct'], batch['ht'], batch['bt']
        u = h#torch.cat([c, h], dim=-1)
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        # x to (I-B)x
        x_flat = self.causal_state.disentangle(x_flat, u)
        #x = x_flat.reshape(-1, 3, self.input_dim)
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size, 1, self.obs_embedding_dim).repeat(1,length,1)
        
        # Inference
        x_recon, mus, logvars, zs = self.net(x_flat)
        # (I-B)x to x
        x_recon = self.causal_state.reconstruct(x_recon)
        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
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

        # sparsity
        B_sparsity = torch.norm(self.causal_state.Bs, p=1) / x.shape[0]
        
        # B reconstruction
        B_recon = self.reconstruction_loss(self.causal_state.batch_Bs, b, 'gaussian')
        # VAE training
        loss = recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs + self.B_sparsity * B_sparsity# + B_recon
        #########################   training step  #########################
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", past_kld_dyn)
        self.log("train_kld_dynamics", future_kld_dyn)
        self.log("train_kld_observation", kld_obs)
        self.log("train_B_sparsity", B_sparsity)
        self.log("train_B_recon", B_recon)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, c, h, b = batch['xt'], batch['yt'], batch['ct'], batch['ht'], batch['bt']
        u = h#torch.cat([c, h], dim=-1)
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        # x to (I-B)x
        x_flat = self.causal_state.disentangle(x_flat, u)
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size, 1, self.obs_embedding_dim).repeat(1,length,1)
        
        # Inference
        x_recon, mus, logvars, zs = self.net(x_flat)
        # (I-B)x to x
        x_recon = self.causal_state.reconstruct(x_recon)
        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
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

        # sparsity
        B_sparsity = torch.norm(self.causal_state.Bs, p=1) / x.shape[0]
        
        # VAE training
        loss = recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs + self.B_sparsity * B_sparsity
        #########################   validation step  #########################
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()

        zt_true = batch["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)
        # count accuracy of B predictions
        batch_B_tpr = []
        batch_B_fdr = []
        for i in range(batch_size):
            graph_thres = 0.1
            #B_est = postprocess(check_array(self.causal_state.batch_Bs[i]), graph_thres=graph_thres)
            n = torch.nonzero(b[i]).size(0)
            B_est = check_array(top_k_abs_tensor(self.causal_state.batch_Bs[i], n))
            B_gt = check_array(b[i])
            B_gt[B_gt < graph_thres] = 0
            acc = count_accuracy(bin_mat(B_est), bin_mat(B_gt))
            batch_B_tpr.append(acc['tpr'])
            batch_B_fdr.append(acc['fdr'])
        B_avg_tpr = np.mean(batch_B_tpr)
        B_avg_fdr = np.mean(batch_B_fdr)
        
        self.log("val_B_avg_tpr", B_avg_tpr)
        self.log("val_B_avg_fdr", B_avg_fdr)
        
        self.log("val_mcc", mcc) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_normal", past_kld_dyn)
        self.log("val_kld_dynamics", future_kld_dyn)
        self.log("val_kld_observation", kld_obs)
        self.log("val_B_sparsity", B_sparsity)
        
        B_recon = self.reconstruction_loss(self.causal_state.batch_Bs, b, 'gaussian')
        self.log("val_B_recon", B_recon)

        return loss
    
    def sample(self, n=64):
        with torch.no_grad():
            e = torch.randn(n, self.z_dim, device=self.device)
            eps, _ = self.spline.inverse(e)
        return eps

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []
    
class ModularShiftsFixedB(pl.LightningModule):
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
        B_sparsity=0.0025,
        decoder_dist='gaussian',
        correlation='Pearson'):
        '''Nonlinear ICA for general causal processes with modualar distribution shifts'''
        super().__init__()
        # Transition prior must be L (Linear), NP (Nonparametric)
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
        self.B_sparsity = B_sparsity
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
        self.net = BetaVAE_MLP(input_dim=input_dim, 
                                z_dim=self.z_dim, 
                                hidden_dim=hidden_dim)

        # Initialize transition prior
        if trans_prior == 'L':
            self.transition_prior = MBDTransitionPrior(lags=lag, 
                                                       latent_size=self.dyn_dim, 
                                                       bias=False)
        elif trans_prior == 'NP':
            self.transition_prior = NPChangeTransitionPrior(lags=lag, 
                                                            latent_size=self.dyn_dim,
                                                            embedding_dim=dyn_embedding_dim,
                                                            num_layers=4, 
                                                            hidden_dim=hidden_dim)
            
        # Initialize causal adjacency matrix in observed variables
        self.B = nn.Parameter(check_tensor(torch.randn(self.input_dim, self.input_dim)))
        self.I = check_tensor(torch.eye(self.input_dim), astype=self.B)

        # base distribution for calculation of log prob under the model
        self.register_buffer('dyn_base_dist_mean', torch.zeros(self.dyn_dim))
        self.register_buffer('dyn_base_dist_var', torch.eye(self.dyn_dim))
        self.register_buffer('obs_base_dist_mean', torch.zeros(self.obs_dim))
        self.register_buffer('obs_base_dist_var', torch.eye(self.obs_dim))
        
    @property
    def dyn_base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.dyn_base_dist_mean, self.dyn_base_dist_var)

    @property
    def obs_base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.obs_base_dist_mean, self.obs_base_dist_var)
    
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

    def forward(self, batch):
        x, y, c = batch['xt'], batch['yt'], batch['ct']
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        # x to (I-B)x
        x_flat = self.causal_state.disentangle(x_flat, u)
        _, mus, logvars, zs = self.net(x_flat)
        return zs, mus, logvars       

    def training_step(self, batch, batch_idx):
        x, y, c = batch['xt'], batch['yt'], batch['ct']
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        # x to (I-B)x
        x_flat = torch.matmul(x_flat, (self.I - self.B))
        #x = x_flat.reshape(-1, 3, self.input_dim)
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size, 1, self.obs_embedding_dim).repeat(1,length,1)
        
        # Inference
        x_recon, mus, logvars, zs = self.net(x_flat)
        # (I-B)x to x
        x_recon = torch.matmul(x_recon, torch.inverse(self.I - self.B))
        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
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

        # sparsity
        B_sparsity = torch.norm(self.B, p=1)
        
        # VAE training
        loss = recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs + self.B_sparsity * B_sparsity# + B_recon
        #########################   training step  #########################
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", past_kld_dyn)
        self.log("train_kld_dynamics", future_kld_dyn)
        self.log("train_kld_observation", kld_obs)
        self.log("train_B_sparsity", B_sparsity)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, c = batch['xt'], batch['yt'], batch['ct']
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        # x to (I-B)x
        x_flat = torch.matmul(x_flat, (self.I - self.B))
        #x = x_flat.reshape(-1, 3, self.input_dim)
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size, 1, self.obs_embedding_dim).repeat(1,length,1)
        
        # Inference
        x_recon, mus, logvars, zs = self.net(x_flat)
        # (I-B)x to x
        x_recon = torch.matmul(x_recon, torch.inverse(self.I - self.B))
        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
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

        # sparsity
        B_sparsity = torch.norm(self.B, p=1)
        
        # VAE training
        loss = recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs + self.B_sparsity * B_sparsity
        #########################   validation step  #########################
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()

        zt_true = batch["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)
        # count accuracy of B predictions
        # batch_B_tpr = []
        # batch_B_fdr = []
        # for i in range(batch_size):
        #     graph_thres = 0.1
        #     #B_est = postprocess(check_array(self.causal_state.batch_Bs[i]), graph_thres=graph_thres)
        #     n = torch.nonzero(b[i]).size(0)
        #     B_est = check_array(top_k_abs_tensor(self.causal_state.batch_Bs[i], n))
        #     B_gt = check_array(b[i])
        #     B_gt[B_gt < graph_thres] = 0
        #     acc = count_accuracy(bin_mat(B_est), bin_mat(B_gt))
        #     batch_B_tpr.append(acc['tpr'])
        #     batch_B_fdr.append(acc['fdr'])
        # B_avg_tpr = np.mean(batch_B_tpr)
        # B_avg_fdr = np.mean(batch_B_fdr)
        
        # self.log("val_B_avg_tpr", B_avg_tpr)
        # self.log("val_B_avg_fdr", B_avg_fdr)
        
        self.log("val_mcc", mcc) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_normal", past_kld_dyn)
        self.log("val_kld_dynamics", future_kld_dyn)
        self.log("val_kld_observation", kld_obs)
        self.log("val_B_sparsity", B_sparsity)

        return loss
    
    def sample(self, n=64):
        with torch.no_grad():
            e = torch.randn(n, self.z_dim, device=self.device)
            eps, _ = self.spline.inverse(e)
        return eps

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []