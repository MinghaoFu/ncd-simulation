"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from .components.beta import BetaVAE_CNN, BetaVAE_Physics
from .components.transition import (MBDTransitionPrior, 
                                    NPChangeTransitionPrior,
                                    NPTransitionPrior)
from .components.mlp import MLPEncoder, MLPDecoder, Inference
from .components.flow import ComponentWiseCondSpline
from .metrics.correlation import compute_mcc
from .components.SAVI import build_SAVI
from torch.autograd import Variable
import ipdb as pdb

class Slot2Z_dist(nn.Module):
    def __init__(self, hidden_dim,num_slots=7,  # at most 6 objects per scene
        slot_size=128,):
        super().__init__()
        slot_dim = num_slots * slot_size
        self.net = nn.Sequential(
            nn.Linear(slot_dim, slot_dim//2),
            nn.GELU(),
            nn.Linear( slot_dim//2, hidden_dim*2),
            nn.GELU())
    
    def forward(self, slots):
        slots = slots.reshape(slots.shape[0],slots.shape[1],-1)
        return self.net(slots)
    
    def reparametrize(self,mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps

class Z2Slot(nn.Module):
    def __init__(self,hidden_dim,num_slots=7,  # at most 6 objects per scene
        slot_size=128,):
        super().__init__()
        slot_dim = num_slots * slot_size
        self.net = nn.Sequential(
            nn.Linear( hidden_dim, slot_dim//2),
            nn.GELU(),
            nn.Linear(slot_dim//2,slot_dim),
            nn.GELU(),
            )
    
    def forward(self,z):
        return self.net(z)

class ModularShifts(pl.LightningModule):

    def __init__(
        self,
        lag=3,
        num_slots=7,  # at most 6 objects per scene
        hidden_dim=128,
        dyn_embedding_dim=0,
        obs_embedding_dim=0,
        trans_prior='NP',
        lr=1e-4,
        infer_mode='F',
        beta=0.0025,
        gamma=0.0075,
        sigma=0.0025,
        num_factors = 72,
        pretrain_savi = None,
        fix_savi = False,
        decoder_dist='gaussian',
        correlation='Pearson',
        if_mcc_in_the_end=True):
        '''Nonlinear ICA for general causal processes with modualar distribution shifts'''
        super().__init__()
        # Transition prior must be L (Linear), NP (Nonparametric)
        assert trans_prior in ('L', 'NP')
        self.obs_dim = 0
        self.dyn_dim = hidden_dim
        self.obs_embedding_dim = obs_embedding_dim
        self.dyn_embedding_dim = dyn_embedding_dim
        self.num_factors = num_factors
        self.z_dim = self.obs_dim + self.dyn_dim
        self.lag = lag
        self.lr = lr
        self.lag = lag
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.infer_mode = infer_mode
        self.num_slots = num_slots
        self.if_mcc_in_the_end = if_mcc_in_the_end
        # Domain embeddings (dynamics)
        if dyn_embedding_dim > 0:
            self.dyn_embed_func = nn.Embedding(nclass, dyn_embedding_dim)
        if obs_embedding_dim > 0:
            self.obs_embed_func = nn.Embedding(nclass, obs_embedding_dim)

        # Factorized inference
        self.net = build_SAVI(num_slots=num_slots)
        self.slot2zs = Slot2Z_dist(hidden_dim,num_slots=num_slots,)  # at most 6 objects per scene)
        self.zs2slot = Z2Slot(hidden_dim,num_slots=num_slots,)  # at most 6 objects per scene)
        
        transition_priors = [ ]
        # Initialize transition prior
        if trans_prior == 'L':
            self.transition_prior = MBDTransitionPrior(lags=lag, 
                                                      latent_size=self.dyn_dim, 
                                                      bias=False)
        elif trans_prior == 'NP':
            self.transition_prior = NPTransitionPrior(lags=lag, 
                                                        latent_size=self.dyn_dim,
                                                        #embedding_dim=dyn_embedding_dim,
                                                        num_layers=4, 
                                                        hidden_dim=hidden_dim)
        
        self.savi_recon_loss_ratio = 0.0 if fix_savi else 1.0
        self.pretrain_savi = pretrain_savi
        
        if self.pretrain_savi is not None:
            self.net.load_state_dict(torch.load(self.pretrain_savi)['state_dict'])
        
        
        # for i in range(3):
        #     if trans_prior == 'L':
        #         transition_prior = MBDTransitionPrior(lags=lag, 
        #                                               latent_size=self.dyn_dim, 
        #                                               bias=False)
        #     elif trans_prior == 'NP':
        #         transition_prior = NPChangeTransitionPrior(lags=lag, 
        #                                                    latent_size=self.dyn_dim,
        #                                                    embedding_dim=dyn_embedding_dim,
        #                                                    num_layers=4, 
        #                                                    hidden_dim=hidden_dim)
        #     transition_priors.append(transition_prior)
        # self.transition_priors = nn.ModuleList(transition_priors)

        # base distribution for calculation of log prob under the model
        if self.dyn_dim > 0:
            self.register_buffer('dyn_base_dist_mean', torch.zeros(self.dyn_dim))
            self.register_buffer('dyn_base_dist_var', torch.eye(self.dyn_dim))
        if self.obs_dim > 0:
            self.register_buffer('obs_base_dist_mean', torch.zeros(self.obs_dim))
            self.register_buffer('obs_base_dist_var', torch.eye(self.obs_dim))
            
        self.zt_true = np.zeros((self.num_factors,0))
        self.zt_recon = np.zeros((self.z_dim,0))

    
    def reset_zt(self):
        self.zt_true = np.zeros((self.num_factors,0))
        self.zt_recon = np.zeros((self.z_dim,0))

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
        x, y, = batch['img'], batch['annotation'], 
        batch_size, length, _ = x.shape #1,4,3,64,64
        outdict = self.net(x)
        
        
        
        #x_flat = x.view(-1, self.input_dim)
        # _, mus, logvars, zs = self.net(x_flat)
        # return zs, mus, logvars
        
        return      

    def training_step(self, batch, batch_idx):
        x, y, = batch['img'], batch['annotation'], 
        
        #c = torch.squeeze(c).to(torch.int64)
        # a = torch.squeeze(a).to(torch.int64)
        batch_size, length, nc, h, w = x.shape
        x_flat = x.view(-1, nc, h, w)
        
        # if self.dyn_dim > 0:
        #     dyn_embeddings = self.dyn_embed_func(c)
        # if self.obs_dim > 0:
        #     obs_embeddings = self.obs_embed_func(c)
        #     obs_embeddings = obs_embeddings.reshape(batch_size,1,self.obs_embedding_dim).repeat(1,length,1)
        # # Inference
        
        kernel_dist, post_slots, encoder_out = self.net.encode(x)
        # [B, T, num_slots, 2C] [B, T, num_slots, C] [B, T, 3, H, W]
        
        # slot -> z  [B, T, num_slots, C] --> [B,T,z_dim]
        mu_var = self.slot2zs(post_slots,)
        mus = mu_var[:, :, :self.z_dim]
        logvars = mu_var[:, :, self.z_dim:]
        zs = self.slot2zs.reparametrize(mus, logvars)
        post_slots_recon =self.zs2slot(zs)
        
        
        x_recon, post_recons, post_masks, _ = \
                self.net.decode(post_slots.flatten(0, 1))
        
        ## slot loss
        slot_kld_loss = self.net.kld_loss(kernel_dist,
                                  post_slots)
        ## slot recon loss
        slot_recon_loss = F.mse_loss(post_slots_recon, post_slots.reshape(post_slots.shape[0],post_slots.shape[1],-1))
        
        
        
        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, nc, h, w)
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
        residuals = [ ]
        logabsdet = [ ]
        # Two action branches
        log_qz_future = log_qz[:,self.lag:]
        residuals, logabsdet = self.transition_prior(zs[:,:,:self.dyn_dim],)
        log_pz_future = torch.sum(self.dyn_base_dist.log_prob(residuals), dim=1) + logabsdet
        future_kld_dyn = (torch.sum(torch.sum(log_qz_future,dim=-1),dim=-1) - log_pz_future) / (length-self.lag)
        future_kld_dyn = future_kld_dyn.mean()
        # for a_idx in range(3):
        #     residuals_action, logabsdet_action = self.transition_priors[a_idx](zs[:,:,:self.dyn_dim], dyn_embeddings)
        #     residuals.append(residuals_action)
        #     logabsdet.append(logabsdet_action)
        # mask = F.one_hot(a+1, num_classes=3)
        # residuals = torch.stack(residuals, axis=1)
        # logabsdet = torch.stack(logabsdet, axis=1)
        # residuals = (residuals * mask[:,:,None,None]).sum(1)
        # logabsdet = (logabsdet * mask).sum(1)
        # log_pz_future = torch.sum(self.dyn_base_dist.log_prob(residuals), dim=1) + logabsdet
        # future_kld_dyn = (torch.sum(torch.sum(log_qz_future,dim=-1),dim=-1) - log_pz_future) / (length-self.lag)
        # future_kld_dyn = future_kld_dyn.mean()

        ### Observation parts ###
        if self.obs_dim > 0:
            p_dist_obs = D.Normal(obs_embeddings[:,:,0].reshape(batch_size, length, 1), 
                                torch.exp(obs_embeddings[:,:,1].reshape(batch_size, length, 1) / 2) )
            log_pz_obs = torch.sum(torch.sum(p_dist_obs.log_prob(zs[:,:,self.dyn_dim:]), dim=1),dim=-1)
            log_qz_obs = torch.sum(torch.sum(log_qz[:,:self.lag,self.dyn_dim:],dim=-1),dim=-1)
            kld_obs = log_qz_obs - log_pz_obs
            kld_obs = kld_obs.mean()
        else:
            kld_obs = 0      

        # VAE training
        # VAE training
        loss = self.savi_recon_loss_ratio * recon_loss + \
                self.beta * past_kld_dyn + self.gamma * future_kld_dyn +\
                self.beta * slot_kld_loss + slot_recon_loss + \
                self.gamma * future_kld_dyn + self.sigma * kld_obs
            
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", past_kld_dyn)
        self.log("train_kld_dynamics", future_kld_dyn)
        self.log("train_kld_observation", kld_obs)
        self.log("train_slot_kld", slot_kld_loss)
        self.log("train_slot_recon", slot_recon_loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, = batch['img'], batch['annotation'], 
        

        batch_size, length, nc, h, w = x.shape
        x_flat = x.view(-1, nc, h, w)
    
        # Inference

        kernel_dist, post_slots, encoder_out = self.net.encode(x)
        # [B, T, num_slots, 2C] [B, T, num_slots, C] [B, T, 3, H, W]
        
        # slot -> z  [B, T, num_slots, C] --> [B,T,z_dim]
        mu_var = self.slot2zs(post_slots,)
        mus = mu_var[:, :, :self.z_dim]
        logvars = mu_var[:, :, self.z_dim:]
        zs = self.slot2zs.reparametrize(mus, logvars)
        post_slots_recon =self.zs2slot(zs)
        
        
        x_recon, post_recons, post_masks, _ = \
                self.net.decode(post_slots.flatten(0, 1))
        
        ## slot loss
        slot_kld_loss = self.net.kld_loss(kernel_dist,
                                  post_slots)
        ## slot recon loss
        slot_recon_loss = F.mse_loss(post_slots_recon, post_slots.reshape(post_slots.shape[0],post_slots.shape[1],-1))
        

        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, nc, h, w)
        
        
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
        residuals = [ ]
        logabsdet = [ ]
        # Two action branches
        
        log_qz_future = log_qz[:,self.lag:]
        residuals, logabsdet = self.transition_prior(zs[:,:,:self.dyn_dim], )
        log_pz_future = torch.sum(self.dyn_base_dist.log_prob(residuals), dim=1) + logabsdet
        future_kld_dyn = (torch.sum(torch.sum(log_qz_future,dim=-1),dim=-1) - log_pz_future) / (length-self.lag)
        future_kld_dyn = future_kld_dyn.mean()
        
        # for a_idx in range(3):
        #     residuals_action, logabsdet_action = self.transition_priors[a_idx](zs[:,:,:self.dyn_dim], dyn_embeddings)
        #     residuals.append(residuals_action)
        #     logabsdet.append(logabsdet_action)
        # mask = F.one_hot(a+1, num_classes=3)
        # residuals = torch.stack(residuals, axis=1)
        # logabsdet = torch.stack(logabsdet, axis=1)
        # residuals = (residuals * mask[:,:,None,None]).sum(1)
        # logabsdet = (logabsdet * mask).sum(1)
        # log_pz_future = torch.sum(self.dyn_base_dist.log_prob(residuals), dim=1) + logabsdet
        # future_kld_dyn = (torch.sum(torch.sum(log_qz_future,dim=-1),dim=-1) - log_pz_future) / (length-self.lag)
        # future_kld_dyn = future_kld_dyn.mean()

        ### Observation parts ###
        if self.obs_dim > 0:
            p_dist_obs = D.Normal(obs_embeddings[:,:,0].reshape(batch_size, length, 1), 
                                torch.exp(obs_embeddings[:,:,1].reshape(batch_size, length, 1) / 2) )
            log_pz_obs = torch.sum(torch.sum(p_dist_obs.log_prob(zs[:,:,self.dyn_dim:]), dim=1),dim=-1)
            log_qz_obs = torch.sum(torch.sum(log_qz[:,:self.lag,self.dyn_dim:],dim=-1),dim=-1)
            kld_obs = log_qz_obs - log_pz_obs
            kld_obs = kld_obs.mean()
        else:
            kld_obs = 0      

        # VAE validating loss
        loss = self.savi_recon_loss_ratio * recon_loss + self.beta * past_kld_dyn + \
            self.beta * slot_kld_loss + slot_recon_loss + \
            self.gamma * future_kld_dyn + self.sigma * kld_obs

        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus[:,0].view(-1, self.z_dim).T.detach().cpu().numpy()
        # zt_true = batch["yt"][:,0].view(-1, 20).T.detach().cpu().numpy()
        zt_true = batch["annotation"][:,-1].view(-1, batch["annotation"].shape[-1]*batch['annotation'].shape[-2]).T.detach().cpu().numpy()
        #mcc = compute_mcc(zt_recon, zt_true, self.correlation)
        
        # record zt_recon and zt_true
        def stack_arrays_horizontally(a, b):
            """
            Stack two numpy arrays horizontally (along their second axis).
            
            Args:
                a: numpy array with shape (dim, n)
                b: numpy array with shape (dim, 1)
            
            Returns:
                c: numpy array with shape (dim, n+1)
            """
            # reshape b to have shape (dim,)
            
            #assert a.shape[0] == b.shape[0]
            
            if a.shape[0] > b.shape[0]:
                # padding
                noise = np.random.normal(0, 1, (a.shape[0]-b.shape[0], b.shape[1]))
                b = np.vstack((b, noise))
                
            
            if a.shape[1] == 0:
                return b
            
            # stack a and b horizontally
            c = np.hstack((a, b))
            return c
        
        if self.if_mcc_in_the_end is True:
            self.zt_recon = stack_arrays_horizontally(self.zt_recon, zt_recon)
            self.zt_true = stack_arrays_horizontally(self.zt_true, zt_true)
        
        if self.if_mcc_in_the_end is False:
            mcc = compute_mcc(zt_recon, zt_true, self.correlation)
            self.log("val_mcc", mcc) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_normal", past_kld_dyn)
        self.log("val_kld_dynamics", future_kld_dyn)
        self.log("val_kld_observation", kld_obs)
        self.log("val_slot_kld", slot_kld_loss)
        self.log("val_slot_recon", slot_recon_loss)

        return loss
    
    def validation_epoch_end(self, outputs):
        # Compute Mean Correlation Coefficient (MCC)
        if self.if_mcc_in_the_end is False:
            return
        mcc = compute_mcc(self.zt_recon, self.zt_true, self.correlation)
        self.log("val_mcc", mcc)
        
        # reset zt_recon and zt_true
        self.reset_zt()
    
    # def sample(self, n=64):
    #     with torch.no_grad():
    #         e = torch.randn(n, self.z_dim, device=self.device)
    #         eps, _ = self.spline.inverse(e)
    #     return eps

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []