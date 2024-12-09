"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
import torch.optim as optim
import wandb
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from .components.beta import BetaVAE_MLP, BetaVAE_MLP_independentnoise
from .components.transition import NPChangeTransitionPrior
from .components.mlp import MLPEncoder, MLPDecoder, Inference, NLayerLeakyMLP, TApproximator
from .components.flow import ComponentWiseCondSpline
from .metrics.correlation import compute_mcc
from Caulimate.Utils.Tools import check_array, check_tensor
from Caulimate.Utils.Lego import PartiallyPeriodicMLP

import ipdb as pdb

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
        correlation='Pearson',
        graph_thres=0.01,
        B_init=None):
        '''Nonlinear ICA for general causal processes with modualar distribution shifts'''
        super().__init__()
        # Transition prior must be L (Linear), NP (Nonparametric)
        self.save_hyperparameters()
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
        self.graph_thres = graph_thres
        # Domain embeddings (dynamics)
        self.dyn_embed_func = nn.Embedding(nclass, dyn_embedding_dim)
        self.obs_embed_func = nn.Embedding(nclass, obs_embedding_dim)
        
        # Flow for nonstationary regimes
        # self.flow = ComponentWiseCondSpline(input_dim=self.obs_dim,
        #                                     context_dim=obs_embedding_dim,
        #                                     bound=bound,
        #                                     count_bins=count_bins,
        #                                     order=order)
        # Factorized inference
        # self.net = BetaVAE_MLP(input_dim=input_dim, 
        #                         z_dim=self.z_dim, 
        #                         hidden_dim=hidden_dim)
        self.net = BetaVAE_MLP_independentnoise(input_dim=input_dim, 
                                z_dim=self.z_dim, 
                                hidden_dim=hidden_dim)
        
        
        
        # Initialize transition prior
        if trans_prior == 'L':
            raise ValueError()
            # self.transition_prior = MBDTransitionPrior(lags=lag, 
            #                                            latent_size=self.dyn_dim, 
            #                                            bias=False)
        elif trans_prior == 'NP':
            self.transition_prior = NPChangeTransitionPrior(lags=lag, 
                                                            latent_size=self.dyn_dim,
                                                            embedding_dim=dyn_embedding_dim,
                                                            num_layers=4, 
                                                            hidden_dim=hidden_dim)
            
        # Initialize causal adjacency matrix in observed variables
        if B_init is not None:
            self.B = nn.Parameter(check_tensor(B_init))
        else:
            self.B = nn.Parameter(check_tensor(torch.randn(self.input_dim, self.input_dim)))
        # for i in range(1, self.input_dim):
        #     self.B.data[i, i - 1] = 1
        # self.linearM = nn.Linear(self.input_dim, self.input_dim, bias=False)
        # torch.nn.init.xavier_uniform(self.linearM.weight)
        #self.xnoise = nn.Parameter(torch.tensor(0.7071))
        self.xnoise = check_tensor(torch.tensor(0.1))

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
        
    def neg_loglikelihood_loss(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0
        distribution = torch.distributions.normal.Normal(x_recon, self.xnoise)
        likelihood = distribution.log_prob(x)
        recon_loss = -likelihood.sum().div(batch_size)
        return recon_loss
    
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
            
        elif distribution == 'conditional_gaussian':
            recon_loss = self.neg_loglikelihood_loss(x, x_recon, distribution)

        return recon_loss
    
    def DAG_loss(self, B):
        if len(B.shape) == 2:
            B = B.unsqueeze(0)  
        matrix_exp = torch.exp(B * B)
        traces = torch.sum(torch.diagonal(matrix_exp, dim1=-2, dim2=-1)) - B.shape[0] * B.shape[1]
        return traces

    def forward(self, batch):
        print("forward")
        return 

    def training_step(self, batch, batch_idx):
        x, y, c, b, v = batch['xt'], batch['yt'], batch['ct'], batch['bt'], batch['vt']
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        # x to (I-B)x
        x_flat = x_flat#torch.matmul(self.M[None, :, :].repeat(x_flat.shape[0], 1, 1), x_flat.unsqueeze(2)).squeeze(2)
        #x = x_flat.reshape(-1, 3, self.input_dim)
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size, 1, self.obs_embedding_dim).repeat(1,length,1)
        
        B = self.B #torch.tril(self.B, diagonal=-1)
        # Inference
        y_recon, self.x_noise, mus, logvars, zs = self.net(x_flat)

        # (I-B)x to x
        #x_recon = self.linearM(y_recon) 
        x_recon = y_recon#torch.matmul(y_recon, torch.inverse(check_tensor(torch.eye(self.input_dim)) - B))
        
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
        # M_sparsity = 0 
        # for param in self.linearM.parameters():
        #     M_sparsity += torch.norm(param, 1)  # L1 norm
        # W_sparsity = torch.norm(torch.inverse(self.linearM.weight), 1)
        B_sparsity = torch.norm(torch.tril(self.B), 1)
        B_DAG_loss = self.DAG_loss(torch.tril(self.B))
        # VAE training
        loss = recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs 
        if B_sparsity > 1:
            loss += self.B_sparsity * B_sparsity
        loss += 1e-2 * B_DAG_loss
        #########################   training step  #########################
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = batch["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        train_z_mcc = compute_mcc(zt_recon, zt_true, self.correlation)
        print('train_z_mcc: {}'.format(train_z_mcc))
        self.log("train_z_mcc", train_z_mcc)
        
        train_y_mcc = compute_mcc(y_recon.view(-1, self.input_dim).T.detach().cpu().numpy(), v.view(-1, self.input_dim).T.detach().cpu().numpy(), self.correlation)
        self.log("train_y_mcc", train_y_mcc)
        
        train_x_y_mcc = compute_mcc(y_recon.view(-1, self.input_dim).T.detach().cpu().numpy(), x.view(-1, self.input_dim).T.detach().cpu().numpy(), self.correlation)
        self.log("train_x_y_mcc", train_x_y_mcc)
        
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", past_kld_dyn)
        self.log("train_kld_dynamics", future_kld_dyn)
        self.log("train_kld_observation", kld_obs)
        self.log("train_B_sparsity", B_sparsity)
        self.log("train_B_DAG", B_DAG_loss)
        
        if train_z_mcc > 0.8:
            print('recon noise free:{}'.format(torch.nn.MSELoss()(v, x_recon)))
            print(v[0][0], x_recon[0][0], x[0][0])
        t_recon_loss = self.reconstruction_loss(x[:,:self.lag], v[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], v[:,self.lag:], self.decoder_dist))/(length-self.lag)
        self.log("train_true_recon_loss", t_recon_loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, c, b, v = batch['xt'], batch['yt'], batch['ct'], batch['bt'], batch['vt']
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        B = self.B#torch.tril(self.B, diagonal=-1)
        # x to (I-B)x
        x_flat = x_flat#torch.matmul(self.M[None, :, :].repeat(x_flat.shape[0], 1, 1), x_flat.unsqueeze(2)).squeeze(2)
        #x = x_flat.reshape(-1, 3, self.input_dim)
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size, 1, self.obs_embedding_dim).repeat(1,length,1)
        
        # Inference
        y_recon, self.x_noise, mus, logvars, zs = self.net(x_flat)
        # (I-B)x to x
        #x_recon = self.linearM(y_recon) 
        x_recon = y_recon#torch.matmul(y_recon, torch.inverse(check_tensor(torch.eye(self.input_dim)) - B))
        
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
        # M_sparsity = 0 
        # for param in self.linearM.parameters():
        #     M_sparsity += torch.norm(param, 1)  # L1 norm
        # W_sparsity = torch.norm(torch.inverse(self.linearM.weight), 1)
        B_sparsity = torch.norm(B, 1)
        B_DAG_loss = self.DAG_loss(B)
        # VAE training
        loss = recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs 
        if B_sparsity > 1:
            loss += self.B_sparsity * B_sparsity
        loss += 1e-2 * B_DAG_loss
        #########################   validation step  #########################
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()

        zt_true = batch["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)
        
        #self.log("val_B_recon", B_recon)
        B_gt = check_array(b[0])
        # W_gt = np.eye(self.input_dim) - B_gt
        # M_gt = np.linalg.inv(W_gt) # x = g_z @ M_gt 
        
        # M_est = check_array(self.linearM.weight) # x = M_est @ g_z
        # W_est = np.linalg.inv(M_est)
        # W_est[np.abs(W_est) < 0.14] = 0
        
        # _, col_index = linear_sum_assignment(1 / np.abs(W_est))
        # PW_ica = np.zeros_like(W_est)
        # PW_ica[col_index] = W_est
        # # obtain a vector to scale
        # D_ = np.diag(PW_ica)[:, np.newaxis]
        # # estimate an adjacency matrix
        # W_est_postrocess = PW_ica / D_
        B_est = check_array(B)#np.eye(len(W_est_postrocess)) - W_est_postrocess
        
        #M_post = postprocess(M_est, graph_thres=self.graph_thres)
        fig = plot_solutions([B_gt, B_est], ['B_gt', 'B_est'], add_value=True, logger=self.logger)
        self.logger.experiment.log({"Fig": [wandb.Image(fig)]})
        

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
        #self.log("val_B_sparsity", B_sparsity)

        return loss
    
    def sample(self, n=64):
        with torch.no_grad():
            e = torch.randn(n, self.z_dim, device=self.device)
            eps, _ = self.spline.inverse(e)
        return eps

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []