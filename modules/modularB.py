"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
import torch.optim as optim
import wandb
import io
from torch.nn import functional as F
from .components.beta import BetaVAE_MLP
from .components.transition import (MBDTransitionPrior, 
                                    NPChangeTransitionPrior)
from .components.mlp import MLPEncoder, MLPDecoder, Inference, NLayerLeakyMLP, TApproximator
from .components.flow import ComponentWiseCondSpline
from .metrics.correlation import compute_mcc
from ..minghao_utils import check_tensor, check_array, threshold_till_dag, count_accuracy, bin_mat, postprocess, top_k_abs_tensor, plot_solution, mask_tri

import ipdb as pdb
    
class causal_state(pl.LightningModule):
    def __init__(
        self,
        nclass,
        output_dim, 
        sea_embedding_dim=2,
        activation=None,
    ):
        super().__init__()
        self.sea_embedding = nn.Embedding(nclass, output_dim**2)
        self.net_s = NLayerLeakyMLP(1, output_dim**2, num_layers=3)
        self.net_h = TApproximator(output_dim, output_dim**2, 1000, 0.2)#NLayerLeakyMLP(1, output_dim**2, num_layers=4) # nn.Linear(u_dim, output_dim**2) #
        self.output_dim = output_dim    
        self.activation = activation
        self.start_epoch = 0
        self.Bs = check_tensor(torch.eye(output_dim))
        
    def pretrain_sparse(self, train_loader, pretrain_epochs):
        optimizer = optim.SGD(self.parameters(), lr=0.01)  # Choose an appropriate optimizer and learning rate

        for epoch in range(pretrain_epochs):
            for batch_data in train_loader:
                x, h, s = batch_data['xt'], batch_data['ht'], batch_data['ct']
                s = check_tensor(s.unsqueeze(1).to(torch.int64))
                optimizer.zero_grad()
                output = self(x, s, h)  # Forward pass through the network
                loss = torch.mean(output ** 2)  # Define the loss as the mean squared output
                
                loss.backward()  # Backward pass to compute gradients
                optimizer.step()  # Update the network's parameters based on the gradients
        print('--- Pretrained sparse model loss: {}'.format(loss))
        
    def generate_I_B(self, x, s, h, Bs=None):
        '''
            x: [batch_size, output_dim]
            u: [batch_size, u_dim]
        '''
        flat_batch_size = x.shape[0]
        batch_size = s.shape[0]
        length = flat_batch_size // batch_size
        if Bs is not None:
            mat_Bs = Bs
        else:
            # get causal hidden state
            #s = torch.unsqueeze(s, dim=1).to(torch.float32)
            flat_sea_vary = self.sea_embedding(s)
            flat_h_vary = self.net_h(h) 
            if self.activation is not None:
                flat_h_vary = self.activation(flat_h_vary)
            mat_Bs = (flat_h_vary * flat_sea_vary + flat_sea_vary).reshape(-1, self.output_dim, self.output_dim)
            
            for i in range(self.output_dim):
                mat_Bs[:, i, i] = 0
                
            mat_Bs = mask_tri(mat_Bs / max(100 / (self.current_epoch + 1), 1), 3)

        # store I and B for reconstruct
        self.batch_Bs = mat_Bs
        # reshape to lags+length
        if self.current_epoch < self.start_epoch and self.training:
            self.Bs = check_tensor(torch.zeros((flat_batch_size, self.output_dim, self.output_dim)), astype=x) 
        else:
            self.Bs = mat_Bs.repeat_interleave(length, dim=0) #check_tensor(torch.zeros((flat_batch_size, self.output_dim, self.output_dim)), astype=x) 
        self.Is = check_tensor(torch.eye(self.output_dim).unsqueeze(0).repeat(flat_batch_size, 1, 1), astype=x)

class disentangle(causal_state):
    def __init__(
        self,
        nclass,
        output_dim, 
        sea_embedding_dim=2,
        activation=None,
    ):
        super().__init__(
            nclass,
            output_dim, 
            sea_embedding_dim,
            activation)
        
        #self.net = NLayerLeakyMLP(output_dim + 2, output_dim, num_layers=4)
        
    def forward(self, x, s, h, Bs=None):
        '''
            x: [batch_size, output_dim]
            u: [batch_size, u_dim]
        '''
        
        self.generate_I_B(x, s, h, Bs)
        x = torch.bmm((self.Is - self.Bs), x.unsqueeze(2)).squeeze(2)
        return x
        # flat_batch_size = x.shape[0]
        # batch_size = s.shape[0]
        # length = flat_batch_size // batch_size
        # return self.net(torch.cat([x, s.unsqueeze(1).repeat_interleave(length, dim=0), h.repeat_interleave(length, dim=0)], dim=1))
    
class reconstruct(causal_state):
    def __init__(
        self,
        nclass,
        output_dim, 
        sea_embedding_dim=2,
        activation=None,
    ):
        super().__init__(
            nclass,
            output_dim, 
            sea_embedding_dim,
            activation)
        
    def forward(self, x, s, h, Bs=None):
        '''
            x: [batch_size, output_dim]
            u: [batch_size, u_dim]
        '''
        #x = F.sigmoid(x)
        self.generate_I_B(x, s, h, Bs)
        x = torch.linalg.solve(self.Is - self.Bs, x.unsqueeze(2)).squeeze(2)
        
        return x

    def neg_likelihood(self, x, g_z):
        '''
            x: [batch, length, output_dim]
            gz: [batch, length, output_dim]
        '''
        mu = torch.bmm(torch.inverse(self.Is - self.Bs), g_z.unsqueeze(2)).squeeze(2)
        Sigma = 0.1 * torch.bmm(torch.inverse(self.Is - self.Bs), torch.inverse(self.Is - self.Bs).permute(0, 2, 1))
        
        likelihood = - 0.5 * self.output_dim * (torch.bmm((x - mu).unsqueeze(1), Sigma).squeeze(1) * (x - mu)).sum(dim=1)
        - 0.5 * torch.linalg.slogdet(Sigma)[1] 
        - 0.5 * self.output_dim * torch.log(torch.tensor(torch.pi))

        return -likelihood.sum()
    
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
        correlation='Pearson',
        graph_thres=0.1):
        '''Nonlinear ICA for general causal processes with modualar distribution shifts'''
        super().__init__()
        self.save_hyperparameters()
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
        self.graph_thres = graph_thres
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

        self.disentangle = disentangle(nclass=nclass,
                                                output_dim=self.input_dim, 
                                                activation=F.tanh)
        self.reconstruct = reconstruct(nclass=nclass,
                                        output_dim=self.input_dim, 
                                        activation=F.tanh)

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

    def DAG_loss(self, B):
        if len(B.shape) == 2:
            B = B.unsqueeze(0)  
        matrix_exp = torch.exp(B * B)
        traces = torch.sum(torch.diagonal(matrix_exp, dim1=-2, dim2=-1)) - B.shape[0] * B.shape[1]
        return traces
    
    def forward(self, batch):
        x, y, c, h, b = batch['xt'], batch['yt'], batch['ct'], batch['ht'], batch['bt']
        batch_size, length, _ = x.shape
        u = h#torch.cat([c, h], dim=-1)
        
        x_flat = x.view(-1, self.input_dim)
        # x to (I-B)x
        x_flat = self.disentangle(x_flat, c, h, b)
        _, mus, logvars, zs = self.net(x_flat)
        return zs, mus, logvars       

    def training_step(self, batch, batch_idx):
        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
                
        x, y, v, c, h, b = batch['xt'], batch['yt'], batch['vt'], batch['ct'], batch['ht'], batch['bt']
        u = torch.cat([c, h], dim=-1)
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        # x to (I-B)x
        v_flat = self.disentangle(x_flat, c, h)
        print('true v / disen_v mse: {}'.format(self.reconstruction_loss(v.reshape(v_flat.shape), v_flat, 'gaussian')))
        #x = x_flat.reshape(-1, 3, self.input_dim)
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size, 1, self.obs_embedding_dim).repeat(1,length,1)
        
        # Inference
        v_recon, mus, logvars, zs = self.net(v_flat)
        
        # (I-B)x to x
        x_recon = self.reconstruct(v_recon, c, h)
        #neg_likelihood = self.reconstruct.neg_likelihood(x_flat, v_recon)
        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
        v_recon_loss = self.reconstruction_loss(v_flat, v_recon, self.decoder_dist) 
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
        B_sparsity = torch.norm(self.reconstruct.Bs, p=1) / x.shape[0]
        
        # B reconstruction
        #B_recon = self.reconstruction_loss(self.disentangle.Bs + self.reconstruct.Bs, b, 'gaussian') # no training
        #self.log("train_B_recon", B_recon)
        
        # VAE training
        # self.reconstruction_loss(v.reshape(v_recon.shape), v_recon, 'gaussian')
        loss = recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs + self.B_sparsity * B_sparsity# + B_recon
        #loss += 0.001 * v_recon_loss
        v2x_x = self.reconstruction_loss(v.reshape(v_recon.shape), v_recon, 'gaussian')
        print('true v / v_recon: {}'.format(v2x_x))
        ## DAG
        # DAG_loss = self.DAG_loss(self.reconstruct.batch_Bs)
        # loss += 0.001 * DAG_loss
        #########################   training step  #########################
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()

        zt_true = batch["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)
        self.log("train_mcc", mcc)
        print("train_mcc: {}".format(mcc))
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_v_recon_loss", v_recon_loss)
        self.log("train_kld_normal", past_kld_dyn)
        self.log("train_kld_dynamics", future_kld_dyn)
        self.log("train_kld_observation", kld_obs)
        self.log("train_B_sparsity", B_sparsity)
        
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, v, c, h, b = batch['xt'], batch['yt'], batch['vt'], batch['ct'], batch['ht'], batch['bt']
        u = torch.cat([c, h], dim=-1)
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        # x to (I-B)x
        v_flat = self.disentangle(x_flat, c, h, b)
        #x = x_flat.reshape(-1, 3, self.input_dim)
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size, 1, self.obs_embedding_dim).repeat(1,length,1)
        
        # Inference
        v_recon, mus, logvars, zs = self.net(v_flat)
        # (I-B)x to x
        x_recon = self.reconstruct(v_recon, c, h)
        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)
        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
        v_recon_loss = self.reconstruction_loss(v_flat, v_recon, self.decoder_dist) 
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
        B_sparsity = torch.norm(self.reconstruct.Bs, p=1) / x.shape[0]
        
        # B reconstruction
        #B_recon = self.reconstruction_loss(self.disentangle.Bs + self.reconstruct.Bs, b, 'gaussian') # no training
        #self.log("train_B_recon", B_recon)
        # VAE training
        loss = recon_loss + 0.001 * v_recon_loss + 0.001 + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs + self.B_sparsity * B_sparsity
        #########################   validation step  #########################
        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()

        zt_true = batch["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)
        # count accuracy of B predictions
        # batch_B_tpr = []
        # batch_B_fdr = []
        # batch_B_shd = []
        # for i in range(batch_size):
        #     graph_thres = 0.1
        #     #B_est = postprocess(check_array(self.causal_state.batch_Bs[i]), graph_thres=graph_thres)
        #     n = torch.nonzero(b[i]).size(0)
        #     B_est = check_array(top_k_abs_tensor(self.causal_state.batch_Bs[i], n))
        #     B_gt = check_array(b[i])
        #     acc = count_accuracy(bin_mat(B_est), bin_mat(B_gt))
        #     batch_B_tpr.append(acc['tpr'])
        #     batch_B_fdr.append(acc['fdr'])
        #     batch_B_shd.append(acc['shd'])
            
        # print(np.nonzero(B_est), np.nonzero(B_gt))
        # B_avg_tpr = np.mean(batch_B_tpr)
        # B_avg_fdr = np.mean(batch_B_fdr)
        # B_avg_shd = np.mean(batch_B_shd)
        
        # self.log("val_B_avg_tpr", B_avg_tpr)
        # self.log("val_B_avg_fdr", B_avg_fdr)
        # self.log("val_B_avg_shd", B_avg_shd)
        
        self.log("val_mcc", mcc) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_v_recon_loss", v_recon_loss)
        self.log("val_kld_normal", past_kld_dyn)
        self.log("val_kld_dynamics", future_kld_dyn)
        self.log("val_kld_observation", kld_obs)
        #self.log("val_B_sparsity", B_sparsity)
        
        
        #self.log("val_B_recon", B_recon)
        fig = plot_solution(check_array(b[10]), check_array(self.reconstruct.batch_Bs[10]), postprocess(check_array(self.reconstruct.batch_Bs[10]), graph_thres=self.graph_thres), logger=self.logger)
        self.logger.experiment.log({"B": [wandb.Image(fig)]})
        
        return loss
    
    def sample(self, n=64):
        with torch.no_grad():
            e = torch.randn(n, self.z_dim, device=self.device)
            eps, _ = self.spline.inverse(e)
        return eps

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []
    
# class ModularShiftsFixedB(pl.LightningModule):
#     def __init__(
#         self, 
#         input_dim,
#         length,
#         obs_dim,
#         dyn_dim, 
#         lag,
#         nclass,
#         hidden_dim=128,
#         dyn_embedding_dim=2,
#         obs_embedding_dim=2,
#         trans_prior='NP',
#         lr=1e-4,
#         infer_mode='F',
#         bound=5,
#         count_bins=8,
#         order='linear',
#         beta=0.0025,
#         gamma=0.0075,
#         sigma=0.0025,
#         B_sparsity=0.0025,
#         decoder_dist='gaussian',
#         correlation='Pearson'):
#         '''Nonlinear ICA for general causal processes with modualar distribution shifts'''
#         super().__init__()
#         # Transition prior must be L (Linear), NP (Nonparametric)
#         assert trans_prior in ('L', 'NP')
#         self.obs_dim = obs_dim
#         self.dyn_dim = dyn_dim
#         self.obs_embedding_dim = obs_embedding_dim
#         self.dyn_embedding_dim = dyn_embedding_dim
#         self.z_dim = obs_dim + dyn_dim
#         self.lag = lag
#         self.input_dim = input_dim
#         self.lr = lr
#         self.lag = lag
#         self.length = length
#         self.beta = beta
#         self.gamma = gamma
#         self.sigma = sigma
#         self.B_sparsity = B_sparsity
#         self.correlation = correlation
#         self.decoder_dist = decoder_dist
#         self.infer_mode = infer_mode
#         # Domain embeddings (dynamics)
#         self.dyn_embed_func = nn.Embedding(nclass, dyn_embedding_dim)
#         self.obs_embed_func = nn.Embedding(nclass, obs_embedding_dim)
#         # Flow for nonstationary regimes
#         self.flow = ComponentWiseCondSpline(input_dim=self.obs_dim,
#                                             context_dim=obs_embedding_dim,
#                                             bound=bound,
#                                             count_bins=count_bins,
#                                             order=order)
#         # Factorized inference
#         self.net = BetaVAE_MLP(input_dim=input_dim, 
#                                 z_dim=self.z_dim, 
#                                 hidden_dim=hidden_dim)

#         # Initialize transition prior
#         if trans_prior == 'L':
#             self.transition_prior = MBDTransitionPrior(lags=lag, 
#                                                        latent_size=self.dyn_dim, 
#                                                        bias=False)
#         elif trans_prior == 'NP':
#             self.transition_prior = NPChangeTransitionPrior(lags=lag, 
#                                                             latent_size=self.dyn_dim,
#                                                             embedding_dim=dyn_embedding_dim,
#                                                             num_layers=4, 
#                                                             hidden_dim=hidden_dim)
            
#         # Initialize causal adjacency matrix in observed variables
#         self.B = nn.Parameter(check_tensor(torch.randn(self.input_dim, self.input_dim)))
#         self.I = check_tensor(torch.eye(self.input_dim), astype=self.B)

#         # base distribution for calculation of log prob under the model
#         self.register_buffer('dyn_base_dist_mean', torch.zeros(self.dyn_dim))
#         self.register_buffer('dyn_base_dist_var', torch.eye(self.dyn_dim))
#         self.register_buffer('obs_base_dist_mean', torch.zeros(self.obs_dim))
#         self.register_buffer('obs_base_dist_var', torch.eye(self.obs_dim))
        
#     @property
#     def dyn_base_dist(self):
#         # Noise density function
#         return D.MultivariateNormal(self.dyn_base_dist_mean, self.dyn_base_dist_var)

#     @property
#     def obs_base_dist(self):
#         # Noise density function
#         return D.MultivariateNormal(self.obs_base_dist_mean, self.obs_base_dist_var)
    
#     def reparameterize(self, mean, logvar, random_sampling=True):
#         if random_sampling:
#             eps = torch.randn_like(logvar)
#             std = torch.exp(0.5*logvar)
#             z = mean + eps*std
#             return z
#         else:
#             return mean

#     def reconstruction_loss(self, x, x_recon, distribution):
#         batch_size = x.size(0)
#         assert batch_size != 0

#         if distribution == 'bernoulli':
#             recon_loss = F.binary_cross_entropy_with_logits(
#             x_recon, x, size_average=False).div(batch_size) 

#         elif distribution == 'gaussian':
#             recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

#         elif distribution == 'sigmoid_gaussian':
#             x_recon = F.sigmoid(x_recon)
#             recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

#         return recon_loss

#     def forward(self, batch):
#         x, y, c = batch['xt'], batch['yt'], batch['ct']
#         batch_size, length, _ = x.shape
#         x_flat = x.view(-1, self.input_dim)
#         # x to (I-B)x
#         x_flat = self.causal_state.disentangle(x_flat, u)
#         _, mus, logvars, zs = self.net(x_flat)
#         return zs, mus, logvars       

#     def training_step(self, batch, batch_idx):
#         x, y, c = batch['xt'], batch['yt'], batch['ct']
#         c = torch.squeeze(c).to(torch.int64)
#         batch_size, length, _ = x.shape
#         x_flat = x.view(-1, self.input_dim)
#         # x to (I-B)x
#         x_flat = torch.matmul(x_flat, (self.I - self.B))
#         #x = x_flat.reshape(-1, 3, self.input_dim)
#         dyn_embeddings = self.dyn_embed_func(c)
#         obs_embeddings = self.obs_embed_func(c)
#         obs_embeddings = obs_embeddings.reshape(batch_size, 1, self.obs_embedding_dim).repeat(1,length,1)
        
#         # Inference
#         x_recon, mus, logvars, zs = self.net(x_flat)
#         # (I-B)x to x
#         x_recon = torch.matmul(x_recon, torch.inverse(self.I - self.B))
#         # Reshape to time-series format
#         x_recon = x_recon.view(batch_size, length, self.input_dim)
#         mus = mus.reshape(batch_size, length, self.z_dim)
#         logvars  = logvars.reshape(batch_size, length, self.z_dim)
#         zs = zs.reshape(batch_size, length, self.z_dim)
#         # VAE ELBO loss: recon_loss + kld_loss
#         recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
#         (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
#         q_dist = D.Normal(mus, torch.exp(logvars / 2))
#         log_qz = q_dist.log_prob(zs)

#         ### Dynamics parts ###
#         # Past KLD <=> N(0,1) #
#         p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag,:self.dyn_dim]), torch.ones_like(logvars[:,:self.lag, :self.dyn_dim]))
#         log_pz_past = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag,:self.dyn_dim]),dim=-1),dim=-1)
#         log_qz_past = torch.sum(torch.sum(log_qz[:,:self.lag,:self.dyn_dim],dim=-1),dim=-1)
#         past_kld_dyn = log_qz_past - log_pz_past
#         past_kld_dyn = past_kld_dyn.mean()
#         # Future KLD #
#         log_qz_future = log_qz[:,self.lag:]
#         residuals, logabsdet = self.transition_prior(zs[:,:,:self.dyn_dim], dyn_embeddings)
#         log_pz_future = torch.sum(self.dyn_base_dist.log_prob(residuals), dim=1) + logabsdet
#         future_kld_dyn = (torch.sum(torch.sum(log_qz_future,dim=-1),dim=-1) - log_pz_future) / (length-self.lag)
#         future_kld_dyn = future_kld_dyn.mean()

#         ### Observation parts ###
#         p_dist_obs = D.Normal(obs_embeddings[:,:,0].reshape(batch_size, length, 1), 
#                               torch.exp(obs_embeddings[:,:,1].reshape(batch_size, length, 1) / 2) )
#         log_pz_obs = torch.sum(torch.sum(p_dist_obs.log_prob(zs[:,:,self.dyn_dim:]), dim=1),dim=-1)
#         log_qz_obs = torch.sum(torch.sum(log_qz[:,:self.lag,self.dyn_dim:],dim=-1),dim=-1)
#         kld_obs = log_qz_obs - log_pz_obs
#         kld_obs = kld_obs.mean()      

#         # sparsity
#         B_sparsity = torch.norm(self.B, p=1)
        
#         # VAE training
#         loss = recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs + self.B_sparsity * B_sparsity# + B_recon
#         #########################   training step  #########################
#         self.log("train_elbo_loss", loss)
#         self.log("train_recon_loss", recon_loss)
#         self.log("train_kld_normal", past_kld_dyn)
#         self.log("train_kld_dynamics", future_kld_dyn)
#         self.log("train_kld_observation", kld_obs)
#         self.log("train_B_sparsity", B_sparsity)
        
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         x, y, c = batch['xt'], batch['yt'], batch['ct']
#         c = torch.squeeze(c).to(torch.int64)
#         batch_size, length, _ = x.shape
#         x_flat = x.view(-1, self.input_dim)
#         # x to (I-B)x
#         x_flat = torch.matmul(x_flat, (self.I - self.B))
#         #x = x_flat.reshape(-1, 3, self.input_dim)
#         dyn_embeddings = self.dyn_embed_func(c)
#         obs_embeddings = self.obs_embed_func(c)
#         obs_embeddings = obs_embeddings.reshape(batch_size, 1, self.obs_embedding_dim).repeat(1,length,1)
        
#         # Inference
#         x_recon, mus, logvars, zs = self.net(x_flat)
#         # (I-B)x to x
#         x_recon = torch.matmul(x_recon, torch.inverse(self.I - self.B))
#         # Reshape to time-series format
#         x_recon = x_recon.view(batch_size, length, self.input_dim)
#         mus = mus.reshape(batch_size, length, self.z_dim)
#         logvars  = logvars.reshape(batch_size, length, self.z_dim)
#         zs = zs.reshape(batch_size, length, self.z_dim)
#         # VAE ELBO loss: recon_loss + kld_loss
#         recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
#         (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
#         q_dist = D.Normal(mus, torch.exp(logvars / 2))
#         log_qz = q_dist.log_prob(zs)

#         ### Dynamics parts ###
#         # Past KLD <=> N(0,1) #
#         p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag,:self.dyn_dim]), torch.ones_like(logvars[:,:self.lag, :self.dyn_dim]))
#         log_pz_past = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag,:self.dyn_dim]),dim=-1),dim=-1)
#         log_qz_past = torch.sum(torch.sum(log_qz[:,:self.lag,:self.dyn_dim],dim=-1),dim=-1)
#         past_kld_dyn = log_qz_past - log_pz_past
#         past_kld_dyn = past_kld_dyn.mean()
#         # Future KLD #
#         log_qz_future = log_qz[:,self.lag:]
#         residuals, logabsdet = self.transition_prior(zs[:,:,:self.dyn_dim], dyn_embeddings)
#         log_pz_future = torch.sum(self.dyn_base_dist.log_prob(residuals), dim=1) + logabsdet
#         future_kld_dyn = (torch.sum(torch.sum(log_qz_future,dim=-1),dim=-1) - log_pz_future) / (length-self.lag)
#         future_kld_dyn = future_kld_dyn.mean()

#         ### Observation parts ###
#         p_dist_obs = D.Normal(obs_embeddings[:,:,0].reshape(batch_size, length, 1), 
#                               torch.exp(obs_embeddings[:,:,1].reshape(batch_size, length, 1) / 2) )
#         log_pz_obs = torch.sum(torch.sum(p_dist_obs.log_prob(zs[:,:,self.dyn_dim:]), dim=1),dim=-1)
#         log_qz_obs = torch.sum(torch.sum(log_qz[:,:self.lag,self.dyn_dim:],dim=-1),dim=-1)
#         kld_obs = log_qz_obs - log_pz_obs
#         kld_obs = kld_obs.mean()      

#         # sparsity
#         B_sparsity = torch.norm(self.B, p=1)
        
#         # VAE training
#         loss = recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs + self.B_sparsity * B_sparsity
#         #########################   validation step  #########################
#         # Compute Mean Correlation Coefficient (MCC)
#         zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()

#         zt_true = batch["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
#         mcc = compute_mcc(zt_recon, zt_true, self.correlation)
#         # count accuracy of B predictions
#         # batch_B_tpr = []
#         # batch_B_fdr = []
#         # for i in range(batch_size):
#         #     graph_thres = 0.1
#         #     #B_est = postprocess(check_array(self.causal_state.batch_Bs[i]), graph_thres=graph_thres)
#         #     n = torch.nonzero(b[i]).size(0)
#         #     B_est = check_array(top_k_abs_tensor(self.causal_state.batch_Bs[i], n))
#         #     B_gt = check_array(b[i])
#         #     B_gt[B_gt < graph_thres] = 0
#         #     acc = count_accuracy(bin_mat(B_est), bin_mat(B_gt))
#         #     batch_B_tpr.append(acc['tpr'])
#         #     batch_B_fdr.append(acc['fdr'])
#         # B_avg_tpr = np.mean(batch_B_tpr)
#         # B_avg_fdr = np.mean(batch_B_fdr)
        
#         # self.log("val_B_avg_tpr", B_avg_tpr)
#         # self.log("val_B_avg_fdr", B_avg_fdr)
        
#         self.log("val_mcc", mcc) 
#         self.log("val_elbo_loss", loss)
#         self.log("val_recon_loss", recon_loss)
#         self.log("val_kld_normal", past_kld_dyn)
#         self.log("val_kld_dynamics", future_kld_dyn)
#         self.log("val_kld_observation", kld_obs)
#         self.log("val_B_sparsity", B_sparsity)

#         return loss
    
#     def sample(self, n=64):
#         with torch.no_grad():
#             e = torch.randn(n, self.z_dim, device=self.device)
#             eps, _ = self.spline.inverse(e)
#         return eps

#     def configure_optimizers(self):
#         opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
#         return [opt_v], []