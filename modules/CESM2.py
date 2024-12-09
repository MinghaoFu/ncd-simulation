"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
torch.set_printoptions(profile="full")

import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
import torch.optim as optim
import wandb
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from .components.beta import BetaVAE_MLP, BetaVAE_MLP_independentnoise
from .components.transition import NPChangeTransitionPrior
from .components.mlp import MLPEncoder, MLPDecoder, Inference, NLayerLeakyMLP, TApproximator
from .components.flow import ComponentWiseCondSpline
from .metrics.correlation import compute_mcc
from Caulimate.Utils.Tools import check_array, check_tensor
from Caulimate.Utils.Lego import PartiallyPeriodicMLP
from einops import repeat

import ipdb as pdb

class CESM2ModularShiftsFixedB(pl.LightningModule):
    def __init__(
            self,
            input_dim,
            length,
            obs_dim,
            dyn_dim,
            lag,
            nclass,
            nclass_pos = 104,
            hidden_dim=128,
            dyn_embedding_dim=2,
            obs_embedding_dim=2,
            pos_embedding_dim=4,
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
            decoder_dist='gaussian', \
            obs_noise=False,
            correlation='Pearson',
            graph_thres=0.01,
            recon_coord=True,
            B_init=None,
            mask=None):
        '''Nonlinear ICA for general causal processes with modualar distribution shifts'''
        super().__init__()
        # Transition prior must be L (Linear), NP (Nonparametric)
        self.save_hyperparameters()
        assert trans_prior in ('L', 'NP')
        self.obs_dim = obs_dim
        self.dyn_dim = dyn_dim
        self.obs_embedding_dim = obs_embedding_dim
        self.dyn_embedding_dim = dyn_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
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
        self.recon_coord = recon_coord
        self.nclass_pos = nclass_pos
        # Domain embeddings (dynamics)
        self.dyn_embed_func = nn.Embedding(nclass, dyn_embedding_dim)
        self.obs_embed_func = nn.Embedding(nclass, obs_embedding_dim)
        self.pos_embed_func = PartiallyPeriodicMLP(2, 8, self.pos_embedding_dim, False, periodic_ratio=0)
        self.decode_coord = nn.Sequential(nn.LeakyReLU(0.2),
                                       nn.Linear(self.z_dim + self.pos_embedding_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, 2+1)
                                    )
        self.noise = obs_noise
        self.mask = check_tensor(mask) if mask is not None else check_tensor(torch.ones(input_dim, input_dim) - torch.eye(input_dim))

        # Flow for nonstationary regimes
        # self.flow = ComponentWiseCondSpline(input_dim=self.obs_dim,
        #                                     context_dim=obs_embedding_dim,
        #                                     bound=bound,
        #                                     count_bins=count_bins,
        #                                     order=order)
        # Factorized inference

        if self.noise:
            self.net = BetaVAE_MLP_independentnoise(input_dim=input_dim,
                                                    z_dim=self.z_dim,
                                                    hidden_dim=hidden_dim)
            self.xnoise = check_tensor(torch.tensor(0.1))
        else:
            self.net = BetaVAE_MLP(input_dim=input_dim,
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
                                                            embedding_dim=dyn_embedding_dim * 2, # with theta
                                                            num_layers=4,
                                                            hidden_dim=hidden_dim)

        # Initialize causal adjacency matrix in observed variables
        if B_init is not None:
            self.B = nn.Parameter(check_tensor(B_init))
        else:
            self.B = nn.Parameter(check_tensor(torch.randn(self.input_dim, self.input_dim)))

        self.B_net = PartiallyPeriodicMLP(1, 8, self.input_dim ** 2, 12)  # uncertain
        self.theta_net = PartiallyPeriodicMLP(dyn_embedding_dim + 1, 8, dyn_embedding_dim * 2, 12)
        # x = b(t)x + g(z) + e
        # zt = T(zt-1, theta(c))
        # for i in range(1, self.input_dim):
        #     self.B.data[i, i - 1] = 1
        # self.linearM = nn.Linear(self.input_dim, self.input_dim, bias=False)
        # torch.nn.init.xavier_uniform(self.linearM.weight)
        # self.xnoise = nn.Parameter(torch.tensor(0.7071))
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
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
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

    def heteroscedastic_neg_loglikelihood_loss(self, x, x_recon, distribution, noise, Bs):
        # TODO: implement heteroscedastic loss (need double check)
        # TODO: unfold length-dimension so that it can support time-varying B in the future
        x = x.to(torch.double)
        x_recon = x_recon.to(torch.double)
        noise = noise.to(torch.double)
        Bs = Bs.to(torch.double)

        batch_size, length, input_dim = x.shape
        Bs = torch.repeat_interleave(Bs, repeats=length, dim=0)
        Bs = Bs.view(batch_size, length, input_dim, input_dim)

        batch_size, length, input_dim = x.shape
        pi_term = torch.tensor(2 * torch.pi, device=x.device)
        eps = 1e-6  # Small constant for numerical stability


        '''I = torch.repeat_interleave(torch.eye(input_dim, device=x.device).unsqueeze(0), repeats=batch_size*length, dim=0)
        I_minus_Bs = I - Bs
        inv_I_minus_B_t = I_minus_Bs.inverse()
        mu = inv_I_minus_B_t, '''

        recon_loss = 0
        for b in range(batch_size):
            for t in range(length):
                #I = torch.eye(input_dim, device=x.device)
                B_t = Bs[b, t]  # [input_dim, input_dim]

                # Compute mu
                I_minus_B_t = torch.eye(input_dim, device=x.device) - B_t
                inv_I_minus_B_t = torch.inverse(I_minus_B_t + torch.eye(input_dim, device=x.device) * eps)

                inv_I_minus_B_t = inv_I_minus_B_t.clone()
                with torch.no_grad():
                    inv_I_minus_B_t.clamp_(min=eps)

                mu_t = torch.matmul(inv_I_minus_B_t, x_recon[b, t, :].unsqueeze(-1)).squeeze(-1)
                #mu_t = x_recon[b, t]
                # Compute Sigma
                # Assuming noise represents the diagonal elements of Sigma for simplicity
                Sigma_t = torch.matmul(torch.matmul(inv_I_minus_B_t, torch.diag(noise[b, t] + eps)),
                                       inv_I_minus_B_t.transpose(-2, -1)) + torch.eye(input_dim, device=x.device) * eps
                #print('hhh', torch.linalg.cond(Sigma_t))

                # Compute the negative log-likelihood for each data point
                x_diff = x[b, t] - mu_t

                #torch.linalg.inv(Sigma_t)
                mahalanobis_distance = torch.matmul(torch.matmul(x_diff.unsqueeze(0), torch.inverse(Sigma_t)), x_diff.unsqueeze(-1))
                log_det_Sigma_t = torch.log(max(torch.det(Sigma_t), torch.tensor(eps).to(x.device)))

                recon_loss += 0.5 * (log_det_Sigma_t + mahalanobis_distance + input_dim * torch.log(pi_term))

        recon_loss = recon_loss / length / batch_size
        return recon_loss


    def reconstruction_loss(self, x, x_recon, distribution, noise=None, Bs=None):
        x = x.to(torch.float32)
        x_recon = x_recon.to(torch.float32)
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

        elif distribution == 'heteroscedastic':
            recon_loss = self.heteroscedastic_neg_loglikelihood_loss(x, x_recon, distribution, noise, Bs)

        return recon_loss

    def smooth_loss(self, h, theta, eps = 1e-6):
        # Sort h and theta by h
        sorted_indices = torch.argsort(h, dim=0).squeeze()
        h_sorted = h[sorted_indices]
        theta_sorted = theta[sorted_indices]

        # Compute first derivatives of theta with respect to h
        # Use torch.diff to compute differences and divide by differences in h for non-uniform spacing
        h_diff = torch.diff(h_sorted.squeeze(), dim=0)
        theta_diff = torch.diff(theta_sorted, dim=0)

        # Avoid division by zero
        first_derivatives = theta_diff / (h_diff.unsqueeze(-1) + eps)

        # Compute the second derivatives (approximation)
        second_derivatives = torch.diff(first_derivatives, dim=0)

        # Smooth loss is the sum of squares of the second derivatives
        smooth_loss = (second_derivatives ** 2).sum()

        return smooth_loss

    def DAG_loss(self, B):
        if len(B.shape) == 2:
            B = B.unsqueeze(0)
        matrix_exp = torch.exp(B * B)
        traces = torch.sum(torch.diagonal(matrix_exp, dim1=-2, dim2=-1)) - B.shape[0] * B.shape[1]
        return traces

    def forward(self, batch):
        x, c, h, coords = batch['xt'], batch['ct'], batch['ht'], batch['st']
        if len(coords.shape) > 2:
            coords = coords[0]
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim) # [bs*len, 104]
        # x to (I-B)x
        x_flat = x_flat  # torch.matmul(self.M[None, :, :].repeat(x_flat.shape[0], 1, 1), x_flat.unsqueeze(2)).squeeze(2)
        Bs = self.generate_Bs(h)
        # Inference
        if self.noise:
            y_recon, x_noise, mus, logvars, zs = self.net(x_flat)
        else:
            y_recon, mus, logvars, zs = self.net(x_flat)

            # coord transformation
            embed_c = self.embed_coord
        # (I-B)x to x
        # x_recon = self.linearM(y_recon)
        x_recon = torch.bmm(y_recon.unsqueeze(1),
                            torch.repeat_interleave(torch.inverse(check_tensor(torch.eye(self.input_dim)) - Bs),
                                                    repeats=length, dim=0))
        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        return mus, logvars, x_recon, y_recon, zs, Bs, coords

    def generate_Bs(self, h):
        Bs = self.B + self.B_net(h.unsqueeze(1)).reshape(-1, self.input_dim, self.input_dim)  # torch.tril(self.B, diagonal=-1)
        Bs = Bs * self.mask
        return Bs
    
    def training_step(self, batch, batch_idx):
        x, c, h, s = batch['xt'], batch['ct'], batch['ht'], batch['st']
        # x: observation [bs, 3, 104]
        # h: time index [bs]
        # c: domain index -- month index [bs, 1]
        # s: coordinates, (batch, 2) # -> [lat, lon] [bs, 104, 2]
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, input_dim = x.shape
        x_flat = x.view(-1, self.input_dim)
        # x to (I-B)x
        x_flat = x_flat  # torch.matmul(self.M[None, :, :].repeat(x_flat.shape[0], 1, 1), x_flat.unsqueeze(2)).squeeze(2)
        # x = x_flat.reshape(-1, 3, self.input_dim)
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size, 1, self.obs_embedding_dim).repeat(1, length, 1)
        theta = self.theta_net(torch.cat([dyn_embeddings, h.unsqueeze(1)], dim=1))  # [bs,4]

        # Bs: (batch, input_dim, input_dim)
        Bs = self.generate_Bs(h)
        # Inference
        if self.noise:
            y_recon, x_noise, mus, logvars, zs = self.net(x_flat)
            x_noise = x_noise.view(batch_size, length, self.input_dim)
        else:
            y_recon, mus, logvars, zs = self.net(x_flat)

        x_recon = torch.bmm(y_recon.unsqueeze(1),
                            torch.repeat_interleave(torch.inverse(check_tensor(torch.eye(self.input_dim)) - Bs),
                                                    repeats=length, dim=0))

        y_recon = y_recon.unsqueeze(1).view(batch_size, length, self.input_dim)

        if self.recon_coord:
            # s repeat: [bs, 104, 2] -> [bs, 3, 104, 2]
            s_rpt = s.unsqueeze(1).repeat(1, length, 1, 1)
            # zs repeat: [bs*length, 7] -> [bs*length, 7, 104] -> [bs, 3, 7, 104] -> [bs, 3, 104, 7]
            zs_rpt = zs.unsqueeze(2).repeat(1, 1, input_dim).view(batch_size, length, -1, input_dim).permute(0, 1, 3, 2)
            pos_embeddings = self.pos_embed_func(s_rpt)
            coord_recon_input = torch.cat((pos_embeddings, zs_rpt), dim=3).view(batch_size*length*input_dim, -1)
            recon_xNcoord = self.decode_coord(coord_recon_input).view(batch_size*length, -1, input_dim) #

            recon_xNcoord = torch.bmm(recon_xNcoord,
                                torch.repeat_interleave(torch.inverse(check_tensor(torch.eye(self.input_dim)) - Bs), #[bs, 104, 104] -> [bs*3, 104, 104]
                                                        repeats=length, dim=0)).view(batch_size, length, input_dim, -1)
            xNcoord = torch.cat((x.unsqueeze(3), s_rpt), dim=-1) #[bs, 3, 104, 3]
            recon_loss = self.reconstruction_loss(xNcoord[:, :self.lag], recon_xNcoord[:, :self.lag], self.decoder_dist) + \
                         (self.reconstruction_loss(xNcoord[:, self.lag:], recon_xNcoord[:, self.lag:], self.decoder_dist)) / (
                                 length - self.lag)
        else:
            # (I-B)x to x
            # x_recon = self.linearM(y_recon)

            # VAE ELBO loss: recon_loss + kld_loss
            if self.decoder_dist == 'heteroscedastic':
                assert self.noise
                recon_loss = self.reconstruction_loss(x[:, :self.lag], y_recon[:, :self.lag], self.decoder_dist, noise=x_noise[:, :self.lag], Bs=Bs) + \
                             (self.reconstruction_loss(x[:, self.lag:], y_recon[:, self.lag:], self.decoder_dist, noise=x_noise[:, self.lag:], Bs=Bs)) / (
                                     length - self.lag)
            else:
                recon_loss = self.reconstruction_loss(x[:, :self.lag], x_recon[:, :self.lag], self.decoder_dist) + \
                             (self.reconstruction_loss(x[:, self.lag:], x_recon[:, self.lag:], self.decoder_dist)) / (
                                         length - self.lag)

        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)

        ### Dynamics parts ###
        # -> zt
        # -> con_zt = f(zt, zt-1) = Gaussian
        # Past KLD <=> N(0,1) #
        p_dist = D.Normal(torch.zeros_like(mus[:, :self.lag, :self.dyn_dim]),
                          torch.ones_like(logvars[:, :self.lag, :self.dyn_dim]))
        log_pz_past = torch.sum(torch.sum(p_dist.log_prob(zs[:, :self.lag, :self.dyn_dim]), dim=-1), dim=-1)
        log_qz_past = torch.sum(torch.sum(log_qz[:, :self.lag, :self.dyn_dim], dim=-1), dim=-1)
        past_kld_dyn = log_qz_past - log_pz_past
        past_kld_dyn = past_kld_dyn.mean()
        # Future KLD #
        log_qz_future = log_qz[:, self.lag:]
        residuals, logabsdet = self.transition_prior(zs[:, :, :self.dyn_dim], theta) #[bs,1,6] [bs]
        log_pz_future = torch.sum(self.dyn_base_dist.log_prob(residuals), dim=1) + logabsdet

        future_kld_dyn = (torch.sum(torch.sum(log_qz_future, dim=-1), dim=-1) - log_pz_future) / (length - self.lag)
        future_kld_dyn = future_kld_dyn.mean()

        ### Observation parts ###
        p_dist_obs = D.Normal(obs_embeddings[:, :, 0].reshape(batch_size, length, 1),
                              torch.exp(obs_embeddings[:, :, 1].reshape(batch_size, length, 1) / 2))
        log_pz_obs = torch.sum(torch.sum(p_dist_obs.log_prob(zs[:, :, self.dyn_dim:]), dim=1), dim=-1)
        log_qz_obs = torch.sum(torch.sum(log_qz[:, :self.lag, self.dyn_dim:], dim=-1), dim=-1)
        kld_obs = log_qz_obs - log_pz_obs
        kld_obs = kld_obs.mean()

        # sparsity
        # M_sparsity = 0
        # for param in self.linearM.parameters():
        #     M_sparsity += torch.norm(param, 1)  # L1 norm
        # W_sparsity = torch.norm(torch.inverse(self.linearM.weight), 1)
        B_sparsity = torch.norm(Bs, 1)
        B_DAG_loss = self.DAG_loss(self.B)
        # VAE training
        loss = 0.1*recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs
        # if B_sparsity > 1:
        #     loss += self.B_sparsity * B_sparsity
        loss += self.B_sparsity * B_sparsity
        loss += 1e-1 * B_DAG_loss

        smooth_loss = self.smooth_loss(h, theta)
        loss += 1e-4 * smooth_loss

        #########################   training step  #########################
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", past_kld_dyn)
        self.log("train_kld_dynamics", future_kld_dyn)
        self.log("train_kld_observation", kld_obs)
        self.log("train_B_sparsity", B_sparsity)
        self.log("train_B_DAG", B_DAG_loss)
        self.log("train_theta_smooth", smooth_loss)


        return loss

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % 10 != 0:
            return
        x, c, coords = batch['xt'], batch['ct'], batch['st']
        if len(coords.shape) > 2:
            coords = coords[0]
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        # x to (I-B)x
        x_flat = x_flat  # torch.matmul(self.M[None, :, :].repeat(x_flat.shape[0], 1, 1), x_flat.unsqueeze(2)).squeeze(2)
        # x = x_flat.reshape(-1, 3, self.input_dim)
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size, 1, self.obs_embedding_dim).repeat(1, length, 1)

        Bs = self.B + self.B_net(h.unsqueeze(1)).reshape(-1, self.input_dim,
                                                         self.input_dim)  # torch.tril(self.B, diagonal=-1)
        # Inference
        y_recon, x_noise, mus, logvars, zs = self.net(x_flat)

        # (I-B)x to x
        # x_recon = self.linearM(y_recon)
        x_recon = torch.bmm(y_recon.unsqueeze(1),
                            torch.repeat_interleave(torch.inverse(check_tensor(torch.eye(self.input_dim)) - Bs),
                                                    repeats=length, dim=0))

        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        B_est = check_array(self.B)  # np.eye(len(W_est_postrocess)) - W_est_postrocess

        # M_post = postprocess(M_est, graph_thres=self.graph_thres)
        DAG_fig = plot_solutions([B_est], ['B_est'], add_value=False)
        plt.cla()

        self.logger.experiment.log({"DAG": [wandb.Image(DAG_fig)]})

        extent = [torch.min(coords[:, 0]).item(), torch.min(coords[:, 1]).item(), torch.max(coords[:, 0]).item(),
                  torch.max(coords[:, 1]).item()]
        map_fig = plot_DAG_on_map(B_est, coords, extent, threshold=self.graph_thres)
        map_fig.savefig('./DAG_on_map.png')
        self.logger.experiment.log({"DAG_on_map": [wandb.Image(map_fig)]})
        return Bs, mus, logvars

    def test_step(self, batch, batch_idx):
        x, c, h, s = batch['xt'], batch['ct'], batch['ht'], batch['st']
        # x: observation [bs, 3, 104]
        # h: time index [bs]
        # c: domain index -- month index [bs, 1]
        # s: coordinates, (batch, 2) # -> [lat, lon] [bs, 104, 2]
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, input_dim = x.shape
        x_flat = x.view(-1, self.input_dim)
        # x to (I-B)x
        x_flat = x_flat  # torch.matmul(self.M[None, :, :].repeat(x_flat.shape[0], 1, 1), x_flat.unsqueeze(2)).squeeze(2)
        # x = x_flat.reshape(-1, 3, self.input_dim)
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size, 1, self.obs_embedding_dim).repeat(1, length, 1)
        theta = self.theta_net(torch.cat([dyn_embeddings, h.unsqueeze(1)], dim=1))  # [bs,4]

        # Bs: (batch, input_dim, input_dim)
        Bs = self.generate_Bs(h)
        # Inference
        if self.noise:
            y_recon, x_noise, mus, logvars, zs = self.net(x_flat)
            x_noise = x_noise.view(batch_size, length, self.input_dim)
        else:
            y_recon, mus, logvars, zs = self.net(x_flat)

        x_recon = torch.bmm(y_recon.unsqueeze(1),
                            torch.repeat_interleave(torch.inverse(check_tensor(torch.eye(self.input_dim)) - Bs),
                                                    repeats=length, dim=0))

        y_recon = y_recon.unsqueeze(1).view(batch_size, length, self.input_dim)
    
    def sample(self, n=64):
        with torch.no_grad():
            e = torch.randn(n, self.z_dim, device=self.device)
            eps, _ = self.spline.inverse(e)
        return eps

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999),
                                  weight_decay=0.0001)
        return [opt_v], []