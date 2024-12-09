"""model.py"""

import torch
import ipdb as pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacfwd, vmap
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from .keypoint import SpatialSoftmax
from .mlp import NLayerLeakyMLP, NLayerLeakyNAC
import ipdb as pdb

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class BetaVAE_CNN(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3, hidden_dim=256):
        super(BetaVAE_CNN, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 64, 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, hidden_dim, 4, 1),            # B, hidden_dim,  1,  1
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            View((-1, hidden_dim*1*1)),                 # B, hidden_dim
            nn.Linear(hidden_dim, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),               # B, hidden_dim
            View((-1, hidden_dim, 1, 1)),               # B, hidden_dim,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 64, 4),      # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 64, 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 128, 128
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

class BetaVAE_Physics(nn.Module):
    """Visual Encoder/Decoder for Ball dataset."""
    def __init__(self, z_dim=10, nc=3, nf=16, norm_layer='Batch', hidden_dim=512):
        super(BetaVAE_Physics, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        k = z_dim//2
        self.k = k
        height = 64
        width = 64
        lim=[-5., 5., -5., 5.]
        self.height = height
        self.width = width
        self.lim = lim
        x = np.linspace(lim[0], lim[1], width // 4)
        y = np.linspace(lim[2], lim[3], height // 4)
        z = np.linspace(-1., 1., k)
        self.register_buffer('x', torch.FloatTensor(x))
        self.register_buffer('y', torch.FloatTensor(y))
        self.register_buffer('z', torch.FloatTensor(z))

        self.integrater = SpatialSoftmax(height=height//4, width=width//4, channel=k, lim=lim)
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nf, 7, 1, 3),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 4) x 16 x 16
            nn.Conv2d(nf * 4, k, 1, 1)
        )

        self.fc = nn.Sequential(View((-1, k * 16 * 16)),
                                nn.Linear(k * 16 * 16, z_dim)
                                )

        self.decoder = nn.Sequential(            
            nn.ConvTranspose2d(self.k, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.ConvTranspose2d(nf * 2, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf * 2, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf, 3, 7, 1, 3))

        # self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def keypoint_to_heatmap(self, keypoint, inv_std=10.):
        # keypoint: B x n_kp x 2
        # heatpmap: B x n_kp x (H / 4) x (W / 4)
        # ret: B x n_kp x (H / 4) x (W / 4)
        height = self.height // 4
        width = self.width // 4

        mu_x, mu_y = keypoint[:, :, :1].unsqueeze(-1), keypoint[:, :, 1:].unsqueeze(-1)
        y = self.y.view(1, 1, height, 1)
        x = self.x.view(1, 1, 1, width)

        g_y = (y - mu_y)**2
        g_x = (x - mu_x)**2
        dist = (g_y + g_x) * inv_std**2

        hmap = torch.exp(-dist)

        return hmap

    def _encode(self, x):
        heatmap = self.encoder(x)
        batch_size = heatmap.shape[0]
        mu = self.integrater(heatmap)
        mu = mu.view(batch_size, -1)
        logvar = self.fc(heatmap)
        return torch.cat((mu,logvar), dim=-1)

    def _decode(self, z):
        kpts = z.view(-1, self.k, 2)
        hmap = self.keypoint_to_heatmap(kpts)
        return self.decoder(hmap)

class BetaVAE_MLP(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, input_dim=3, z_dim=10, hidden_dim=128):
        super(BetaVAE_MLP, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
                                       nn.Linear(input_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, 2*z_dim)
                                    )
        # Fix the functional form to ground-truth mixing function
        self.decoder = nn.Sequential(  nn.LeakyReLU(0.2),
                                       nn.Linear(z_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, input_dim)
                                    )


        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):

        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

class BetaTVAE_MLP(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, input_dim=3, z_dim=10, t_embed_dim=2, hidden_dim=128):
        super(BetaTVAE_MLP, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.t_embed_dim = t_embed_dim
        self.encoder = nn.Sequential(
                                       nn.Linear(input_dim + t_embed_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, 2*z_dim)
                                    )
        # # Fix the functional form to ground-truth mixing function
        # self.decoder_s = nn.Sequential(  nn.LeakyReLU(0.2),
        #                                nn.Linear(z_dim, hidden_dim),
        #                                nn.LeakyReLU(0.2),
        #                                nn.Linear(hidden_dim, hidden_dim),
        #                                nn.LeakyReLU(0.2),
        #                                nn.Linear(hidden_dim, input_dim)
        #                             )
        
        self.decoder = nn.Sequential(  nn.LeakyReLU(0.2),
                                       nn.Linear(z_dim + t_embed_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, input_dim)
                                    )
        

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, t_embedding, return_z=True):
        # indices = torch.arange(b.shape[-1])
        # b[:, indices, indices] += 1
        xt = torch.cat([x, t_embedding], axis=-1).view(-1, self.input_dim + self.t_embed_dim)
        distributions = self._encode(xt)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]

        z = reparametrize(mu, logvar)
        zt = torch.cat([z, t_embedding.view(-1, self.t_embed_dim)], axis=-1)
        x_recon = self._decode(zt)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
    
class BetaStateVAE_MLP(nn.Module):
    def __init__(self, input_dim=3, z_dim=10, hidden_dim=128, num_layers=3, masks=None):
        super(BetaStateVAE_MLP, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
                                       nn.Linear(input_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, 2*z_dim)
                                    )
        # Fix the functional form to ground-truth mixing function
        self.decoder = nn.Sequential(  
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(z_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, input_dim)
                                    )
        
        self.masks = masks
        if masks is not None:
            gs = [NLayerLeakyMLP(in_features=z_dim, 
                                out_features=1, 
                                num_layers=num_layers, 
                                hidden_dim=hidden_dim,
                                mask = masks[i]) for i in range(input_dim)]
        else:
            gs = [NLayerLeakyMLP(in_features=z_dim, 
                                out_features=1, 
                                num_layers=num_layers, 
                                hidden_dim=hidden_dim,
                                mask = None) for i in range(input_dim)]
        self.gs = nn.ModuleList(gs)
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True, standardize=False):

        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        # avoid scaling problem
        if standardize:
            div = 0 #1e-6
            mu_mean = mu.mean(dim=-1, keepdim=True)
            mu_std = mu.std(dim=-1, keepdim=True)
            mu = (mu - mu_mean) / (mu_std + div)
            logvar_mean = logvar.mean(dim=-1, keepdim=True)
            logvar_std = logvar.std(dim=-1, keepdim=True)
            logvar = (logvar - logvar_mean) / (logvar_std + div)

        z = reparametrize(mu, logvar)
        x_recon, jac_mat = self._decode(z, self.masks)

        if return_z:
            return jac_mat, x_recon, mu, logvar, z
        else:
            return jac_mat, x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z, masks=None): # z is x state, mixing matrix mask
        
        _, input_dim = z.shape    
        x_recon = []
        jac_mat = []
        for i in range(input_dim):
            if masks is None:
                inputs = z
            else:
                mask = masks[i]
                inputs = z*mask
            xi_recon = self.gs[i](inputs)
            with torch.enable_grad():
                # pdd = jacobian(self.gs[i], inputs, create_graph=True, vectorize=True)
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
            # logabsdet = torch.log(torch.abs(pdd[:,0,-1]))
            # sum_log_abs_det_jacobian += logabsdet
            x_recon.append(xi_recon)
            jac_mat.append(pdd)

        x_recon = torch.cat(x_recon, dim=-1)
        jac_mat = torch.cat(jac_mat, dim=-2)
        jac_mat = jac_mat #* torch.ones((input_dim, input_dim)).fill_diagonal_(0).to(jac_mat.device)
        # jac_mat_{i,j} != 0 imples x_j -> x_i, consistent with B in other codes
        return x_recon, jac_mat
    
class BetaVAE_MLP_independentnoise(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, input_dim=3, z_dim=10, hidden_dim=128):
        super(BetaVAE_MLP_independentnoise, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
                                       nn.Linear(input_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, 2*z_dim + input_dim)
                                    )
        # Fix the functional form to ground-truth mixing function
        self.decoder = nn.Sequential(  nn.LeakyReLU(0.2),
                                       nn.Linear(z_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, input_dim)
                                    )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def treatment_effect(self, z):
        x_recon = self._decode(z)
        effects = []
        for i in range(z.shape[0]):
            zi = z.clone()
            zi[i] = 0
            zi_effect = x_recon - self._decode(zi)
            effects.append(zi_effect)
            
        return effects
    
    def forward(self, x, return_z=True):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:2*self.z_dim]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)
        x_noise = distributions[:, 2*self.z_dim:]

        if return_z:
            return x_recon, x_noise, mu, logvar, z
        else:
            return x_recon, x_noise, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
    

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()