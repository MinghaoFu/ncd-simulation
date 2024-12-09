"""model.py"""

from base64 import decode
import torch
import ipdb as pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from torch.func import jacfwd, vmap
import ipdb as pdb
from components.mlp import NLayerLeakyMLP

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

class BetaVAE_MLP(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, input_dim=3, output_dim=3, z_dim=10, hidden_dim=128, slope=0.2, encoder_n_layers=3, decoder_n_layers=1):
        super(BetaVAE_MLP, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim

        # encoder
        encoder = [
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(slope),
            nn.Linear(hidden_dim, 2*z_dim)
        ]
        for _ in range(encoder_n_layers): 
            encoder[-2:-2] = [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(slope)] 
        self.encoder = nn.Sequential(*encoder)

        # Fix the functional form to ground-truth mixing function
        decoder = [
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(slope),
            nn.Linear(hidden_dim, output_dim)
        ]
        for _ in range(decoder_n_layers): 
            decoder[-2:-2] = [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(slope)] 
        self.decoder = nn.Sequential(*decoder)

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
        # avoid scaling problem for structure learning
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