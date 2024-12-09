
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl

from torch.nn.utils import spectral_norm
from Caulimate.Utils.Tools import check_tensor
from Caulimate.Utils.GraphUtils import eudistance_mask
from Caulimate.Utils.Lego import CustomMLP, PartiallyPeriodicMLP

class B_net(nn.Module):
    def __init__(self, m_embed_dim, hid_dim, output_dim, cos_len, input_dim=1, periodic_ratio=0.2):
        super(B_net, self).__init__()
        self.m_embedding = nn.Embedding(12, m_embed_dim)
        self.m_encoder = CustomMLP(m_embed_dim, hid_dim, output_dim, 3)
        
        self.t_encoder = PartiallyPeriodicMLP(input_dim, hid_dim, output_dim, cos_len, periodic_ratio)
        
        self.fc = CustomMLP(2*hid_dim, hid_dim, output_dim, 3)
        # init.uniform_(self.fc1.bias, -5000, 5000)

    def forward(self, t):
        m = (t % 12).to(torch.int64).squeeze(dim=1)
        m_embed = self.m_embedding(m)
        t_embed = t
        xm = self.m_encoder(m_embed)
        xt = self.t_encoder(t_embed)
        
        return xt + xm 


class TApproximator(nn.Module):
    def __init__(self, args, m_embed_dim, hid_dim, cos_len, periodic_ratio=0.2):
        super(TApproximator, self).__init__()
        self.m_embedding = nn.Embedding(12, m_embed_dim)
        self.m_encoder = CustomMLP(m_embed_dim, hid_dim, 1, 3)
        
        self.t_encoder = PartiallyPeriodicMLP(1, hid_dim, 1, cos_len, periodic_ratio)
        
        self.fc = CustomMLP(2*hid_dim, hid_dim, 1, 3)
        # init.uniform_(self.fc1.bias, -5000, 5000)

    def forward(self, t):
        m = (t % 12).to(torch.int64).squeeze(dim=1)
        m_embed = self.m_embedding(m)
        t_embed = t
        xm = self.m_encoder(m_embed)
        xt = self.t_encoder(t_embed)
        
        return xt + xm 

class GolemModel(pl.LightningModule):
    def __init__(self, 
                 args, 
                 d, 
                 coords,
                 in_dim=1, 
                 equal_variances:bool=True,
                 seed=1, 
                 mask=None,
                 fast:bool=False):
        super().__init__()
        self.save_hyperparameters()
        self.loss = args.loss
        self.d = d
        self.seed = seed
        self.batch_size = args.batch_size
        self.equal_variances = equal_variances
        if mask is not None:
            self.mask = check_tensor(mask)
        else:
            self.mask = check_tensor(torch.ones(d, d) - torch.eye(d))
        self.fast = fast
        
        self.in_dim = in_dim
        self.num = args.num
        self.coords = check_tensor(coords)

        self.tol = args.tol
        self.B_lags = []
        
        self.gradient = []
        
        if self.fast: # real-world data
            self.B = nn.Parameter(check_tensor(torch.randn(self.d, self.d)))
            self.B_net = PartiallyPeriodicMLP(1, 8, self.d ** 2, 12)  # uncertain
            #self.B_net = B_net(2, 8, self.mask.sum().item(), args.cos_len, input_dim=1, periodic_ratio=0.1)
        else:
            self.TApproximators = nn.ModuleList()
            for _ in range(self.mask.sum().item()):
                self.TApproximators.append(TApproximator(args, 2, 32, args.cos_len, periodic_ratio=0.1))

        

    def decompose_t_batch(self, t_batch):
        a_batch = t_batch // 100
        b_batch = t_batch % 100
        return a_batch, b_batch
    
    def apply_spectral_norm(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                spectral_norm(module)
        
    def generate_B(self, T):
        if self.fast:
            batch_size = T.shape[0]
            B_flat = self.B_net(T)
            B = check_tensor(torch.zeros(batch_size, self.d, self.d))
            indices = torch.nonzero(self.mask, as_tuple=False)
            for i in range(B_flat.shape[0]):
                for j, (row, col) in enumerate(indices):
                    B[i, row, col] = B_flat[i, j]
        else:
            #T_embed = (1 / self.alpha) * torch.cos(self.beta * T + self.bias)
            T_embed = T
            _B = []
            for layer in self.TApproximators:
                B_i = layer(T)
                B_i_sparse = B_i.masked_fill_(torch.abs(B_i) < self.tol, 0)
                _B.append(B_i_sparse)
            _B = torch.cat(_B, dim=1)
            B = check_tensor(torch.zeros((_B.shape[0], self.d, self.d)))
            row_ids, col_ids = torch.where(self.mask == 1)
            B[:, row_ids, col_ids] = _B
        
        return B
        
    def _preprocess(self, B):
        B = B.clone()
        B_shape = B.shape
        if len(B_shape) == 3:  # Check if B is a batch of matrices
            for i in range(B_shape[0]):  # Iterate over each matrix in the batch
                B[i].fill_diagonal_(0)
        else:
            print("Input tensor is not a batch of matrices.")
            B.data.fill_diagonal_(0)
        return B
        
    def forward(self, X, T):
        #B = self.B + self.B_net(T).reshape(-1, self.d, self.d)  # torch.tril(self.B, diagonal=-1)
        B = self.generate_B(T)
        
        batch_size = X.shape[0]
        losses = {}
        total_loss = 0
        X = X - X.mean(axis=0, keepdim=True)
        likelihood = torch.sum(self._compute_likelihood(X, B)) / batch_size
        
        for l in self.loss.keys():
            if l == 'L1':
                #  + torch.sum(self._compute_L1_group_penalty(B))
                losses[l] = self.loss[l] * (torch.sum(self._compute_L1_penalty(B))) / batch_size
                total_loss += losses[l]
            elif l == 'dag':
                losses[l] = self.loss[l] * self._compute_h(B)
                total_loss += losses[l]
            elif l == 'grad':
                losses[l] = self.loss[l] * torch.sum(self._compute_gradient_penalty(B, T)) / batch_size
                total_loss += losses[l]
            elif l == 'flat':
                losses[l] = self.loss[l] * torch.sum(torch.pow(B[:, 1:] - B[:, :-1], 2)) / batch_size
                total_loss += losses[l]
        
        losses['likelihood'] = likelihood
        losses['total_loss'] = total_loss + likelihood
        #self.gradient.append(self._compute_gradient_penalty(losses['total_loss']).cpu().detach().item())

        return losses
        
    def _compute_likelihood(self, X, B):
        X = X.unsqueeze(1)
        if self.equal_variances:
            return 0.5 * self.d * torch.log(
                torch.square(
                    torch.linalg.norm(X - torch.bmm(X, B))
                )
            ) - torch.linalg.slogdet(check_tensor(torch.eye(self.d)) - B)[1]
        else:
            return 0.5 * torch.sum(
                torch.log(
                    torch.sum(
                        torch.square(X - X @ B), dim=0
                    )
                )
            ) - torch.linalg.slogdet(check_tensor(torch.eye(self.d)) - B)[1]

    def _compute_L1_penalty(self, B):
        return torch.norm(B, p=1, dim=(-2, -1)) 
   
    def _compute_L1_group_penalty(self, B):
        return torch.norm(B, p=2, dim=(0))

    def _compute_h(self, B):
        # B: (batch_size, d, d)
        matrix_exp = torch.linalg.matrix_exp(B * B)# torch.exp(B * B)
        traces = torch.sum(torch.diagonal(matrix_exp, dim1=-2, dim2=-1), dim=-1) - B.shape[1]
        return torch.sum(traces) / B.shape[0]

    def _compute_smooth_penalty(self,B_t):
        B = B_t.clone().data
        batch_size = B.shape[0]
        for i in range(batch_size):
            b_fft = torch.fft.fft2(B[i])
            b_fftshift = torch.fft.fftshift(b_fft)
            center_idx = b_fftshift.shape[0] // 2
            b_fftshift[center_idx, center_idx] = 0.0
            b_ifft = torch.fft.ifft2(torch.fft.ifftshift(b_fftshift))
            B[i] = b_ifft
            
        return torch.norm(B, p=1, dim=(-2, -1))
    
    def _compute_gradient_penalty(self, loss):
        gradients = torch.autograd.grad(outputs=loss, inputs=self.linear1.parameters(), retain_graph=True)
        gradient_norm1 = torch.sqrt(sum((grad**2).sum() for grad in gradients))
        
        return gradient_norm1