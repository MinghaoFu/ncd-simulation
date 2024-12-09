import torch.nn as nn
import torch
from components.mlp import NLayerLeakyMLP  
from torch.func import jacfwd, vmap

class NPObsPrior(nn.Module):

    def __init__(
        self, 
        latent_size, 
        input_dim,
        num_layers=3, 
        hidden_dim=64):
        super().__init__()
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))   
        self.input_dim = input_dim   
        self.latent_size = latent_size   
        gs = [NLayerLeakyMLP(in_features=latent_size+1, 
                             out_features=1, 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim) for i in range(input_dim)]
        
        self.gs = nn.ModuleList(gs)
    
    def forward(self, z, xs, masks=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, _ = z.shape
        xx, yy = xs.reshape(-1, 1, self.input_dim), z.reshape(-1, self.latent_size)
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(self.input_dim):
            if masks is None:
                inputs = torch.cat((yy, xx[:,:,i]),dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((yy*mask, xx[:,:,i]),dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                #pdd = jacobian(self.gs[i], inputs, create_graph=True, vectorize=True)
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
            # Determinant of low-triangular mat is product of diagonal entries
            #logabsdet = torch.log(torch.abs(torch.diag(pdd[:,0,:,-1])))
            logabsdet = torch.log(torch.abs(pdd[:,0,-1]))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, self.input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size))
        return residuals, sum_log_abs_det_jacobian