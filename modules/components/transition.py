"""Prior Network"""
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from torch.func import jacfwd, vmap
from .mlp import NLayerLeakyMLP, NLayerLeakyNAC
from .base import GroupLinearLayer
import ipdb as pdb

# class MBDTransitionPrior(nn.Module):

#     def __init__(self, lags, latent_size, bias=False):
#         super().__init__()
#         # self.init_hiddens = nn.Parameter(0.001 * torch.randn(lags, latent_size))    
#         # out[:,:,0] = (x[:,:,0]@conv.weight[:,:,0].T)+(x[:,:,1]@conv.weight[:,:,1].T) 
#         # out[:,:,1] = (x[:,:,1]@conv.weight[:,:,0].T)+(x[:,:,2]@conv.weight[:,:,1].T)
#         self.L = lags      
#         self.transition = GroupLinearLayer(din = latent_size, 
#                                            dout = latent_size, 
#                                            num_blocks = lags,
#                                            diagonal = False)
#         self.bias = bias
#         if bias:
#             self.b = nn.Parameter(0.001 * torch.randn(1, latent_size))
    
#     def forward(self, x, mask=None):
#         # x: [BS, T, D] -> [BS, T-L, L+1, D]
#         batch_size, length, input_dim = x.shape
#         # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
#         # x = torch.cat((init_hiddens, x), dim=1)
#         x = x.unfold(dimension = 1, size = self.L+1, step = 1)
#         x = torch.swapaxes(x, 2, 3)
#         shape = x.shape

#         x = x.reshape(-1, self.L+1, input_dim)
#         xx, yy = x[:,-1:], x[:,:-1]
#         if self.bias:
#             residuals = torch.sum(self.transition(yy), dim=1) + self.b - xx.squeeze()
#         else:
#             residuals = torch.sum(self.transition(yy), dim=1) - xx.squeeze()
#         residuals = residuals.reshape(batch_size, -1, input_dim)
#         # Dummy jacobian matrix (0) to represent identity mapping
#         log_abs_det_jacobian = torch.zeros(batch_size, device=x.device)
#         return residuals, log_abs_det_jacobian

class NPStatePrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size, 
        input_dim,
        num_layers=3, 
        hidden_dim=64):
        super().__init__()
        self.L = lags
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
        batch_size, length, _ = z.shape
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
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length), dim=1)
        return residuals, sum_log_abs_det_jacobian
    
class NPTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size, 
        num_layers=3, 
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=lags*latent_size+1, 
                             out_features=1, 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
    
    def forward(self, x, masks=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
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
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian

class NPChangeTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size,
        embedding_dim, 
        num_layers=3,
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=hidden_dim+lags*latent_size+1, 
                             out_features=1, 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
        self.fc = NLayerLeakyMLP(in_features=embedding_dim,
                                 out_features=hidden_dim,
                                 num_layers=2,
                                 hidden_dim=hidden_dim)

    def forward(self, x, embeddings, t=None, masks=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        # embeddings: [BS, embed_dims]
        batch_size, length, input_dim = x.shape
        embeddings = self.fc(embeddings)
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            if masks is None:
                inputs = torch.cat((embeddings, yy, xx[:,:,i]),dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((embeddings, yy*mask, xx[:,:,i]),dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                # pdd = jacobian(self.gs[i], inputs, create_graph=True, vectorize=True)
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
            # Determinant of low-triangular mat is product of diagonal entries
            logabsdet = torch.log(torch.abs(pdd[:,0,-1]))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian


class NPMaskChangeTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size,
        embedding_dim, 
        num_layers=3,
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=hidden_dim+lags*latent_size+1, 
                             out_features=1, 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
        self.fc = NLayerLeakyMLP(in_features=embedding_dim,
                                 out_features=hidden_dim,
                                 num_layers=2,
                                 hidden_dim=hidden_dim)

        self.nn = NLayerLeakyMLP(in_features=latent_size * 2,
                                 out_features=latent_size,
                                 num_layers=2,
                                 hidden_dim=hidden_dim)

    def forward(self, x, embeddings, t=None, masks=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        # embeddings: [BS, embed_dims]
        batch_size, length, input_dim = x.shape
        embeddings = self.fc(embeddings)
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        logits = nn.Parameter(torch.zeros(input_dim, input_dim))
        ins_mask = self.sample_mask(logits, tau=0.3)

        x = self.nn(torch.cat([x, x * mask], dim=-1))

        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            if masks is None:
                inputs = torch.cat((embeddings, yy, xx[:,:,i]),dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((embeddings, yy*mask, xx[:,:,i]),dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                # pdd = jacobian(self.gs[i], inputs, create_graph=True, vectorize=True)
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
            # Determinant of low-triangular mat is product of diagonal entries
            logabsdet = torch.log(torch.abs(pdd[:,0,-1]))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian
    
    def sample_mask(self, logits, tau=0.3):
        num_vars = len(logits)
        mask = self.gumbel_sigmoid(logits, tau=tau)
        non_diagonal_mask = torch.ones(num_vars, num_vars) - torch.eye(num_vars)
        # Set diagonal entries to 0
        mask = mask * non_diagonal_mask
        return mask

    def sample_logistic(self, shape, out=None):
        U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
        return torch.log(U) - torch.log(1-U)

    def gumbel_sigmoid(self, logits, tau=1):
        dims = logits.dim()
        logistic_noise = self.sample_logistic(logits.size(), out=logits.data.new())
        y = logits + logistic_noise
        return torch.sigmoid(y / tau)

class NPChangeInstantaneousTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size,
        embedding_dim, 
        num_layers=3,
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=hidden_dim+lags*latent_size+1+i, 
                             out_features=1, 
                             num_layers=0, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
        self.fc = NLayerLeakyMLP(in_features=embedding_dim,
                                 out_features=hidden_dim,
                                 num_layers=2,
                                 hidden_dim=hidden_dim)
        self.u = nn.Parameter(torch.zeros(latent_size, latent_size))

    def forward(self, x, embeddings, alphas):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        # embeddings: [BS, embed_dims]
        batch_size, length, input_dim = x.shape
        embeddings = self.fc(embeddings)
        embeddings = embeddings.unsqueeze(1).repeat(1, length-self.L, 1).reshape(-1, embeddings.shape[-1])
        # prepare data
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)

        
        # get residuals and |J|
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            inputs = torch.cat([xx[:,:,j] * alphas[i][j] for j in range(i)] + [embeddings, yy, xx[:,:,i]], dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
            # Determinant: product of diagonal entries, sum of last entry
            logabsdet = torch.log(torch.abs(pdd[:,0,-1]))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian
    

 
class NPInstantaneousTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size, 
        num_layers=3, 
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=lags*latent_size+1+i, 
                             out_features=1, 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
    
    def forward(self, x):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # prepare data
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        # MSCL
        mask = self.sample_mask(self.u, tau=0.3)
        # get residuals and |J|
        residuals = [ ]

        hist_jac = []
        
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            # mask
            yy[:,-self.input_dim:] =  yy[:,-self.input_dim:] * mask[i]

            inputs = torch.cat([yy] + [xx[:,:,j] for j in range(i)] + [xx[:,:,i]], dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
            # Determinant: product of diagonal entries, sum of last entry
            logabsdet = torch.log(torch.abs(pdd[:,0,-1]))

            # hist_jac.append(torch.unsqueeze(pdd[:,0,:self.L*input_dim], dim=1))
            hist_jac.append(torch.unsqueeze(pdd[:,0,:-1], dim=1))
            # hist_jac.append(torch.unsqueeze(torch.abs(pdd[:,0,:self.L*input_dim]), dim=1))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        # hist_jac = torch.cat(hist_jac, dim=1) # BS * input_dim * (L * input_dim)
        # hist_jac = torch.mean(hist_jac, dim=0) # input_dim * (L * input_dim)
        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian, hist_jac
    
    def inference_mask(self, tau=0.3):
        num_vars = len(self.u)
        mask = torch.sigmoid(self.u / tau)
        non_diagonal_mask = torch.ones(num_vars, num_vars) - torch.eye(num_vars)
        # Set diagonal entries to 0
        mask = mask * non_diagonal_mask
        return mask
    
    def sample_mask(self, logits, tau=0.3):
        num_vars = len(logits)
        mask = self.gumbel_sigmoid(logits, tau=tau)
        non_diagonal_mask = torch.ones(num_vars, num_vars) - torch.eye(num_vars)
        # Set diagonal entries to 0
        mask = mask * non_diagonal_mask
        return mask

    def sample_logistic(self, shape, out=None):
        U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
        return torch.log(U) - torch.log(1-U)

    def gumbel_sigmoid(self, logits, tau=0.3):
        dims = logits.dim()
        logistic_noise = self.sample_logistic(logits.size(), out=logits.data.new())
        y = logits + logistic_noise
        return torch.sigmoid(y / tau)
