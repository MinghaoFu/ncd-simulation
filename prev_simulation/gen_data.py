import numpy as np
import os

import networkx as nx
from scipy.stats import ortho_group
import math

from Caulimate.Data.SimDAG import simulate_time_vary_weight, simulate_random_dag
from Caulimate.Utils.Tools import bin_mat, seed_everything

def randomly_set_zeros_to_ones(matrix, num_elements_to_set):
    # Get the shape of the matrix
    rows, cols = matrix.shape
    
    # Find the indices of zero elements not on the diagonal
    zero_indices = [(i, j) for i in range(rows) for j in range(cols) if matrix[i, j] == 0 and i != j]
    
    # Randomly select indices to set to 1
    selected_indices = np.random.choice(len(zero_indices), size=num_elements_to_set, replace=False)
    
    # Set the selected elements to 1
    for index in selected_indices:
        i, j = zero_indices[index]
        matrix[i, j] = 1
    
    return matrix

def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope

def leaky_ReLU(D, negSlope):
    assert negSlope > 0
    leaky1d = np.vectorize(leaky_ReLU_1d)
    return leaky1d(D, negSlope)

def generateUniformMat(Ncomp, condT):
    """
    generate a random matrix by sampling each element uniformly at random
    check condition number versus a condition threshold
    """
    A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
    for i in range(Ncomp):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    while np.linalg.cond(A) > condT:
        # generate a new A matrix!
        A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    return A

def generate_random_mixing_list(input_dim, output_dim, Nlayer):
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        if input_dim == 1:
            A = np.random.uniform(-1, 1, (input_dim, input_dim))
        else:
            A = ortho_group.rvs(input_dim)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)
        
    if input_dim == 1:
        A = np.random.uniform(-1, 1, (input_dim, input_dim))
    else:
        A = ortho_group.rvs(input_dim)
    A = A[:, :output_dim]
    mixingList.append(A)

    return mixingList

def nonparam_gen(save_dir='./data', 
                    NClass=4, 
                    lags=2, 
                    Nlayer=2, 
                    length=1, 
                    condList=[], 
                    negSlope=0.2, 
                    dyn_latent_size=2, # zs_dim
                    obs_latent_size=2, # zc_dim
                    observed_size=2, 
                    transitions=[], 
                    noise_scale=0.1, 
                    batch_size=40000, 
                    Niter4condThresh=1e4, 
                    varyMean=True, 
                    seed=42):
    dataset_name = "nonparam_gen"
    path = os.path.join(save_dir, dataset_name)
    print('--- Save to {}'.format(path))
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        A = np.random.uniform(1, 2, (dyn_latent_size, dyn_latent_size))  # - 1
        for i in range(dyn_latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(dyn_latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []

    for l in range(Nlayer - 1):
        # generate causal matrix first:
        if latent_size == 1:
            A = np.random.uniform(-1, 1, (dyn_latent_size, dyn_latent_size))
        else:
            A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)
    A = ortho_group.rvs(observed_size)
    A = A[:dyn_latent_size, :]
    mixingList.append(A)

    # Domain-varying edges
    edge_pairs = [(1,2), (3,4)]
    if dyn_latent_size < 5:
        edge_pairs = [(0,1)]
    edge_weights = np.random.uniform(-1.25, 1.25,(NClass, len(edge_pairs)))

    # get modulation parameters
    varMat = np.random.uniform(0.01, 1, (NClass, obs_latent_size))
    if varyMean:
        meanMat = np.random.uniform(-0.5, 0.5, (NClass, obs_latent_size))
    else:
        meanMat = np.zeros((NClass, obs_latent_size))

    zt = []; xt = []; ct = []; 
    zt_ns = []; xt_ns = []; ct_ns = []; 


    for j in range(NClass):
        ct.append(j * np.ones(batch_size))
        z_l = np.random.normal(0, 1, (batch_size, lags, dyn_latent_size))
        z_l = (z_l - np.mean(z_l, axis=0 ,keepdims=True)) / np.std(z_l, axis=0 ,keepdims=True)
        # Change observation
        z_o = np.random.normal(0, 1, (batch_size, lags+length, obs_latent_size))
        z_o = (z_o - np.mean(z_o, axis=0 ,keepdims=True)) / np.std(z_o, axis=0 ,keepdims=True)
        z_o = np.multiply(z_o, varMat[j,:])
        z_o = np.add(z_o, meanMat[j,:])
        # Change dynamics
        for p_idx, pair in enumerate(edge_pairs):
            transitions[0][pair[0], pair[1]] = edge_weights[j, p_idx]
        # Mixing lags
        mixedDat = np.concatenate((z_l,z_o[:,:lags]), axis=-1)
        for i in range(lags):
            zt.append(np.copy(mixedDat[:,i,:]))
        for l in range(Nlayer):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_l = np.copy(mixedDat)
        for i in range(lags):
            # s lags
            obs_noise = np.random.uniform(0, noise_scale, (batch_size, observed_size))
            x_l[:,i,:] += obs_noise

            for j in range(observed_size - 1):
                x_l[:,i,j+1] += x_l[:,i,j] * 0.2
            # x lags
            xt.append(x_l[:,i,:])
            # xt.append(x_l[:,i,:])

        # Mixing future
        for i in range(length):
            # Generate noise term first
            z_t = np.random.normal(0, noise_scale, (batch_size, dyn_latent_size))
            # Transition function
            for l in range(lags):
                z_t += leaky_ReLU(np.dot(z_l[:,l,:], transitions[l]), negSlope)
            z_t = leaky_ReLU(z_t, negSlope)
            # Mixing function
            mixedDat = np.concatenate((z_t,z_o[:,lags+i]), axis=-1)
            zt.append(np.copy(mixedDat))
            for l in range(Nlayer):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            # generate x following causal order
            # for k in ordered_vertices:
            #     parents = list(G.predecessors(k))
            #     x_mixingList = generate_random_mixing_list(len(parents) + latent_size, 1, Nlayer) # [observation, latent, noise]
            #     obs_noise = np.random.normal(0, noise_scale, (batch_size, 1))
            #     mixedDat = np.concatenate((z_t, z_o[:,lags+i], x_t[:, parents] * B[parents, k]), axis=-1)
                
            #     for mixingMat in x_mixingList:    
            #         mixedDat = leaky_ReLU(mixedDat, negSlope)
            #         mixedDat = np.dot(mixedDat, mixingMat)
            #     x_t[:, k:k+1] = mixedDat# + obs_noise
            #obs_noise = np.random.normal(0, noise_scale, (batch_size, observed_size))
            obs_noise = np.random.uniform(0, noise_scale, (batch_size, observed_size))

            x_t += obs_noise
            st.append(np.copy(x_t))
            #x_t = leaky_ReLU(x_t, negSlope)
            # Chain structure
            for i in range(observed_size - 1):
                x_t[:, i + 1] += x_t[:, i] * 0.2
            # x_t[:,1] = x_t[:,0] * Bs[:, 0, 1] + obs_noise[:,1]
            # x_t = np.matmul(x_t, np.linalg.inv(np.eye(observed_size) - B).T)
            xt.append(x_t)
            z_l = np.concatenate((z_l, z_t[:,np.newaxis,:]), axis=1)[:,1:,:]

        zt = np.array(zt).transpose(1,0,2); st = np.array(st).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0); ht = np.array(ht).transpose(1,0);
        zt_ns.append(zt); st_ns.append(st); xt_ns.append(xt); ct_ns.append(ct); ht_ns.append(ht); 
        zt = [];  st=[]; xt = []; ct = []; ht=[];
        for l in range(lags):
            np.save(os.path.join(path, "W%d_%d"%(lags-l, j)), transitions[l])
        np.save(os.path.join(path, "varMat"), varMat)
        np.save(os.path.join(path, "meanMat"), meanMat) 

    np.save(os.path.join(path, "B"), B)
    np.save(os.path.join(path, "Bs"), Bs)
    #B_mask = randomly_set_zeros_to_ones(B_bin, int(math.log(observed_size ** 2)))
    B_mask = np.tril(np.ones((observed_size, observed_size)), k=-1)   
    M_mask = bin_mat(np.linalg.inv(np.eye(observed_size) - B_mask))
    np.save(os.path.join(path, "B_mask"), B_mask)
    np.save(os.path.join(path, "M_mask"), M_mask)
    zt_ns = np.vstack(zt_ns)
    xt_ns = np.vstack(xt_ns)
    ct_ns = np.vstack(ct_ns)
    ht_ns = np.vstack(ht_ns)
    st_ns = np.vstack(st_ns)
    np.savez(os.path.join(path, "data"), 
            zt = zt_ns, 
            xt = xt_ns,
            ct = ct_ns,
            ht = ht_ns,
            st = st_ns)
    
if __name__ == "__main__":
    seed_everything(42)
    data = nonparametric_ts(observed_size=8, dyn_latent_size=3, obs_latent_size=0, lags=2, Nlayer=1, seed=42)    
    #data = pnl_modular_gaussian_ts(observed_size=2, dyn_latent_size=2, obs_latent_size=0)    


