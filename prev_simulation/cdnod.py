from causallearn.search.ConstraintBased.CDNOD import cdnod
from lingam.utils import make_dot
from Caulimate.Data.SimDAG import simulate_random_dag, simulate_weight
from Caulimate.Utils.Tools import bin_mat, makedir
from causallearn.utils.GraphUtils import GraphUtils
import numpy as np  
import networkx as nx   
import matplotlib.pyplot as plt 
import os

seed = 4
obs_dim = 10
n_samples = 6000
save_dir = 'cdnod'

dag = simulate_random_dag(obs_dim, 1, 'ER', seed)
mat = simulate_weight(dag, ((-10.5, -0.5), (0.5, 10.5)), seed)
dot = make_dot(bin_mat(mat), labels=[f'x{str(i+1)}' for i in range(obs_dim)])

# Save pdf
# dot.render('dag')
# Save png
dot.format = 'png'
dot.render(os.path.join(save_dir, 'gt'))

def simulate_data(n_samples, n_features, causal_matrix):
    # Initialize data matrix
    data = np.zeros((n_samples, n_features))
    c_indx = np.linspace(1, n_samples, n_samples)
    c_indx = np.expand_dims(c_indx, axis=1) 
    
    # Generate random noise
    noise = np.random.uniform(-0.1, 0.1, size=(n_samples, n_features))
    
    # Simulate causal relationships
    for i in range(n_features):
        parents = np.nonzero(causal_matrix[:, i])[0]  # Find indices of parents for variable i
        if len(parents) > 0:
            parent_values = data[:, parents]
            # Simulate non-linear relationship, e.g., quadratic
            data[:, i] = np.sum(parent_values ** 3, axis=1) + noise[:, i]
        else:
            data[:, i] = noise[:, i]  # No parents, just noise
            
    return data + c_indx, c_indx

# simulate data
data, c_indx = simulate_data(n_samples, obs_dim, dag)

# or customized parameters
alpha = 0.01
cg = cdnod(data, c_indx, aplha=alpha)

# visualization using pydot
# note that the last node is the c_indx
cg.draw_pydot_graph()

# or save the graph


pyd = GraphUtils.to_pydot(cg.G)
pyd.write_png(os.path.join(save_dir, 'cdnod.png'))

