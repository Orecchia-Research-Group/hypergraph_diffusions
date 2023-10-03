"""
Computing diffusions for arbitrary submodular cut functions

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import pdb

from semi_supervised_manifold_learning import generate_spirals, build_knn_hypergraph, plot_label_comparison_binary
from diffusion_functions import diffusion
from multiprocessing import Pool

"""
HELPERS

"""
# For M a positive definite matrix, compute logdet using Cholesky decomposition
def logdet(M):
    L = np.linalg.cholesky(M)
    return 2*np.sum(np.log(np.diag(L)))

# Turn a sparse hypergraph object into a list of hyperedge lists
def unpack_sparse_hypergraph(sparse_h, rank):
    hypergraph = []
    he = sparse_h.col
    k = 0
    for r in rank:
        hypergraph.append([he[j] for j in range(k, k + int(r))])
        k += int(r)
    return hypergraph
"""
SUBMODULAR OBJECTIVES 

Some submodular functions for testing submodular evaluation utils.

Submodular functions take as arguments a Python set, subset of integers 0,..,n-1.
"""

# Returns min(1, |S|, n-|S|)
def trivial_cardinality_cut_fn(S, n):
    return min([1, len(S), n-len(S)])

# Returns min(1, |S intersect h|,|(V\S) intersect h|) where V = {0,...,n-1}
# note: since V contains h, (V\S intersect h) = (h\S)
def cardinality_cut_fn(S, h):
    return min([1, len(S.intersection(h)), len(h.difference(S))])

# Computes the mutual information using global covariance matrix K
# Optional: logdet_dict stores evaluations of log(det(K_A, A)) for sets A
# NOTE: this is modifying logdet_dict inside the function call.
def mutual_information(S, h, K, logdet_dict = None):
    h_cap_S = h.intersection(S)
    h_minus_S = h.difference(S)
    # if S is disjoint from h OR if S contains h, mutual information is zero
    if (not h_cap_S) or (not h_minus_S):
        return 0
    # if there's some nontrivial overlap, continue
    MI_h = 0
    for sign, target_set in [(1,h_cap_S),(1,h_minus_S), (-1,h)]:
        if (logdet_dict is not None) and frozenset(target_set) in logdet_dict:
            logdet_dict['utilized']+=1
            MI_h += sign*logdet_dict[frozenset(target_set)]
        else:
            # if there's only one element, the log-determinant is the log of that element
            if len(target_set)==1:
                logdet_eval = np.log(np.asscalar(K[list(target_set), list(target_set)]))
            else:
                logdet_eval = logdet(K[np.ix_(list(target_set),list(target_set))])
                if logdet_dict is not None:
                    logdet_dict[frozenset(target_set)] = logdet_eval
            MI_h += sign*logdet_eval
    # full definition includes normalization by factor of 1/2
    return 0.5*MI_h

"""
LOVASZ EXTENSION

Generic algorithm for computing a subgradient (maximizing element of the base polytope)
and subsequent extension value.
"""

# F is some real-valued (assumed submodular) fn on subsets of [n], x is some real-valued
# n-dimensional numpy array
# optional argument to minimize number of times we re-sort x
def greedy_subgradient(F, x, idx_decreasing=None):
    n = x.size
    # if x is not flat (i.e. x.shape = (n,1)) flatten
    if x.ndim > 1:
        # make sure we're dealing with an (n,1) vector, otherwise return an error
        assert x.shape[1]==1
        x = np.reshape(x, newshape = n)
    y = np.full(n, fill_value = np.nan)
    # if sorted input is not provided, sort.
    if idx_decreasing is None:
        idx_decreasing = (-x).argsort()
    # need evals for emptyset (i=0) and full set (i=n), hence iterating to n (inclusive)
    subset_evals = {i:F(set(idx_decreasing[:i]),n) for i in range(0,n+1)}
    for i in range(n):
        # subset_evals[0] corresponds to the empty set, subset evals[1] corresponds to 
        # S_j for j the first element of idx_decreasing. Hence, compared to indexing in
        # e.g. Bach 2013, y[pi[j]] = F(S_{:j}) - F(S_{:j-1}), we shift up by 1
        y[idx_decreasing[i]] = subset_evals[i+1]-subset_evals[i]
    return y

def greedy_extension(F, x):
    return np.inner(x, greedy_subgradient(F, x))

"""
SUBMODULAR DIFFUSION FUNCTION

Computes subgradient and fn value for diffusion in diffusion_functions.py
"""
# a function which takes in a point x and a hyperedge and returns (grad_h(x), f_h(x))
# the contributions to the subgradient and potential function from that hyperedge at x
def subgrad_eval(x,h_list,cut_fn,idx_decreasing,W_h):
    n = len(idx_decreasing)
    # define the submodular cut function for the hyperedge
    F_h = lambda S, _: cut_fn(S, set(h_list))

    # get y_h \in R^|h| the subgradient. Note: we run this on the full 
    # vector x, not a restricted x_h, but all entries of y_h corresponding
    # to nodes not in h should be 0
    y_h = greedy_subgradient(F_h, x, idx_decreasing=idx_decreasing)

    # check that all nodes not in h have corresponding entry 0
    excluded_idxs = set(range(n)).difference(set(h_list))
    assert np.all(y_h[list(excluded_idxs)]==0)

    # contribution to global gradient is W_h * delta_h * y_h
    delta_h = np.inner(y_h, x)
    grad_h=W_h*delta_h*y_h

    # contribution to global potential is (1/2)*W_h * delta_h^2
    f_h= 0.5*W_h*delta_h**2
    return (grad_h, f_h)

# Evaluate hypergraph potential subgradient using Lovasz extension wrt some submodular hyperedge cut function
# cut_fn(S,h) should take in two sets
def submodular_subgradient(cut_fn, x, sparse_h, rank, W, D, center_id=None,
                                 hypergraph_node_weights=None, parallelize = True):
    hypergraph = unpack_sparse_hypergraph(sparse_h, rank)

    n = x.shape[0]
    # flatten x
    x = np.reshape(x, newshape = n)
    # sort entries of x to provide to greedy submodular subgradient evaluation
    idx_decreasing = (-x).argsort()

    if hypergraph_node_weights is None:
        hypergraph_node_weights = {tuple(e): [1] * len(e) for e in hypergraph}
    #subgrad_eval_by_index = lambda i: subgrad_eval(i,x,hypergraph,cut_fn,idx_decreasing,W)

    grad_list = list()
    f_list = list()
    if parallelize:
        items = [(x,h_list,cut_fn,idx_decreasing,W[i,i]) for i,h_list in enumerate(hypergraph)]
        pool = Pool()
        grad_list, f_list = zip(*pool.starmap(subgrad_eval,items))
    else:
        for i,h_list in enumerate(hypergraph):
            grad_h, f_h = subgrad_eval(x,h_list,cut_fn,idx_decreasing,W[i,i])

            grad_list.append(grad_h)
            f_list.append(f_h)
    
    # I don't understand the role that the variable y is serving--currently returning a dummy in its place
    return np.reshape(np.sum(np.array(grad_list),axis=0),newshape = (n,1)), np.nan, np.sum(f_list)

# A specific implementation of mutual info for ease of parallelization
# a function which takes in a point x and a hyperedge and returns (grad_h(x), f_h(x))
# the contributions to the subgradient and potential function from that hyperedge at x
def mutual_info_subgrad_eval(K,x,h_list,idx_decreasing,W_h):
    n = len(idx_decreasing)
    # define the submodular cut function for the hyperedge
    F_h = lambda S, _: mutual_information(S, set(h_list), K)#, logdet_dict = None)

    y_h = greedy_subgradient(F_h, x, idx_decreasing=idx_decreasing)
    excluded_idxs = set(range(n)).difference(set(h_list))
    assert np.all(y_h[list(excluded_idxs)]==0)
    delta_h = np.inner(y_h, x)
    grad_h=W_h*delta_h*y_h
    f_h= 0.5*W_h*delta_h**2
    return (grad_h, f_h)

# A parallelized method for computing the subgradient with respect to the mutual information cut fn
# K is the Kernel matrix
def parallelized_mutual_info_subgradient(K, x, sparse_h, rank, W, D, center_id=None, hypergraph_node_weights=None, parallelize = True):
    hypergraph = unpack_sparse_hypergraph(sparse_h, rank)

    n = x.shape[0]
    # flatten x
    x = np.reshape(x, newshape = n)
    # sort entries of x to provide to greedy submodular subgradient evaluation
    idx_decreasing = (-x).argsort()

    if hypergraph_node_weights is None:
        hypergraph_node_weights = {tuple(e): [1] * len(e) for e in hypergraph}
    
    grad_list = list()
    f_list = list()
    if parallelize:
        items = [(K,x,h_list,idx_decreasing,W[i,i]) for i,h_list in enumerate(hypergraph)]
        pool = Pool()
        grad_list, f_list = zip(*pool.starmap(mutual_info_subgrad_eval,items))
    else:
        for i,h_list in enumerate(hypergraph):
            grad_h, f_h = mutual_info_subgrad_eval(K,x,h_list,idx_decreasing,W[i,i])

            grad_list.append(grad_h)
            f_list.append(f_h)
    

    return np.reshape(np.sum(np.array(grad_list),axis=0),newshape = (n,1)), np.nan, np.sum(f_list)

"""
TESTS
"""
def test_trivial_cut_fn():
    # Trivial cut function should return x_max - x_min
    x = np.random.normal(size = 50)
    return greedy_extension(trivial_cardinality_cut_fn, x) == (max(x)-min(x))

# test out the cardinality cut fn on random x and randomly chosen ``hyperedge'' sets h
# runs random independent tests for varying values of n
def test_cardinality_cut_fn():
    for n in [5, 50, 500]:
        for _ in range(100):
            h_set = set(np.random.choice(n, size = n-1, replace = False))#np.random.randint(low = 1, high=n), replace = False))
            F = lambda S, n: cardinality_cut_fn(S, h_set, n)
            x = np.random.normal(size = n)
            # Trivial cut function should return x(h)_max - x(h)_min
            x_h = x[list(h_set)]
            assert greedy_extension(F, x)==(max(x_h)-min(x_h))
            y_h = greedy_subgradient(F, x)
            excluded_idx = set(range(n)).difference(h_set)
            assert len(excluded_idx)==1
            excluded_idx = list(excluded_idx)[0]
            assert y_h[excluded_idx]==0
    return True

# build a hypergraph and compute subgradients for a generic submodular hyperedge cut fn
# Uses an objective cut_func(S, h) which is assumed to take in sets S and h and be submodular.
def submodular_semisupervised_clustering(method='PPR', teleportation_factor = 0.5, error_tolerance = 0.1):
    # generate new data
    _,data_matrix = generate_spirals(verbose = False) #n_pts = 50,  start_theta = np.pi/5, num_rotations = 0.9, verbose = False)
    plt.plot(data_matrix[:,0], data_matrix[:,1],'o')
    plt.show()

    # build a hypergraph from k-nearest-nbs of each point
    k = 5
    knn_hgraph_dict = build_knn_hypergraph(data_matrix,k)
    n = knn_hgraph_dict['n']
    m = knn_hgraph_dict['m']
    k = knn_hgraph_dict['degree']
    hypergraph = knn_hgraph_dict['hypergraph']

    # build the Gaussian kernel associated w/data e^-alpha*||x_i-x_j||^2
    bandwidth = 10e-3
    K = np.exp(-bandwidth*np.square(pairwise_distances(data_matrix)))

    # create an s vector proportionate to label vector, with num_rand_seeds randomly chosen true labels
    num_rand_seeds = int(0.1*n)
    seeded_labels = np.full(shape=(n,1),fill_value = 0)
    random_seeds = np.random.choice(np.arange(n),size = num_rand_seeds)
    seeded_labels[random_seeds[random_seeds < n/2]] = -1
    seeded_labels[random_seeds[random_seeds > n/2]] = 1

    D = np.full(shape=n,fill_value=k)

    if method=='diffusion':
        step_size = 0.1
        x0 = seeded_labels
        s_vector = np.zeros_like(x0)

    elif method=='PPR':
        x0 = np.zeros_like(seeded_labels)
        s_vector = seeded_labels
        effective_lambda = 2*teleportation_factor/(1-teleportation_factor)
        step_size = error_tolerance/(2*(1+effective_lambda))

    # Some options for hyperedge cut functions:
    # Cardinality based cut function, implemented with greedy Lovasz extension computation (slower)
    cardinality_cut_func = lambda *args, **kwargs : submodular_subgradient(cardinality_cut_fn, *args, **kwargs)
    # Mutual information: initialize a dictionary to store logdet evals, hashed by set
    # logdet_dict = dict()
    # logdet_dict['utilized'] = 0 # save a count of how often we're looking up logdets, for my reference
    # MI_h = lambda S, h: mutual_information(S, h, K, logdet_dict)
    # MI_cut_func = lambda *args, **kwargs : submodular_subgradient(MI_h, *args, **kwargs)
    MI_cut_func = lambda *args, **kwargs : parallelized_mutual_info_subgradient(K, *args, **kwargs)

    # for our hypergraph, first specify the edge objective function
    for cut_func, title in [(MI_cut_func,'mutual information'),(cardinality_cut_func,'cardinality-based')]:
        _, x, _ ,_ = diffusion(x0, n, m, D, hypergraph, weights=None, func = cut_func,
                                h = step_size, s=s_vector, T=10, verbose=True)
        if method=='diffusion':
            estimated_labels = x[-1]
        elif method=='PPR':
            estimated_labels = (1-error_tolerance/2)*np.sum(x, axis = 0).flatten()

        _, ax = plt.subplots()
        plot_label_comparison_binary(ax, estimated_labels, data_matrix, titlestring=title)
        plt.show()

submodular_semisupervised_clustering()