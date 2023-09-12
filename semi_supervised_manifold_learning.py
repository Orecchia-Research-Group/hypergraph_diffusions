"""
Tools for performing semi-supervised clustering by diffusing randomly seeded labels

"""

import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import json
from datetime import datetime
from tqdm import tqdm

from diffusion_functions import *
from animate_diffusion import animate_diffusion


import pdb

"""
DATA GENERATION 

Methods for building different "datasets" to cluster on. All methods return two
(2n_pts x 2) numpy arrays: the first is "clean" data, the second has noise added.

All methods construct both arrays such that that the first n/2 columns belong to 
community 1 and the latter n/2 columns all belong to community 2.
"""

def generate_spirals(tightness = 3, num_rotations = 1.5, n_pts = 300, noise_level = 1,verbose = True):
	# generate spiral polar coordinates
	theta = np.sqrt(np.linspace(start = (np.pi/2)**2, stop = (num_rotations*2*np.pi)**2,num=n_pts))
	r = tightness*theta
	# to cartesian coordinates
	spiral_1 = np.vstack([np.multiply(r, np.cos(theta)),np.multiply(r, np.sin(theta))]) 
	noisy_spiral_1 = spiral_1+ np.random.normal(scale = noise_level,size=(2,n_pts))
	# create second spiral by rotating by angle alpha in the plane
	alpha = np.pi
	rot_mat = np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
	spiral_2 = np.matmul(rot_mat,spiral_1)
	noisy_spiral_2 = spiral_2+ np.random.normal(scale = noise_level,size=(2,n_pts))

	if verbose:
		for man_1,man_2 in [(spiral_1,spiral_2),(noisy_spiral_1,noisy_spiral_2)]:
			plt.plot(man_1[0,:],man_1[1,:],'o',color='r')
			plt.plot(man_2[0,:],man_2[1,:],'o',color='b')
			plt.show()

	# combine into one dataset
	clean_data = np.hstack([spiral_1,spiral_2]).T
	noisy_data = np.hstack([noisy_spiral_1,noisy_spiral_2]).T

	return clean_data, noisy_data

def generate_overlapping_rings(r_1 = 2, r_2 = 3, n_pts = 300, x_shift = 3, 
	y_shift = 0, noise_level = 0.2, verbose = True):
	theta = np.linspace(start = 0, stop = 2*np.pi,num=n_pts)
	ring_1 = np.vstack([np.multiply(r_1, np.cos(theta)),np.multiply(r_1, np.sin(theta))])
	noisy_ring_1 = ring_1 + np.random.normal(scale=noise_level,size=(n_pts,2)).T

	# I'd like to change the density between the two
	ring_2 = np.vstack([np.multiply(r_2, np.cos(theta)),np.multiply(r_2, np.sin(theta))])
	ring_2 = ring_2 + np.hstack([np.full(shape=(n_pts,1),fill_value = x_shift),
								 np.full(shape=(n_pts,1),fill_value = y_shift)]).T
	noisy_ring_2 = ring_2 + np.random.normal(scale=noise_level,size=(n_pts,2)).T

	if verbose:
		for man_1,man_2 in [(ring_1,ring_2),(noisy_ring_1,noisy_ring_2)]:
			plt.plot(man_1[0,:],man_1[1,:],'o',color='r')
			plt.plot(man_2[0,:],man_2[1,:],'o',color='b')
			ax = plt.gca()
			ax.set_aspect('equal')
			plt.show()

	# combine into one dataset
	clean_data = np.hstack([ring_1,ring_2]).T
	noisy_data = np.hstack([noisy_ring_1,noisy_ring_2]).T

	return clean_data, noisy_data

def generate_concentric_highdim(ambient_dim = 5, r_inner = 1.3, r_outer = 2, n_pts = 300, 
								noise_level = 0.1,verbose = True):
	outer_shell = np.random.normal(size=(ambient_dim, n_pts))
	# normalize
	outer_shell = r_outer*np.divide(outer_shell, np.linalg.norm(outer_shell, ord = 2, axis = 0))
	noisy_outer_shell = outer_shell + np.random.normal(scale = noise_level, size=(ambient_dim, n_pts))
	
	# inner data
	# random unit vectors
	inner_sphere = np.random.normal(size=(ambient_dim, n_pts))
	inner_sphere = np.divide(inner_sphere, np.linalg.norm(inner_sphere, ord = 2, axis = 0))
	# sample radii by dim-th root
	radii = r_inner * np.power(np.random.uniform(low = 0.0, high = 1.0, size= n_pts), 1/ambient_dim)
	inner_sphere = np.multiply(radii, inner_sphere)
	
	#clean_data = inner_sphere.T # np.hstack([outer_shell,inner_sphere]).T
	#noisy_data = inner_sphere.T #np.hstack([noisy_outer_shell,inner_sphere]).T
	clean_data = np.hstack([outer_shell,inner_sphere]).T
	noisy_data = np.hstack([noisy_outer_shell,inner_sphere]).T
	
	if verbose:
		plot_projection(clean_data, labels='halves')
		plot_projection(noisy_data, labels='halves')
	
	return clean_data, noisy_data

def plot_projection(high_dim_data, labels = None):
	ax = plt.subplot()
	if labels=='halves':
		community_size = int(high_dim_data.shape[0]/2)
		plt.plot(high_dim_data[:community_size,0], high_dim_data[:community_size,1],'o')
		plt.plot(high_dim_data[community_size:,0], high_dim_data[community_size:,1],'o')
	else:    
		plt.plot(high_dim_data[:,0], high_dim_data[:,1],'o', c = labels)
	ax.set_aspect('equal')
	plt.show()
	return

"""
(HYPER)GRAPH CONSTRUCTION

Methods for building k-nearest-neighbor graphs and hypergraphs.

Assumes that data_matrix is n x 2, and that the first n/2 rows correspond
 to community 1, second n/2 rows correspond to community 2.
"""

def build_knn_graph(data_matrix, k):
	return kneighbors_graph(data_matrix, k, mode='connectivity', include_self=True)


def build_knn_hypergraph(data_matrix,k):
	nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data_matrix)
	_, indices = nbrs.kneighbors(data_matrix)

	n = data_matrix.shape[0]
	m = indices.shape[0]
	hypergraph =[tuple(edge) for edge in list(indices)]

	# the 'node dict' is the trivial one?
	node_dict = dict(zip(np.arange(n),np.arange(n)))
	# label all pts in spiral 1 as 0, label all pts in spiral 2 as 1
	labels = np.hstack([np.full(shape=int(n/2),fill_value = -1),np.full(shape=int(n/2),fill_value = 1)])
	label_names = dict({0:'spiral_1',1:'sprial_2'})

	# node_dict, labels, label_names
	return dict({'n':n,'m':m,'degree':k,'hypergraph':hypergraph,
		'node_dict': node_dict,'labels':labels,'label_names':label_names})

def compute_hypergraph_matrices(n, m, hypergraph, weights, hypergraph_node_weights=None):
    if weights is None:
        weights = defaultdict(lambda: 1)
    values = []
    i = []
    j = []
    w = []
    for row, e in enumerate(hypergraph):
        values.extend([1] * len(e) if hypergraph_node_weights is None else hypergraph_node_weights[e])
        i.extend([row] * len(e))
        j.extend(e)
        w.append(weights[e])
    W = sparse.diags(w).tocsr()
    sparse_h = sparse.coo_matrix((values, (i, j)), shape=(m, n))
    rank = np.array(sparse_h.sum(axis=0)).squeeze()
    return W, sparse_h, rank

"""
GRAPH DIFFUSION

Methods for vanilla graph diffusion

Currently implemented "slowly". Def possible to speedup for symmetric Laplacians
(i.e. undirected graphs).
"""

def is_symmetric(A, rtol=1e-05, atol=1e-08):
	return np.allclose(A, A.T, rtol=rtol, atol=atol)

def graph_quadratic(L,x):
	return x.T@L@x

def eval_graph_cut_fn(D,A,s,x):
	n = A.shape[0]

	D_inv = np.diag(np.divide(1,D))
	L = np.eye(n)-D_inv@A

	return graph_quadratic(L,x)# - s@x
	

def graph_diffusion(x0, D, A, s=None, h=0.5, T=100,verbose = True):
	n = x0.shape[0]
	if np.all(s==None):
		s = np.full(shape=(n,1),fill_value = 0)
	
	D_inv = np.diag(np.divide(1,D))
	L = np.eye(n)-D_inv@A
	
	#if is_symmetric(L):
	#    Sigma,U = np.linalg.eigh(L)
	#    Vh = U.T
	#else:
	#    U,Sigma,Vh = np.linalg.svd(L)
	#assert np.allclose(L,U@np.diag(Sigma)@Vh)
	
	x = np.reshape(x0,newshape = (1,n))
	y = np.reshape(L@x0 - s,newshape = (1,n))
	fx = [graph_quadratic(L,x0)]
	
	x_k = x0
	if verbose:
		t_start = datetime.now()
		print('Starting graph diffusion.')
		print('{:>10s} {:>6s} {:>13s} {:>14s}'.format('Time (s)', '# Iter', '||dx||_D^2', 'F(x(t))'))
	for t in range(T):
		grad = L@x_k - s
		x_k = x_k-h*grad
		x = np.append(x,np.reshape(x_k,newshape = (1,n)),axis=0)
		y = np.append(y,np.reshape(grad,newshape = (1,n)),axis=0)
		fx.append(graph_quadratic(L,x_k))
		if verbose:
			t_now = datetime.now()
			print(f'\r{(t_now - t_start).total_seconds():10.3f} {t:6d} {float(fx[-1]):14.6f} {np.abs(grad).min():10.6f}', end='')
	return x, y ,fx

"""
SINGLE-TRIAL EXPERIMENTS

Running semi-supervised clustering on a knn (hyper)graph via diffusions versus PPR.

Currently implemented "slowly". Def possible to speedup for symmetric Laplacians
(i.e. undirected graphs).
"""
def eval_hypergraph_cut_fn(hypergraph_objective, target_vector, s_vector, sparse_h, rank, W, D):
	_, _, fx = hypergraph_objective(target_vector, s_vector, sparse_h, rank, W, D)
	return fx


def diffusion_knn_clustering(knn_adj_matrix,knn_hgraph_dict,
		s_vector = None, hypergraph_objective = diffusion_functions['infinity'],
		num_rand_seeds = 30, step_size = 1, num_iterations = 100, verbose = True):
	
	# let's extract some parameters
	n = knn_hgraph_dict['n']
	m = knn_hgraph_dict['m']
	k = knn_hgraph_dict['degree']
	hypergraph = knn_hgraph_dict['hypergraph']

	D = np.full(shape=n,fill_value=k)

	# create an initial pt with num_rand_seeds randomly chosen true labels
	x0 = np.full(shape=(n,1),fill_value = 0)
	random_seeds = np.random.choice(np.arange(n),size = num_rand_seeds)
	x0[random_seeds[random_seeds < n/2]] = -1
	x0[random_seeds[random_seeds > n/2]] = 1
	if s_vector is None:
		s_vector = np.zeros_like(x0)

	# for our hypergraph, first specify the edge objective function
	t, x, y ,fx = diffusion(x0, n, m, D, hypergraph, weights=None, func=hypergraph_objective,
						 s=s_vector, h=step_size, T=num_iterations, verbose=verbose)

	W, sparse_h, rank = compute_hypergraph_matrices(n, m, hypergraph, weights=None)
	hypergraph_cut_objective = lambda vec: eval_hypergraph_cut_fn(hypergraph_objective, vec, s_vector, sparse_h, rank, W, D)
	hypergraph_diff_results = dict({'x':x,'y':y,'fx':fx,'objective':hypergraph_cut_objective,'type':'hypergraph'})

	# now run the vanilla graph diffusion
	# STEP SIZE 1/2
	x, y, fx = graph_diffusion(x0, D, knn_adj_matrix, s=s_vector, h=0.5, T=num_iterations,verbose = verbose)

	graph_cut_objective = lambda vec: eval_graph_cut_fn(D,knn_adj_matrix,s_vector,vec)
	graph_diff_results = dict({'x':x,'y':y,'fx':fx,'objective':graph_cut_objective,'type':'graph'})

	return hypergraph_diff_results, graph_diff_results


def PPR_knn_clustering(knn_adj_matrix,knn_hgraph_dict, error_tolerance = 0.1,
		teleportation_factor = 0.5, hypergraph_objective = diffusion_functions['infinity'],
		num_rand_seeds = 30, step_size = 1, num_iterations = 100, verbose = True):

	# teleportation_factor corresponds to a resolvent for lambda = effective_lambda
	effective_lambda = 2*teleportation_factor/(1-teleportation_factor)

	# let's extract some parameters
	n = knn_hgraph_dict['n']
	m = knn_hgraph_dict['m']
	k = knn_hgraph_dict['degree']
	hypergraph = knn_hgraph_dict['hypergraph']

	D = np.full(shape=n,fill_value=k)

	# create an s vector proportionate to label vector, with num_rand_seeds randomly chosen true labels
	seeded_labels = np.full(shape=(n,1),fill_value = 0)
	random_seeds = np.random.choice(np.arange(n),size = num_rand_seeds)
	seeded_labels[random_seeds[random_seeds < n/2]] = -1
	seeded_labels[random_seeds[random_seeds > n/2]] = 1
	s_vector = effective_lambda*seeded_labels

	# step size: epsilon/2*u_R
	step_size = error_tolerance/(2*(1+effective_lambda))

	# Algorithm 1 specifies initialization at 0
	x0 = np.full(shape=(n,1),fill_value = 0)
	_, x, y ,fx = diffusion(x0, n, m, D, hypergraph, weights=None, func=hypergraph_objective,
						 s=s_vector, h=step_size, T=num_iterations, verbose=verbose)				 
	x_out = (1-error_tolerance/2)*np.sum(x, axis = 0).flatten()

	W, sparse_h, rank = compute_hypergraph_matrices(n, m, hypergraph, weights=None)
	hypergraph_cut_objective = lambda vec: eval_hypergraph_cut_fn(hypergraph_objective, vec, s_vector, sparse_h, rank, W, D)
	hypergraph_PPR_results = dict({'x_out':x_out, 'objective':hypergraph_cut_objective,'type':'hypergraph'})

	# Now collect graph PPR vector
	# manually running comparable iterations on the graph
	# L = np.diag(D) - knn_adj_matrix
	D_inv = np.diag(np.divide(1,D))
	# gradient_operator = 2*L + effective_lambda*np.diag(D)
	# x_t = np.full(shape=(n,1),fill_value = 0)
	# x = [x_t]
	# for idx in range(num_iterations):
	# 	x_hat_t = x_t + step_size*D_inv.dot(s_vector)
	# 	x_t = x_hat_t - step_size*(D_inv@gradient_operator).dot(x_hat_t)
	# 	x.append(x_t)
	# x = np.array(x)
	# graph_PPR = (1-error_tolerance/2)*np.sum(x, axis = 0).flatten()

	graph_PPR = np.linalg.solve(a = (1+effective_lambda)*np.eye(n) - D_inv@knn_adj_matrix , b = np.dot(D_inv, s_vector))
	# flatten n x 1 matrix
	graph_PPR = graph_PPR.reshape(n)

	graph_cut_objective = lambda vec: eval_graph_cut_fn(D,knn_adj_matrix,s_vector,vec)
	graph_PPR_results = dict({'x_out':graph_PPR,'objective':graph_cut_objective,'type':'graph'})

	return hypergraph_PPR_results, graph_PPR_results

def compare_estimated_labels(method, generate_data, k, num_iterations,
	diffusion_step_size = None,titlestring=None):
	
	# generate new data
	_,data_matrix = generate_data(verbose = False)
	n = data_matrix.shape[1]

	# build graph/hypergraph
	knn_adj_matrix = build_knn_graph(data_matrix,k)
	knn_hgraph_dict = build_knn_hypergraph(data_matrix,k)

	# run diffusion
	if method=='diffusion':
		hypergraph_diff_results, graph_diff_results = diffusion_knn_clustering(knn_adj_matrix,
						knn_hgraph_dict, num_iterations = num_iterations, verbose = False)
		hypergraph_x = hypergraph_diff_results['x']
		graph_x = graph_diff_results['x']
		return graph_x[-1, :], hypergraph_x[-1, :], data_matrix

	elif method=='PPR':
		hypergraph_PPR_results, graph_PPR_results = PPR_knn_clustering(knn_adj_matrix,
								knn_hgraph_dict, error_tolerance = 0.1, teleportation_factor = 0.5,
								num_iterations = num_iterations, verbose = False)
		return graph_PPR_results['x_out'], hypergraph_PPR_results['x_out'], data_matrix


"""
ASSESMENT UTILITIES

Methods for assessing the performance of estimates produced by diffusions.
"""

def make_sweep_cut(vector, threshold):
	mask = np.full(shape=vector.shape,fill_value = np.nan)
	mask[np.where(vector <= threshold)] = -1
	mask[np.where(vector > threshold)] = 1
	return mask
	
def sweep_cut_classification_error(label_estimates):
	n = label_estimates.shape[0]
	n_pts = int(n/2)
	labels = np.hstack([np.full(shape=n_pts,fill_value = -1),np.full(shape=n_pts,fill_value = 1)])
	return sum(np.reshape(label_estimates,newshape=(n))!=labels)/n

def find_min_sweepcut(node_values,resolution,cut_objective_function, orthogonality_constraint = 'auto'):
	ascending_node_values = np.unique(node_values)
	# sweep from lowest nontrivial cut to highest nontrivial cut
	low = ascending_node_values[1]
	high = ascending_node_values[-2]

	min_observed_value = np.inf
	best_threshold = low

	if orthogonality_constraint=='auto':
		# find orthogonality constraint created by 0-threshold and add 10% buffer
		zero_estimates = make_sweep_cut(node_values, 0)
		orthogonality_constraint = np.abs(np.sum(zero_estimates)/len(zero_estimates))+0.1
	
	sweep_vals = np.append(np.linspace(low,high,num=resolution,endpoint=False),0.0)

	for threshold in sweep_vals:
		label_estimates = make_sweep_cut(node_values, threshold)
		objective_value = cut_objective_function(label_estimates)
		orthogonality_error = np.abs(np.sum(label_estimates)/len(label_estimates))

		if objective_value < min_observed_value and orthogonality_error < orthogonality_constraint:
			min_observed_value = objective_value
			best_threshold = threshold
	return min_observed_value, best_threshold

"""
EXPERIMENTS 

Methods for running specific experiments and generating figures.
"""
def visualize_labels(method='PPR'):
	k = 5
	target_iternum = 50
	titlestring = 'blah'

	_, ax_binary = plt.subplots(nrows=3, ncols=2,figsize=(10, 15))
	problem_index = 0
	for data_generation, problem_kind in [(generate_spirals,' Spirals'), (generate_overlapping_rings,' Rings'),
										(generate_concentric_highdim,' Concentric Highdim')]:
		graph_x_out, hypergraph_x_out, data_matrix = compare_estimated_labels(method,data_generation,k,target_iternum,titlestring=None, diffusion_step_size=1)

		for idx,(x,titlestring) in enumerate([(graph_x_out,'Graph'), (hypergraph_x_out,'Hypergraph')]):
			if problem_index==0:
				plot_label_comparison_binary(ax_binary[problem_index, idx],x, data_matrix,titlestring)
			else:
				plot_label_comparison_binary(ax_binary[problem_index, idx],x, data_matrix,titlestring = 'Abridged')
		problem_index+=1

	plt.suptitle(f'Label estimates \n Iteration {target_iternum}', fontsize = 15)    
	plt.show()


def compare_AUC_curves(method='PPR'):
	k = 5
	num_iterations = 50
	num_trials = 20

	fig, ax = plt.subplots(nrows=3, ncols=1, figsize = (6, 15))
	axes_idx = 0

	for data_generation, problem_kind in [(generate_spirals,'Spirals'), (generate_overlapping_rings,'Rings'),
										(generate_concentric_highdim,'Concentric hyperspheres')]:
		AUC_vals = []
		for trial in range(num_trials):
			# generate new data
			_,data_matrix = data_generation(verbose=False)

			# build graph/hypergraph
			knn_adj_matrix = build_knn_graph(data_matrix,k)
			knn_hgraph_dict = build_knn_hypergraph(data_matrix,k)

			# run diffusion
			if method=='diffusion':
				hypergraph_diff_results, graph_diff_results = diffusion_knn_clustering(knn_adj_matrix,
								knn_hgraph_dict, num_iterations = num_iterations, verbose = False)
				graph_x = graph_diff_results['x']
				hypergraph_x = hypergraph_diff_results['x']

				graph_x_out = graph_x[-1, :]
				hypergraph_x_out = hypergraph_x[-1, :]
			elif method=='PPR':
				hypergraph_diff_results, graph_diff_results = PPR_knn_clustering(knn_adj_matrix,
								knn_hgraph_dict, error_tolerance = 0.1, teleportation_factor = 0.5,
								num_iterations = num_iterations, verbose = False)
				graph_x_out = graph_diff_results['x_out']
				hypergraph_x_out = hypergraph_diff_results['x_out']

			n = data_matrix.shape[0]
			labels = np.hstack([np.full(shape=int(n/2),fill_value = -1),np.full(shape=int(n/2),fill_value = 1)])
			graph_auc_score = metrics.roc_auc_score(labels, graph_x_out)
			hypergraph_auc_score = metrics.roc_auc_score(labels, hypergraph_x_out)

			AUC_vals.append((hypergraph_auc_score, graph_auc_score))
		titlestring = f'AUC Values at Iteration {num_iterations} \n Results from {num_trials} Independent Trials'
		final_plot_AUC_hist(AUC_vals, ax = ax[axes_idx], decorated = (axes_idx == 0), titlestring = titlestring )
		axes_idx+=1
	plt.show()

"""
PLOTTING UTILITIES

Specific visualizations for figures in paper.
"""

def plot_label_comparison_binary(ax, label_vector, data_matrix, titlestring=None):
	sweep_cut_resolution = 100
	error, threshold = find_min_sweepcut(label_vector,sweep_cut_resolution,sweep_cut_classification_error)
	label_estimates = make_sweep_cut(label_vector, threshold)
	error = sweep_cut_classification_error(label_estimates)
	im = ax.scatter(data_matrix[:,0],data_matrix[:,1], c=label_estimates.reshape(-1))

	# figure formatting
	ax.set_aspect('equal')
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.axis('off')
	if titlestring=='Abridged':
		ax.set_title(f'Classification error = {error:.3f}', fontsize = 15)
	else:
		ax.set_title(titlestring +f'\n Classification error = {error:.3f}', fontsize = 15)
	return

def final_plot_AUC_hist(AUC_vals, ax, decorated = False, titlestring = None):
	plt.rcParams.update({'font.size': 15})
		
	hypergraph_vals = [v[0] for v in AUC_vals]
	graph_vals = [v[1] for v in AUC_vals]

	full_values = hypergraph_vals+graph_vals
	_, first_bins = np.histogram(full_values, bins = 10)

	# second style
	ax.hist(graph_vals, bins = first_bins, alpha=0.5, edgecolor = 'black', label = 'graph')
	ax.hist(hypergraph_vals, bins = first_bins, alpha=0.5, edgecolor = 'black', label='hypergraph')

	if decorated:
		if not (titlestring is None):
			ax.set_title(titlestring)
		ax.legend()
		
	# figure formatting
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.tick_params(axis='x', labelsize=15)
	ax.tick_params(axis='y', labelsize=15)
	return

#compare_estimated_labels(method='PPR', generate_data = generate_spirals, k=5, num_iterations = 10)

#compare_AUC_curves(method='PPR')
visualize_labels(method='PPR')