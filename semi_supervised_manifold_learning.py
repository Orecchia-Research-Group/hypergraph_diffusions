"""
Tools for performing semi-supervised clustering by diffusing randomly seeded labels

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import json
from datetime import datetime
from tqdm import tqdm

from diffusion_functions import *
from animate_diffusion import animate_diffusion


import pdb

"""
DATA GENERATION 
from tqdm import tqdm

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

def generate_concentric_circles(r_1 = 2, r_2 = 1.3, n_pts = 300, noise_level = 0.1,verbose = True):
	theta = np.linspace(start = 0, stop = 2*np.pi,num=n_pts)
	ring = np.vstack([np.multiply(r_1, np.cos(theta)),np.multiply(r_1, np.sin(theta))])
	noisy_ring = ring + np.random.normal(scale=noise_level,size=(2,n_pts))

	rand_radii = np.random.uniform(low=0,high=r_2,size = n_pts)
	circle = np.vstack([np.multiply(rand_radii, np.cos(theta)),np.multiply(rand_radii, np.sin(theta))])
	noisy_circle = circle #+ np.random.normal(scale=noise_level,size=(2,n_pts))

	if verbose:
		for man_1,man_2 in [(ring,circle),(noisy_ring,noisy_circle)]:
			plt.plot(man_1[0,:],man_1[1,:],'o',color='r')
			plt.plot(man_2[0,:],man_2[1,:],'o',color='b')
			ax = plt.gca()
			ax.set_aspect('equal')
			plt.show()
		
	# combine into one dataset
	clean_data = np.hstack([ring,circle]).T
	noisy_data = np.hstack([noisy_ring,noisy_circle]).T
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

Running semi-supervised clustering on a knn (hyper)graph via diffusions.

Currently implemented "slowly". Def possible to speedup for symmetric Laplacians
(i.e. undirected graphs).
"""
def eval_hypergraph_cut_fn(hypergraph_objective, target_vector, s_vector, sparse_h, rank, W, D):
	_, _, fx = hypergraph_objective(target_vector, s_vector, sparse_h, rank, W, D)
	return fx


def semi_superivsed_knn_clustering(knn_adj_matrix,knn_hgraph_dict,
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

	# for our hypergraph, first specify the edge objective function
	x, y ,fx = diffusion(x0, n, m, D, hypergraph, weights=None, func=hypergraph_objective, 
						 s=s_vector, h=step_size, T=num_iterations, verbose=verbose)

	if np.all(s_vector==None):
		s_vector = np.full(shape=(n,1),fill_value = 0)

	W, sparse_h, rank = compute_hypergraph_matrices(n, m, hypergraph, weights=None)
	hypergraph_cut_ojbective = lambda vec: eval_hypergraph_cut_fn(hypergraph_objective, vec, s_vector, sparse_h, rank, W, D)
	hypergraph_diff_results = dict({'x':x,'y':y,'fx':fx,'objective':hypergraph_cut_ojbective,'type':'hypergraph'})

	# now run the vanilla graph diffusion
	# STEP SIZE 1/2
	x, y ,fx = graph_diffusion(x0, D, knn_adj_matrix, s=s_vector, h=0.5, T=num_iterations,verbose = verbose)

	graph_cut_objective = lambda vec: eval_graph_cut_fn(D,knn_adj_matrix,s_vector,vec)
	graph_diff_results = dict({'x':x,'y':y,'fx':fx,'objective':graph_cut_objective,'type':'graph'})

	return hypergraph_diff_results, graph_diff_results

"""
ANIMATION

Fun part! Animating our results.

All methods assume you've created a matplotlib subplots object with the right
 number of subplots for the list of results you're feeding in.
"""

# Given a list of results dicts, animate all side-by-side.
def side_by_side_animation(fig,ax, results, data_matrix, animation_fn):
	if isinstance(results, dict):
		num_frames = results['x'].shape[0]
	elif isinstance(results, list):
		num_frames = min([result_dict['x'].shape[0] for result_dict in results])
	
	ani = matplotlib.animation.FuncAnimation(fig, lambda i: call_animation_fn(ax,animation_fn,data_matrix, results,
															i),frames=num_frames,interval=150, repeat=False)
	return ani

def call_animation_fn(ax, animation_fn, data_matrix, results,  i):
	n = data_matrix.shape[0]
	
	if isinstance(results, dict):
		animation_fn(data_matrix, results, ax, i,title=results['type'])
	
	elif isinstance(results, list):
		for idx,results in enumerate(results):
			animation_fn(data_matrix, results, ax[idx], i,title=results['type'])

	return ax

def animate_pts_in_plane(data_matrix, results, ax, i,title=None):
	x = results['x']

	ax.clear()
	
	im = ax.scatter(data_matrix[:,0],data_matrix[:,1],c=x[i])
	# for colorbar implementation, maybe check out solutions here:
	# https://stackoverflow.com/questions/39472017/how-to-animate-the-colorbar-in-matplotlib

	ax.set_title(title)
	return ax

def animate_pt_separation(data_matrix, results, ax, i,title=None):
	x = results['x']

	n = x.shape[1]
	fn_values = x[i]

	ax.clear()
	ax.scatter(fn_values[:int(n/2)],np.full(shape=int(n/2),fill_value = -1))
	ax.scatter(fn_values[int(n/2):],np.full(shape=int(n/2),fill_value = 1))
	# draw a line at zero for ease of visualizing
	ax.plot(np.full(shape=10,fill_value = 0),np.linspace(-1,1,num=10))
	if title:
		ax.set_title(title)
	return ax

def animate_sweep_cut_by_threshold(data_matrix, results, ax, i,title=None):
	x = results['x']

	n = x.shape[1]

	sweep_vals = np.linspace(-1,1,num=x.shape[0])
	threshold = sweep_vals[i]
	final_values = x[-1,:]
	mask = make_sweep_cut(final_values, threshold)
	
	ax.clear()
	ax.scatter(data_matrix[:,0],data_matrix[:,1],c=mask)
	classification_error = sweep_cut_classification_error(mask)
	
	error_string = f' with classification error ={classification_error:.2f} \n threshold = {threshold:.2f}'
	ax.set_title(title+error_string)

	return ax

def animate_sweep_cut_in_plane(data_matrix, results, ax, i,title=None):
	x = results['x']

	cut_objective_function = results['objective']

	ax.clear()
	
	# adaptive option: find the best sweepcut
	if False:
		objective_value,threshold = find_min_sweepcut(x[i],resolution = 100, cut_objective_function = cut_objective_function)
		error = sweep_cut_classification_error(make_sweep_cut(x[i], threshold))
		error_string = f'\n iteration {i:d}, f(sweepcut) = {objective_value:.2f}, threshold = {float(threshold):.2f} \n class. error of energy-min. cut ={error:.2f}'
	
	# non-adaptive: cut around 0
	if True:
		threshold = 0 
		label_estimates = make_sweep_cut(x[i], threshold)
		error = sweep_cut_classification_error(label_estimates)
		objective_value = cut_objective_function(label_estimates)
		error_string = f'\n iteration {i:d}, f(sweepcut) = {objective_value:.2f} \n class. error of threshold=0 cut ={error:.2f}'
		
	#im = ax.scatter(data_matrix[:,0],data_matrix[:,1],c=x[i])
	#plt.colorbar(im, ax=ax)
	ax.scatter(data_matrix[:,0],data_matrix[:,1],c=make_sweep_cut(x[i], threshold))

	#matplotlib.colorbar.ColorbarBase(ax=ax,  values=sorted(x[i]))

	ax.set_title(title+error_string)
	return ax

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
MULTI-TRIAL EXPERIMENTS 

Methods for running many repeated trials and comparing performance.
"""

def compare_estimated_labels(generate_data,k,target_iteration,
	diffusion_step_size,titlestring=None):
	
	# generate new data
	_,data_matrix = generate_data(verbose = False)

	# build graph/hypergraph
	knn_adj_matrix = build_knn_graph(data_matrix,k)
	knn_hgraph_dict = build_knn_hypergraph(data_matrix,k)

	# run diffusion
	n = data_matrix.shape[1]
	hypergraph_diff_results, graph_diff_results = semi_superivsed_knn_clustering(knn_adj_matrix,
					knn_hgraph_dict, num_iterations = target_iteration, verbose = False)

	return graph_diff_results['x'], hypergraph_diff_results['x'], data_matrix


def repeated_clustering_experiments(generate_data,k,diffusion_iterations,
	diffusion_step_size,num_trials,titlestring=None):
	graph_error_vs_iteration = list()
	hypergraph_error_vs_iteration = list()

	# For each trial,
	for idx in tqdm(range(num_trials)):
		# generate new data
		_,data_matrix = generate_data(verbose = False)

		# build graph/hypergraph
		knn_adj_matrix = build_knn_graph(data_matrix,k)
		knn_hgraph_dict = build_knn_hypergraph(data_matrix,k)

		# run diffusion
		n = data_matrix.shape[1]
		hypergraph_diff_results, graph_diff_results = semi_superivsed_knn_clustering(knn_adj_matrix,
						knn_hgraph_dict, num_iterations = diffusion_iterations, verbose = False)

		graph_x = graph_diff_results['x']
		graph_error_vs_iteration.append(np.array([sweep_cut_classification_error(make_sweep_cut(graph_x[i],threshold = 0)) for i in range(diffusion_iterations)]))

		hypergraph_x = hypergraph_diff_results['x']
		hypergraph_error_vs_iteration.append(np.array([sweep_cut_classification_error(make_sweep_cut(hypergraph_x[i],threshold = 0)) for i in range(diffusion_iterations)]))

		# plot what'a happening at 5 steps in:
		if False:
			fig, ax = plt.subplots(1,2,figsize=(12, 6))
			ax[0].scatter(data_matrix[:,0],data_matrix[:,1],c=make_sweep_cut(graph_x[20], 0))
			ax[0].set_title('graph')

			ax[1].scatter(data_matrix[:,0],data_matrix[:,1],c=make_sweep_cut(hypergraph_x[20], 0))
			ax[1].set_title('hypergraph')
			plt.show()


	# two side-by-side figures
	if False:
		iterations = np.arange(0,diffusion_iterations)

		for idx,error_array in enumerate([np.array(graph_error_vs_iteration), np.array(hypergraph_error_vs_iteration)]):
			mean_error_by_iteration = np.mean(error_array,axis=0)
			std_error_by_iteration = np.std(error_array,axis=0)

			ax[idx].plot(iterations, mean_error_by_iteration)
			ax[idx].fill_between(iterations, mean_error_by_iteration-std_error_by_iteration, mean_error_by_iteration+std_error_by_iteration,alpha = 0.5)

		fig, ax = plt.subplots(1,2,figsize=(12, 6))
		ax[0].set_title('graph \n classification error versus iteration number')
		ax[1].set_title('hypergraph \n classification error versus iteration number')
		plt.show()

	# one figure with both trajectories
	if False:
		iterations = np.arange(0,diffusion_iterations)
		plt.figure(figsize=(12, 6))
		plotting_label = ['graph','hypergraph']
		plotting_color = ['b','o']
		for idx,error_array in enumerate([np.array(graph_error_vs_iteration), np.array(hypergraph_error_vs_iteration)]):
			mean_error_by_iteration = np.mean(error_array,axis=0)
			std_error_by_iteration = np.std(error_array,axis=0)

			plt.plot(iterations, mean_error_by_iteration,label = plotting_label[idx])
			plt.fill_between(iterations, mean_error_by_iteration-std_error_by_iteration, mean_error_by_iteration+std_error_by_iteration,alpha = 0.5)
		plt.legend()
		plt.title('Error versus iteration number'+titlestring)
		plt.show()

	return np.array(graph_error_vs_iteration), np.array(hypergraph_error_vs_iteration)

"""
MAIN

Run a cute little demo
"""
def main():
	# generate toy example
	_, data_matrix = generate_spirals(tightness = 3, num_rotations = 2.5, n_pts = 300, noise_level = 1.3,verbose = False)

	# build graph/hypergraph
	k = 5
	knn_adj_matrix = build_knn_graph(data_matrix,k)
	knn_hgraph_dict = build_knn_hypergraph(data_matrix,k)

	n = data_matrix.shape[1]

	# run some diffusions
	hypergraph_diff_results, graph_diff_results = semi_superivsed_knn_clustering(knn_adj_matrix,
																			 knn_hgraph_dict,
																			num_iterations = 25)
	# animate our results
	fig, ax = plt.subplots(1,2,figsize=(12, 6))
	ani = side_by_side_animation(fig,ax, [graph_diff_results,hypergraph_diff_results],data_matrix, animate_sweep_cut_in_plane)
	f = r"spirals_sweep_cut.gif" 
	writergif = matplotlib.animation.PillowWriter(fps=60) 
	ani.save(f, writer=writergif)
	return

if __name__ == '__main__':
	main()


