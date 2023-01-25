import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import pickle
import pandas as pd
import geopandas as gpd
import seaborn as sns

from itertools import chain
from collections import Counter
from tqdm import tqdm
from datetime import datetime

from diffusion_functions import *

""" 
PREPROCESSING METHODS	
"""

# Time-saving way to read in only the review columns that are necessary (i.e. skip the actual review text)
def load_relevant_review_data(input_directory='./data/yelp'):
	cols = ['user_id', 'business_id']
	data = []
	file_name = input_directory+'/yelp_academic_dataset_review.json'
	print('Loading reviews from json:')
	with open(file_name) as f:
		for line in tqdm(f):
			doc = json.loads(line)
			lst = [doc['user_id'], doc['business_id']]
			data.append(lst)
	return pd.DataFrame(data=data, columns=cols)

def build_restaurant_hypergraph(input_directory='./data/yelp',output_directory='./data/yelp'):
	"""
	This function runs *very* time-intensive pre-processing steps and save the results for future use.
	None of these are called in main, I assume that they've each already been run and that the relevant
	data structures are available in output_directory.
	"""
	business_df = pd.read_json(input_directory+'/yelp_academic_dataset_business.json', lines=True)
	# collect businesses which are restaurants
	# since yelp's categories are dumb, I decide something's a restaurant if it's categories contain "restaurants",
	# or "food". Using only "restaurants" results in ~52k businesses, while Gleich reports ~64k. Adding "Food" gets
	# 64k.
	restaurants_df = business_df[business_df['categories'].str.contains('Restaurants|Food', na=False)]
	restaurants_list = restaurants_df['business_id'].unique()

	# read only the review columns that are input_directory (i.e. skip the actual review text)
	review_df = load_relevant_review_data(input_directory='./data/yelp')
	# subselect only rows corresponding to restaurants
	# Note: this is faster than checking if doc['business_id'] is in restaurants_list while reading in above
	review_df = review_df.loc[review_df['business_id'].isin(restaurants_list)]

	# collect business tuples according to which review wrote them
	print('Compiling user hyperedges:')
	tqdm.pandas()
	hypergraph_groupby_obj = review_df.groupby('user_id')['business_id'].progress_apply(list)

	with open(input_directory+"/restaurant_hypergraph","w") as fp:
		json.dump([tuple(business_list) for business_list in hypergraph_groupby_obj], fp)

	return

def assemble_hypergraph(min_edge_cardinality = 3, input_directory='./data/yelp'):
	# load the constructed hypergraph
	with open(input_directory+"/restaurant_hypergraph","r") as fp:
		business_id_hypergraph = json.load(fp)
	# turn edgelists into edge tuples (so they'll be hashable later)
	business_id_hypergraph = [tuple(edgelist) for edgelist in business_id_hypergraph]

	# let's prune our restaurants hypergraph so only edges with sufficiently many restaurants remain
	pruned_hypergraph = [edgetup for edgetup in business_id_hypergraph if len(edgetup) >= min_edge_cardinality]
	# list of nodes that occur in hypergraph when all edges include sufficiently many nodes
	ARCH_BUSINESS_IDS = list(set(chain.from_iterable(pruned_hypergraph)))

	# extract the business data of the nodes in our hypergraph, ordered by node index 
	business_df = pd.read_json(input_directory+'/yelp_academic_dataset_business.json', lines=True)
	ARCH_BUSINESS_DF = business_df.copy(deep=True)
	ARCH_BUSINESS_DF = ARCH_BUSINESS_DF.loc[ARCH_BUSINESS_DF['business_id'].isin(ARCH_BUSINESS_IDS)]
	# Create the dictionary that defines the order for sorting
	sorterIndex = dict(zip(ARCH_BUSINESS_IDS, range(len(ARCH_BUSINESS_IDS))))
	ARCH_BUSINESS_DF['node_idx'] = ARCH_BUSINESS_DF['business_id'].map(sorterIndex)
	ARCH_BUSINESS_DF.sort_values(by = 'node_idx', inplace = True)

	"""
	We now have a dataframe that has all the data about our NODES in one place, including the
	indices by which they'll be indexed in our hypergraph. The last step is to create a version
	of our hypergraph where nodes are referenced by index rather than business_id.

	Rather than iterating through our pruned_hypergraph and matching IDs to indices,
	it's faster to reload the dataframe containing (user_id, business_id) pairs from reviews,
	match the business_ids to node_idxs, collect node_idxs by user via a groupby object, 
	and then fully reconstruct/prune the hypergraph.
	"""

	# read only the review columns that are necessary (i.e. skip the actual review text)
	review_df = load_relevant_review_data(input_directory='./data/yelp')
	# now we want to add the column containing the node index for each business
	review_df = review_df.merge(ARCH_BUSINESS_DF, on = 'business_id', how = 'inner')
	
	# collect business tuples according to which review wrote them
	print('Compiling user hyperedges:')
	tqdm.pandas()
	hypergraph_groupby_obj = review_df.groupby('user_id')['node_idx'].progress_apply(list)
	# turn edgelists into edge tuples (so they'll be hashable later)
	idx_hypergraph = [tuple(edgelist) for edgelist in hypergraph_groupby_obj]
	idx_hypergraph = [edgetup for edgetup in idx_hypergraph if len(edgetup)>=min_edge_cardinality]

	n = len(ARCH_BUSINESS_IDS)
	m = len(idx_hypergraph)

	# building degree array D:
	# turn the list of tuples of business_ids into one list of business_ids with repeats
	business_ID_multilist = chain.from_iterable(pruned_hypergraph)
	# count how many times each id occurs in this list
	degree_dict = Counter(business_ID_multilist)
	# create degree array in order of node indexing
	D = np.array([degree_dict[ARCH_BUSINESS_DF['business_id'].loc[ARCH_BUSINESS_DF['node_idx']==idx].item()] for idx in range(n)])

	print(f'n={n}, m={m}, D has shape {D.shape}')

	return {'n':n,'m':m,'D':D,'hypergraph':idx_hypergraph}, ARCH_BUSINESS_DF

"""
RUNNING DIFFUSIONS

A function that handles running the whole experiment, and various helper functions.
"""
def semi_superivsed_yelp_clustering(node_df, hgraph_dict, target_list,
		s_vector = None, hypergraph_objective = diffusion_functions['infinity'],
		num_rand_seeds = 30, step_size = 1, num_iterations = 10, verbose = True):

	# let's extract some parameters
	n = hgraph_dict['n']
	m = hgraph_dict['m']
	D = hgraph_dict['D']
	hypergraph = hgraph_dict['hypergraph']
	
	# We'll use one of two schemes to incorporate semi-supervision: we'll choose a small number of
	# random seeds to be "true" labels revealed, then we'll either
	# provide that as the initial vector, or
	# provide it as the s vector
	if s_vector=='set_by_seeds':
		# create an initial pt with num_rand_seeds randomly chosen true labels
		s_vector = create_random_seed_vector(target_list, num_rand_seeds, node_df)
		x0 = np.random.normal(size=(n,1))
	else:
		# create an initial pt with num_rand_seeds randomly chosen true labels
		x0 = create_random_seed_vector(target_list, num_rand_seeds, node_df)
		# Deal with s
		if np.all(s_vector==None):
			s_vector = np.full(shape=(n,1),fill_value = 0)

	# Run our diffusion
	x, y ,fx = diffusion(x0, n, m, D, hypergraph, weights=None, func=hypergraph_objective, 
						 s=s_vector, h=step_size, T=num_iterations, verbose=verbose)

	# Define method for later assessing the cut function corresponding to this diffusion on any given vector
	W, sparse_h, rank = compute_hypergraph_matrices(n, m, hypergraph, weights=None)
	hypergraph_cut_ojbective = lambda vec: eval_hypergraph_cut_fn(hypergraph_objective, vec, s_vector, sparse_h, rank, W, D)

	return {'x':x,'y':y,'fx':fx}

# sample num_rand_seeds restaurants from target_list, and return a {0,1}^n indicator vector
def create_random_seed_vector(target_list, num_rand_seeds, node_df):
	random_seed_IDs = np.random.choice(target_list,size = num_rand_seeds)
	return list_to_vec(node_df, random_seed_IDs)

# given a list of restaurants (in our hypergraph) create a vector whose entries are all 0 except for those rstrnts
def list_to_vec(node_df, restaurants_list):
	idx_array = node_df['node_idx'].loc[node_df['business_id'].isin(restaurants_list)].unique()
	
	vec = np.full(shape=(len(node_df),1),fill_value=0)
	vec[idx_array] = 1
	return vec
   
# given a vector in {0,1}^n, return a list of restaurants corresponding to nonzero entries
def vec_to_list(node_df, vec):
	idxs = np.argwhere(vec==1)
	# turn ( n x num nonzero ) array into a list
	idxs = list(np.reshape(idxs,newshape=idxs.size))
	
	return node_df['business_id'].loc[node_df['node_idx'].isin(idxs)].unique()

"""
ASSESSING PERFORMANCE

"""

def find_sweep_cut(node_df, node_vec, threshold):
	restaurants_above_threshold = vec_to_list(node_df, node_vec > threshold)

	return node_df['business_id'].loc[node_df['business_id'].isin(restaurants_above_threshold)]


"""
PLOTTING

Functions for visualization.
"""

# ALWAYS entered as (Lattitude, Longitude)
known_regions = {'New Orleans':[(29.8, 30.1),(-90.3,-89.9)],'USA':[(25,50),(-135,-70)]}

def get_coordinates(target_region):
	if target_region in known_regions.keys():
		coors = known_regions[target_region]
	else:
		raise Exception('Target city\'s geographic coordinates/google maps photo not yet'
				'added to manually-specified dictionary of regional boundaries.')
	return coors

def get_target_geo_df(target_city, business_df):
	coors = get_coordinates(target_city)
	lat_min, lat_max = coors[0]
	lon_min, lon_max = coors[1]

	return business_df.loc[(business_df["longitude"]>lon_min) &
					(business_df["longitude"]<lon_max) &
					(business_df["latitude"]>lat_min) &
					(business_df["latitude"]<lat_max)]

def scatter_pts_by_list(df, target_list, target_region = 'USA', plot_sattelite_photo = False):
	#subset by our target list
	target_df = df.loc[df['business_id'].isin(target_list)]
	scatter_on_map(df, get_coordinates(target_city), colorvals = None, sizevals = None, titlestring=None)

	return

# SHOULD THIS BE DEPRECATED?
def scatter_pts_in_region(df, target_city, plot_sattelite_photo = False):
	#subset for target city
	target_geo_df = get_target_geo_df(target_city, df)
	target_geo_df['labeled as target city?'] = np.where(target_geo_df['city']==target_city,True,False)

	scatter_on_map(target_geo_df, get_coordinates(target_city), colorvals = 'labeled as target city?', sizevals = None, titlestring=None)

	if plot_sattelite_photo:
		# Compare with google earth map
		img = mpimg.imread('./data/yelp/'+str(target_city)+'.png')
		fig, ax = plt.subplots(figsize=(8,6))
		ax.imshow(img)
		ax.set_title('Satellite image of target city, for reference.')
		plt.show()
	return

# colorvals and sizevals can either be an array or a column in df
def scatter_on_map(df, coordinates, colorvals = None, sizevals = None, titlestring=None):
	lat_min, lat_max = coordinates[0]
	lon_min, lon_max = coordinates[1]

	# plot with colors/sizes determined by diffusion value at the target time
	fig, ax = plt.subplots(figsize=(8,6))
	sns.scatterplot(x='longitude',y='latitude',data=df,hue=colorvals, size = sizevals)
	ax.grid()
	ax.set_xlim([lon_min,lon_max])
	ax.set_ylim([lat_min, lat_max])
	plt.title(titlestring)
	plt.show()
	return

def scatter_diffusion_result(node_df, target_region, diffusion_results, time):
	coors = get_coordinates(target_region)

	# get our values of interest
	x = diffusion_results['x']
	timeslice_values = np.reshape(x[time,:],newshape=x.shape[1])
	size_scale = np.divide(np.abs(timeslice_values)+1,np.max(np.abs(timeslice_values)+1))

	# plot with colors/sizes determined by diffusion value at the target time
	titlestring = f'Vertices used in diffusion, colored by diffusion values at time {time} \n Sizes chosen to make it easier to spot pts w large values'
	scatter_on_map(node_df, coors, colorvals = timeslice_values, sizevals = size_scale, titlestring=None)
	
	# compare w scatter from whole plot
	coors = known_regions['USA']
	scatter_on_map(node_df, coors, colorvals = timeslice_values, sizevals = size_scale, titlestring=None)
	return

def get_target_restaurant_IDs(target_city, business_df, target_by):
	if target_by == 'geographic_coordinates':
		df = get_target_geo_df(target_city, business_df)
	elif target_by == 'city_value':
		df = business_df.loc[business_df["city"]==target_city]
	return df

def get_hgraph_stats(hgraph_list):
	# Get a picture of the size of the hyperedges in the FULL hypergraph
	n = len(set(chain.from_iterable(hgraph_list)))
	m = len(hgraph_list)
	edge_cardinalities = [len(edge) for edge in hgraph_list]
	quartiles = [np.percentile(edge_cardinalities, q) for q in [25, 50, 75]]

	print(f'Full hypergraph has {m} hyperedges on {n} nodes. \n'
		  f'     {edge_cardinalities.count(1)/m} = fraction of hyperedges of cardinality 1.\n'
		  f'     min, max cardinality = {min(edge_cardinalities),max(edge_cardinalities)}\n'
		  f'     mean cardinality = {np.mean(edge_cardinalities)}\n'
		  f'     std = {np.std(edge_cardinalities)}\n'
		  f'     quartiles = {quartiles}'
		 )

	plt.hist(edge_cardinalities,range =(min(edge_cardinalities),np.mean(edge_cardinalities)+np.std(edge_cardinalities)))
	plt.title('Histogram for edges of cardinality [min, mean+std]')
	plt.show()
	return
"""
MAIN

And helpers.
"""

def ID_edgetup_to_idxtup(ARCH_BUSINESS_IDS,edgetup):
	return tuple([ARCH_BUSINESS_IDS.index(restaurant) for restaurant in edgetup])

def main(target_city = 'New Orleans', s_vector = None, hypergraph_objective = diffusion_functions['infinity'], min_edge_cardinality = 10,
		num_rand_seeds = 30, step_size = 1e-2, num_iterations = 300, verbose = True, save_experiment = True, plotting = False):
	input_directory = './data/yelp'
	output_directory = './data/yelp'

	saved_args = locals()

	input_directory='./data/yelp'
	with open(input_directory+"/restaurant_hypergraph","r") as fp:
			business_id_hypergraph = json.load(fp)

	pruned_hypergraph_dict, node_df = assemble_hypergraph(min_edge_cardinality)

	if plotting:
		scatter_pts_in_region(node_df, target_city, plot_sattelite_photo = True)
		get_hgraph_stats(pruned_hypergraph_dict['hypergraph'])
	"""
	If s_vector is None, then s_vector will be treated as 0 and the random seed labels will be drawn to create x0. 
	If s_vector is 'set_by_seeds', then x0 will be random normal noise and s_vector will be a {0,1}^n vector created
	using random seed labels.
	"""
	target_df = get_target_restaurant_IDs('New Orleans', node_df, target_by = 'city_value')

	diffusion_results = semi_superivsed_yelp_clustering(node_df, pruned_hypergraph_dict,
		target_list = target_df['business_id'].unique(),
		s_vector = s_vector, hypergraph_objective = hypergraph_objective,
		num_rand_seeds = num_rand_seeds, step_size = step_size, num_iterations = num_iterations, verbose = verbose)
	
	if plotting:
		scatter_diffusion_result(node_df, target_city, diffusion_results, time=0)

	if save_experiment:
		folder_name = str(datetime.now())
		save_path = os.path.join(output_directory, folder_name)
		os.makedirs(save_path)
		with open(save_path+"/diffusion_results.pkl","wb") as fp:
			pickle.dump(diffusion_results, fp)

		saved_args['node_df'] = node_df
		with open(save_path+"/experiment_parameters.pkl","wb") as fp:
			# we can't save the objective function
			del saved_args['hypergraph_objective']
			pickle.dump(saved_args, fp)

	return

if __name__ == '__main__':
	main()