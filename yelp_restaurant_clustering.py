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

	# read only the review columns that are necessary (i.e. skip the actual review text)
	review_df = load_relevant_review_data(input_dictionary='./data/yelp')
	# subselect only rows corresponding to restaurants
	# Note: this is faster than checking if doc['business_id'] is in restaurants_list while reading in above
	review_df = review_df.loc[df['business_id'].isin(restaurants_list)]

	# collect business tuples according to which review wrote them
	print('Compiling user hyperedges:')
	tqdm.pandas()
	hypergraph_groupby_obj = review_df.groupby('user_id')['business_id'].progress_apply(list)

	# let's prune our restaurants hypergraph so only edges with at least 3 restaurants remain
	min_degree_cutoff = 50
	pruned_hypergraph = [edgetup for edgetup in restaurants_hypergraph if len(edgetup) >= min_degree_cutoff]

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
	ARCH_BUSINESS_DF = business_df.loc[business_df['business_id'].isin(ARCH_BUSINESS_IDS)]
	# Create the dictionary that defines the order for sorting
	sorterIndex = dict(zip(ARCH_BUSINESS_IDS, range(len(ARCH_BUSINESS_IDS))))
	ARCH_BUSINESS_DF['node_idx'] = ARCH_BUSINESS_DF['business_id'].map(sorterIndex)
	ARCH_BUSINESS_DF.sort_values(by = 'node_idx', inplace = True)

	"""
	We now have a dataframe that has all the data about our nodes in one place, including the
	indices by which they'll be indexed in our hypergraph. The last step is to create a version
	of our hypergraph where nodes are referenced by index rather than business_id.

	Rather than iterating through our business_id_hypergraph and matching IDs to indices,
	it's faster to reload the dataframe containing (user_id, business_id) pairs from reviews,
	match the business_ids to node_idxs, and then regroup hyperedges by user_id.
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
def semi_superivsed_yelp_clustering(ARCH_BUSINESS_IDS, hgraph_dict, target_list,
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
		random_seed_IDs = np.random.choice(target_list,size = num_rand_seeds)
		s_vector = list_to_vec(ARCH_BUSINESS_IDS, random_seed_IDs)
		
		x0 = np.random.normal(size=(n,1))
	else:
		# create an initial pt with num_rand_seeds randomly chosen true labels
		random_seed_IDs = np.random.choice(target_list,size = num_rand_seeds)
		x0 = list_to_vec(ARCH_BUSINESS_IDS, random_seed_IDs)
		# Deal with s
		if np.all(s_vector==None):
			s_vector = np.full(shape=(n,1),fill_value = 0)

	# Run our diffusion
	x, y ,fx, sparse_h, rank, W = diffusion(x0, n, m, D, hypergraph, weights=None, func=hypergraph_objective, 
						 s=s_vector, h=step_size, T=num_iterations, return_constructions = True, verbose=verbose)

	# Define method for later assessing the cut function corresponding to this diffusion on any given vector
	hypergraph_cut_ojbective = lambda vec: eval_hypergraph_cut_fn(hypergraph_objective, vec, s_vector, sparse_h, rank, W, D)

	return {'x':x,'y':y,'fx':fx}

# given a list of restaurants (in our hypergraph) create a vector whose entries are all 0 except for those rstrnts
def list_to_vec(ARCH_BUSINESS_IDS, restaurants_list):
	idxs = [ARCH_BUSINESS_IDS.index(restaurant) for restaurant in restaurants_list]
	
	vec = np.full(shape=(len(ARCH_BUSINESS_IDS),1),fill_value=0)
	vec[idxs] = 1
	return vec
   
# given a vector in {0,1}^n, return a list of restaurants corresponding to nonzero entries
def vec_to_list(ARCH_BUSINESS_IDS, vec):
	idxs = np.argwhere(vec==1)
	# turn ( n x num nonzero ) array into a list
	idxs = list(np.reshape(idxs,newshape=idxs.size))
	
	return [ARCH_BUSINESS_IDS[idx] for idx in idxs]
"""
PLOTTING

Functions for visualization.
"""

# ALWAYS entered as (Lattitude, Longitude)
known_cities = {'New Orleans':[(29.8, 30.1),(-90.3,-89.9)],'US':[(25,50),(135,-70)]}

def get_target_geo_df(target_city, business_df):
	if target_city in known_cities.keys():
		coors = known_cities[target_city]
	else:
		raise Exception('Target city geographic coordinates/google maps photo not yet added to manually-specified dictionary.')

	lat_min, lat_max = coors[0]
	lon_min, lon_max = coors[1]

	return business_df.loc[(business_df["longitude"]>lon_min) &
					(business_df["longitude"]<lon_max) &
					(business_df["latitude"]>lat_min) &
					(business_df["latitude"]<lat_max)]

def scatter_pts_on_map(business_df, target_city,plot_sattelite_photo = False):
	#subset for target city
	target_geo_df = get_target_geo_df(target_city, business_df)

	target_geo_df['labeled as target city?'] = np.where(target_geo_df['city']==target_city,True,False)
	fig, ax = plt.subplots(figsize=(8,6))
	sns.scatterplot(x='longitude',y='latitude',data=target_geo_df,hue='labeled as target city?')
	ax.grid()
	plt.show()

	if plot_sattelite_photo:
		# Compare with google earth map
		img = mpimg.imread('./data/yelp/'+str(target_city)+'.png')
		fig, ax = plt.subplots(figsize=(8,6))
		ax.imshow(img)
		ax.set_title('Satellite image of target city, for reference.')
		plt.show()

	return

def scatter_diffusion_result(ARCH_BUSINESS_IDS, business_df, target_locale, diffusion_results, time):
	# define the geographic scope we'll plot
	coors = known_cities[target_locale]
	lat_min, lat_max = coors[0]
	lon_min, lon_max = coors[1]

	# extract the coordinates of the nodes in our hypergraph, ordered by node index 
	ARCH_BUSINESS_DF = business_df.loc[business_df['business_id'].isin(ARCH_BUSINESS_IDS)]
	# Create the dictionary that defines the order for sorting
	sorterIndex = dict(zip(ARCH_BUSINESS_IDS, range(len(ARCH_BUSINESS_IDS))))
	ARCH_BUSINESS_DF['arch_idx'] = ARCH_BUSINESS_DF['business_id'].map(sorterIndex)
	ARCH_BUSINESS_DF.sort_values(by = 'arch_idx', inplace = True)

	# get our values of interest
	x = diffusion_results['x']
	timeslice_values = np.reshape(x[time,:],newshape=x.shape[1])
	size_scale = np.divide(np.abs(timeslice_values)+1,np.max(np.abs(timeslice_values)+1))

	# plot with colors/sizes determined by diffusion value at the target time
	fig, ax = plt.subplots(figsize=(8,6))
	sns.scatterplot(x='longitude',y='latitude',data=ARCH_BUSINESS_DF,hue=timeslice_values, size = size_scale)
	ax.grid()
	ax.set_xlim([lon_min,lon_max])
	ax.set_ylim([lat_min, lat_max])
	plt.title(f'Vertices used in diffusion, colored by diffusion values at time {time} \n Sizes chosen to make it easier to spot pts w large values')
	plt.show()

	# compare w scatter from whole plot
	fig, ax = plt.subplots(figsize=(8,6))
	sns.scatterplot(x='longitude',y='latitude',data=business_df)
	ax.grid()
	ax.set_xlim([lon_min,lon_max])
	ax.set_ylim([lat_min, lat_max])
	plt.title('All restaurants in dataset in this region \n All points same size/color')
	plt.show()

	return

"""
MAIN

And helpers.
"""

def ID_edgetup_to_idxtup(ARCH_BUSINESS_IDS,edgetup):
	return tuple([ARCH_BUSINESS_IDS.index(restaurant) for restaurant in edgetup])

def main(target_city = 'New Orleans', s_vector = None, hypergraph_objective = diffusion_functions['infinity'], num_rand_seeds = 30, 
			step_size = 1e-3, num_iterations = 3, verbose = True, save_experiment = True, plotting = True ):
	input_directory = './data/yelp'
	output_directory = './data/yelp'

	saved_args = locals()

	business_df = pd.read_json(input_directory+'/yelp_academic_dataset_business.json', lines=True)
	# sub-select only those businesses which are restaurants.
	# since yelp's categories are dumb, I decide something's a restaurant if it's categories contain "restaurants",
	# or "food". Using only "restaurants" results in ~52k businesses, while Gleich reports ~64k. Adding "Food" gets
	# 64k.
	business_df = business_df.loc[business_df['categories'].str.contains('Restaurants|Food', na=False)]

	if False:
		scatter_pts_on_map(business_df, target_city,plot_sattelite_photo = True)

	# Collect labesl for our target city
	target_database_df = business_df.loc[business_df['city']==target_city]
	database_target_list = list(target_database_df['business_id'])

	target_geo_df = get_target_geo_df(target_city, business_df)
	geo_target_list = list(target_geo_df['business_id'])

	hgraph_dict, ARCH_BUSINESS_IDS = assemble_hypergraph_from_file()

	geo_targets_in_hypergraph = list(set(ARCH_BUSINESS_IDS).intersection(set(geo_target_list)))
	database_targets_in_hypergraph = list(set(ARCH_BUSINESS_IDS).intersection(set(database_target_list)))

	"""
	If s_vector is None, then s_vector will be treated as 0 and the random seed labels will be drawn to create x0. 
	If s_vector is 'set_by_seeds', then x0 will be random normal noise and s_vector will be a {0,1}^n vector created
	using random seed labels.
	"""
	diffusion_results = semi_superivsed_yelp_clustering(ARCH_BUSINESS_IDS, hgraph_dict, geo_targets_in_hypergraph,
								s_vector, hypergraph_objective, num_rand_seeds, step_size, num_iterations, verbose)

	if plotting:
		scatter_diffusion_result(ARCH_BUSINESS_IDS, business_df, target_city, diffusion_results, time=0)

	if save_experiment:
		folder_name = str(datetime.now())
		save_path = os.path.join(output_directory, folder_name)
		os.makedirs(save_path)
		with open(save_path+"/diffusion_results.pkl","wb") as fp:
			pickle.dump(diffusion_results, fp)
		with open(save_path+"/experiment_parameters.pkl","wb") as fp:
			# we can't save the objective function
			del saved_args['hypergraph_objective']
			pickle.dump(saved_args, fp)

	return

if __name__ == '__main__':
	main()