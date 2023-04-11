from diffusion_functions import *
from yelp_restaurant_clustering import *
import yelp_restaurant_clustering
import os

"""
	FILE FOR RUNNING A BUNCH OF EXPENSIVE EXPERIMENTS
	Total est time: 9s x 900 iterations = 2.25 hours
"""
def print_message(counter):
	print(f'------------- \n Running experiment {counter} \n-------------')
	return

def main():
	# first need to create datastructure that will be accessed in future experiments:
	restaurant_hypergraph_filename = './data/yelp/restaurant_hypergraph'
	if not os.path.exists(restaurant_hypergraph_filename):
		print('Building restaurant hypergraph:\n')
		build_restaurant_hypergraph(input_directory='./data/yelp',output_directory='./data/yelp')
	else:
		print('Using existing restaurant hypergraph file.\n')

	experiment_counter = 1

	"""
		EXPERIMENT 1: random seed as x0, min_edge_cardinality = 10

		est. time/iteration = 9s
	"""
	print_message(experiment_counter)
	yelp_restaurant_clustering.main(target_city = 'New Orleans', s_vector = None, hypergraph_objective = diffusion_functions['infinity'], min_edge_cardinality = 10,
			num_rand_seeds = 30, step_size = 1e-2, num_iterations = 300, verbose = True, save_experiment = True, plotting = False)
	experiment_counter+=1

	"""
		EXPERIMENT 2: random seed as s vector, min_edge_cardinality = 10

		s vector might require smaller step size?

		est. time/iteration = 9s
	"""
	print_message(experiment_counter)
	yelp_restaurant_clustering.main(target_city = 'New Orleans', s_vector = 'set_by_seeds', hypergraph_objective = diffusion_functions['infinity'], min_edge_cardinality = 10,
			num_rand_seeds = 30, step_size = 1e-2, num_iterations = 1, verbose = True, save_experiment = True, plotting = False)
	experiment_counter+=1
	"""
		EXPERIMENT 3: random seed as x0, bigger step size, min_edge_cardinality = 10

		est. time/iteration = 9s
	"""
	print_message(experiment_counter)
	yelp_restaurant_clustering.main(target_city = 'New Orleans', s_vector = None, hypergraph_objective = diffusion_functions['infinity'], min_edge_cardinality = 10,
			num_rand_seeds = 30, step_size = 1e-1, num_iterations = 300, verbose = True, save_experiment = True, plotting = False)
	experiment_counter+=1

	"""
		EXPERIMENT 4: random seed as x0, biggest step size, min_edge_cardinality = 10

		est. time/iteration = 9s
	"""
	print_message(experiment_counter)
	yelp_restaurant_clustering.main(target_city = 'New Orleans', s_vector = None, hypergraph_objective = diffusion_functions['infinity'], min_edge_cardinality = 10,
			num_rand_seeds = 30, step_size = 1, num_iterations = 300, verbose = True, save_experiment = True, plotting = False)

if __name__ == '__main__':
	main()
