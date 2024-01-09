"""
Convert a python file containing a hypergraph into a matlab file.

"""
import pickle
import pdb
from semi_supervised_manifold_learning import small_example_knn_hypergraph


def build_example_python_file():
    data_matrix, knn_hgraph_dict, D, seeded_labels = small_example_knn_hypergraph()
    N = knn_hgraph_dict['n']
    R = knn_hgraph_dict['m']
    a = seeded_labels.flatten().tolist()
    incidence_list = knn_hgraph_dict['hypergraph']
    # need to convert all numpyints to ints. Doing this in a stupid slow way right now.
    incidence_list = [python_tuple(hyperedge) for hyperedge in incidence_list]
    parameter_homo_list = D.tolist()
    k = knn_hgraph_dict['degree']
    # pdb.set_trace()
    with open('small_example_hypergraph.pickle', 'wb') as handle:
       # pickle.dump([N,R,a,incidence_list,parameter_homo_list], handle)
       pickle.dump([N, R, k, a, incidence_list, parameter_homo_list], handle)
    return


# converts numpyints to ints
def python_tuple(numpy_tuple):
    return [val.item() for val in numpy_tuple]

build_example_python_file()