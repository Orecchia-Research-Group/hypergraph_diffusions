"""
Convert a python file containing a hypergraph into a matlab file.

"""
import os
import sys
import pickle

import numpy as np

import reading
from semi_supervised_manifold_learning import small_example_knn_hypergraph


def prepare_pickle_file(filename):
    # data_matrix, knn_hgraph_dict, D, seeded_labels = small_example_knn_hypergraph()
    basename, ext = os.path.splitext(filename)
    labelname = f'{basename}.label'
    picklename = f'{basename}.pickle'
    N, R, _, hypergraph, weights, _, _ = reading.read_hypergraph(filename)
    if weights is None:
        weights = [1] * R
    incidence_list = []
    parameter_homo_list = []
    R = 0
    for i, hyperedge in enumerate(hypergraph):
        if len(hyperedge) == 0:
            continue
        R += 1
        incidence_list.append(hyperedge)
        parameter_homo_list.append(weights[i])
    assert R == len(incidence_list)
    assert R == len(parameter_homo_list)
    _, labels = reading.read_labels(labelname)
    a = np.zeros_like(labels)
    for li, label in enumerate(np.unique(labels)):
        a[labels == label] = li
    a = a.tolist()
    k = [len(hyperedge) for hyperedge in incidence_list]
    # N = knn_hgraph_dict['n']
    # R = knn_hgraph_dict['m']
    # a = seeded_labels.flatten().tolist()
    # incidence_list = knn_hgraph_dict['hypergraph']
    # need to convert all numpyints to ints. Doing this in a stupid slow way right now.
    incidence_list = [list(hyperedge) for hyperedge in incidence_list]
    # parameter_homo_list = D.tolist()
    # k = knn_hgraph_dict['degree']
    # pdb.set_trace()
    with open(picklename, 'wb') as handle:
       # pickle.dump([N,R,a,incidence_list,parameter_homo_list], handle)
       pickle.dump([N, R, k, a, incidence_list, parameter_homo_list], handle)
    return


def main():
    prepare_pickle_file(sys.argv[1])


if __name__ == '__main__':
    main()
