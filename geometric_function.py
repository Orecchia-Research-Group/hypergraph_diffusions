import os
import argparse
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy import sparse

from diffusion_functions import make_clique_regularizer
import reading


def parse_args():
    parser = argparse.ArgumentParser(description='Run geometric random walks on cliques.')
    parser.add_argument('-g', '--hypergraph', help='Filename of hypergraph to use.', type=str, required=True)
    parser.add_argument('--lamda', '-l', help='Parameter used in personalized pagerank.', type=float, default=0)
    parser.add_argument('-r', '--random-seed', help='Random seed to use for initialization.', type=int, default=None)
    parser.add_argument('-d', '--dimensions', help='Number of embedding dimensions.', type=int, default=2)
    parser.add_argument('-T', '--iterations', help='Maximum iterations for diffusion.', type=int, default=20)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    graph_name = os.path.basename(os.path.splitext(args.hypergraph)[0])
    n, m, D, hypergraph, weights, center_id, hypergraph_node_weights = reading.read_hypergraph(args.hypergraph)
    D = np.array(D)
    D_inv = sparse.diags(1 / D)
    if weights is None:
        weights = defaultdict(lambda: 1)
    _, L = make_clique_regularizer(n, m, D, hypergraph, weights)
    np.random.seed(args.random_seed)
    vs = np.random.choice(np.arange(n), size=args.dimensions, replace=False)
    s = np.zeros((n, args.dimensions))
    for d, v in enumerate(vs):
        s[v, d] = 1
    ds = (s.T / D).T
    coeff = 1 / (args.lamda + 2)
    result = coeff * ds
    mat_mult = (sparse.eye(n) - (D_inv * L) / 2)
    start = datetime.now()
    print(f'{"T":3s} {"Time (s)":8s}')
    for t in range(args.iterations):
        elapsed = (datetime.now() - start).total_seconds()
        coeff *= 2 / (args.lamda + 2)
        ds = mat_mult @ ds
        result += coeff * ds
        print(f'{t+1:3d} {elapsed:8.3f}')
    outfilename = os.path.join('paper_results', f'power_{graph_name}.pickle')
    with open(outfilename, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    main()