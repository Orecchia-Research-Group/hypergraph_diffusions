import sys
import os
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import networkx as nx


def read_graph(filename: str) -> Tuple[nx.Graph, List[int]]:
    edges = pd.read_csv(filename, index_col=None, comment='%', header=None, sep=r'\s+')
    if len(edges.columns) > 2:
        del edges[2]
    left_side_nodes = edges[0].max()
    edges[1] += left_side_nodes
    G = nx.from_edgelist(edges.values)
    # print(len(G))
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    # print(len(G))
    left_side_nodes = [node for node in range(1, left_side_nodes+1) if node in G]
    # print(left_side_nodes[-1], len(left_side_nodes))
    return G, left_side_nodes


def write_hypergraph(n: int, m: int, hypergraph: Dict[Tuple, int], filename: str) -> None:
    degrees = defaultdict(int)
    with open(filename, 'w') as f:
        print(m, n, 11, file=f)
        for nodes, w in hypergraph.items():
            for v in nodes:
                degrees[v] += w
            print(w, *nodes, file=f)
        for v in sorted(degrees.keys()):
            print(degrees[v], file=f)


def hypergraph_from_bipartite(bipartite: nx.Graph, left_side_nodes: List[int]) -> Tuple[int, int, Dict[Tuple, int]]:
    if not nx.is_bipartite(bipartite):
        raise ValueError('Graph is not bipartite.')
    translation_dict = {node: i+1 for i, node in enumerate(left_side_nodes)}
    # print(translation_dict)
    n = len(translation_dict)
    # Keep the largest connected component
    hypergraph = defaultdict(int)
    for node in bipartite:
        if node <= left_side_nodes[-1]:
            continue
        #print(node, len(list(bipartite.neighbors(node))), max(bipartite.neighbors(node)))
        #print(translation_dict[max(bipartite.neighbors(node))])
        hypergraph[tuple(sorted([translation_dict[v] for v in bipartite.neighbors(node)]))] += 1
    m = len(hypergraph)
    return n, m, hypergraph


def process_hypegraph(in_filename, out_filename):
    """Read in a bipartite graph from `in_filename`,
    convert it to a hypergraph and write it in out_filename
    """
    bipartite, left_side_nodes = read_graph(in_filename)
    n, m, hypergraph = hypergraph_from_bipartite(bipartite, left_side_nodes)
    print(n, m)
    write_hypergraph(n, m, hypergraph, out_filename)


def main():
    process_hypegraph(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()