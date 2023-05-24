import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from libpysal.weights import KNN
from libpysal.cg import KDTree
import networkx as nx
import smallestenclosingcircle as sec

from data.hypergraphs2hmetis import write_hypergraph
from diffusion_functions import diffusion


def generate_hypergraph(n, k):
    points = np.random.rand(n, 2)
    kd = KDTree(points)
    wnn = KNN(kd, k)
    graph = set()
    hypergraph = set()
    degree = defaultdict(int)
    hdegree = defaultdict(int)
    hcircle = defaultdict(lambda: ((0, 0), 0))
    for i in range(n):
        hdegree[i] += 1
        for j in wnn[i]:
            new_radius = np.linalg.norm(points[i] - points[j])
            hdegree[j] += 1
            e = (min(i, j), max(i, j))
            if e in graph:
                continue
            graph.add(e)
            degree[i] += 1
            degree[j] += 1
        hyperedge = (i, ) + tuple(wnn[i].keys())
        hcircle[hyperedge] = sec.make_circle(points[list(hyperedge)])
        hypergraph.add(hyperedge)
    return points, degree, sorted(graph), hdegree, sorted(hypergraph), hcircle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, help='Number of nodes')
    parser.add_argument('-k', default=5, type=int, help='Number of neighbors')
    parser.add_argument('-d', '--directory', default='data', type=str, help='Directory to save (hyper)graph')
    parser.add_argument('-s', '--seed', default=None, type=int, help='Random seed')
    parser.add_argument('-l', '--lamda', type=float, default=0.15, help='Lamda value for PPR')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)
    points, degree, graph, hdegree, hypergraph, hcircle = generate_hypergraph(args.n, args.k)

    graph_filename = os.path.join(args.directory, f'geometric_graph_{args.n}.hmetis')
    write_hypergraph(graph_filename, args.n, len(graph), degree, graph)

    hypergraph_filename = os.path.join(args.directory, f'geometric_hypergraph_{args.n}.hmetis')
    write_hypergraph(hypergraph_filename, args.n, len(hypergraph), hdegree, hypergraph)

    position_filename = os.path.join(args.directory, f'geometric_graph_{args.n}_positions.txt')
    with open(position_filename, 'w') as f:
        for x, y in points:
            print(x, y, file=f)

    radius_filename = os.path.join(args.directory, f'geometric_graph_{args.n}_hcircle.txt')
    with open(radius_filename, 'w') as f:
        for i in sorted(hcircle):
            x, y, r = hcircle[i]
            print(x, y, r, file=f)

    G = nx.Graph(graph)
    plt.figure()
    nx.draw(G, with_labels=True, pos=points, font_color='w')
    graph_limits = plt.axis('equal')
    graph_draw_filename = os.path.join(args.directory, f'geometric_graph_{args.n}.png')
    plt.savefig(graph_draw_filename, dpi=500)

    fig, ax = plt.subplots()
    nx.draw(G, with_labels=True, pos=points, edgelist=[], font_color='w', node_color='b')
    for i, (x, y, r) in hcircle.items():
        circle = plt.Circle((x, y), 1.2 * r, color='darkolivegreen', alpha=0.2)
        ax.add_patch(circle)
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    plt.axis(graph_limits)
    # plt.axis('equal')
    hypergraph_draw_filename = os.path.join(args.directory, f'geometric_hypergraph_{args.n}.png')
    plt.savefig(hypergraph_draw_filename, bbox_inches='tight', dpi=500)

    x0 = np.zeros((args.n, 1))
    s = np.zeros((args.n, 1))
    s[0, 0] = 1
    iteration_times, x_graph, _, fx = diffusion(x0, args.n, len(graph), [degree[i] for i in sorted(degree)],
                                          graph, weights=None, s=s, lamda=args.lamda,
                                          h=1, T=100, regularizer='degree')

    iteration_times, x_hyper, _, fx = diffusion(x0, args.n, len(hypergraph), [hdegree[i] for i in sorted(hdegree)],
                                          hypergraph, weights=None, s=s, lamda=args.lamda,
                                          h=1, T=400, regularizer='degree')
    final_x = x_hyper.sum(axis=0) / x_hyper.shape[0]

    global_s = np.random.random((args.n, 1))
    global_s *= np.linalg.norm(s) / np.linalg.norm(global_s)
    iteration_times, x_graph_PR, _, fx = diffusion(x0, args.n, len(graph), [degree[i] for i in sorted(degree)],
                                                graph, weights=None, s=global_s, lamda=args.lamda,
                                                h=1, T=100, regularizer='degree')

    iteration_times, x_hyper_PR, _, fx = diffusion(x0, args.n, len(hypergraph), [hdegree[i] for i in sorted(hdegree)],
                                                hypergraph, weights=None, s=global_s, lamda=args.lamda,
                                                h=1, T=400, regularizer='degree')
    final_x_PR = x_hyper_PR.sum(axis=0) / x_hyper_PR.shape[0]

    x_graph -= x_graph.min()
    final_x -= final_x.min()
    x_graph_PR -= x_graph_PR.min()
    final_x_PR -= final_x_PR.min()
    max_value = max(x_graph.max(), final_x.max())
    max_PR_value = max(x_graph_PR.max(), final_x_PR.max())

    cmap = mpl.cm.copper
    plt.figure()
    nx.draw(G, with_labels=True, node_color=x_graph[-1].reshape(-1), pos=points, font_color='w', cmap=cmap, vmin=0, vmax=max_value)
    diff_graph_filename = os.path.join(args.directory, f'geometric_graph_{args.n}_PPR.png')
    plt.axis(graph_limits)
    plt.savefig(diff_graph_filename, dpi=500)

    fig, ax = plt.subplots()
    nx.draw(G, with_labels=True, node_color=final_x.reshape(-1), edgelist=[], pos=points,
            font_color='w', cmap=cmap, node_size=200, font_size=10, vmin=0, vmax=max_value)
    for i, (x, y, r) in hcircle.items():
        circle = plt.Circle((x, y), 1.2 * r, color='darkolivegreen', alpha=0.05)
        ax.add_patch(circle)
    diff_graph_filename = os.path.join(args.directory, f'geometric_hypergraph_{args.n}_PPR.png')
    plt.axis(graph_limits)
    plt.savefig(diff_graph_filename,  bbox_inches='tight', dpi=500)

    plt.figure()
    nx.draw(G, with_labels=True, node_color=x_graph_PR[-1].reshape(-1), pos=points, font_color='w', cmap=cmap, vmin=0,
            vmax=max_PR_value)
    diff_graph_filename = os.path.join(args.directory, f'geometric_graph_{args.n}_PR.png')
    plt.axis(graph_limits)
    plt.savefig(diff_graph_filename, dpi=500)

    fig, ax = plt.subplots()
    nx.draw(G, with_labels=True, node_color=final_x_PR.reshape(-1), edgelist=[], pos=points,
            font_color='w', cmap=cmap, node_size=200, font_size=10, vmin=0, vmax=max_PR_value)
    for i, (x, y, r) in hcircle.items():
        circle = plt.Circle((x, y), 1.2 * r, color='darkolivegreen', alpha=0.05)
        ax.add_patch(circle)
    diff_graph_filename = os.path.join(args.directory, f'geometric_hypergraph_{args.n}_PR.png')
    plt.axis(graph_limits)
    plt.savefig(diff_graph_filename, bbox_inches='tight', dpi=500)


if __name__ == '__main__':
    main()