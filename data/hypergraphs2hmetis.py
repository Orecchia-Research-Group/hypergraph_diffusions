import os
import sys
import argparse

import json
from collections import defaultdict, OrderedDict
from itertools import product

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def convert_fauci_email(input_directory, verbose=0):
    """fauci-email: a json digest of Anthony Fauci's released emails

    @article{Benson-2021-fauci-emails,
        author = {Austin Benson and Nate Veldt and David F. Gleich},
        title = {fauci-email: a json digest of Anthony Fauci's released emails},
        url = {http://arxiv.org/abs/2108.01239},
        journal = {arXiv},
        year = {2021},
        pages = {2108.01239},
        volume = {cs.SI},
        note = {Code and data available from \\url{https://github.com/nveldt/fauci-email}}
    }

    @misc{Leopold-2021-fauci-emails,
        title = {Anthony Fauci’s Emails Reveal The Pressure That Fell On One Man},
        author = {Natalie Bettendorf and Jason Leopold},
        howpublished = {BuzzFeed News, \\url{https://www.buzzfeednews.com/article/nataliebettendorf/fauci-emails-covid-response}},
        month = {June},
        year = {2021},
        url = {https://s3.documentcloud.org/documents/20793561/leopold-nih-foia-anthony-fauci-emails.pdf},
    }

    Note: Am I REALLY citing buzzfeed
    """

    json_filename = 'fauci-email-data.json'
    MAX_THREAD_PARTICIPANTS = 25
    full_json_filename = os.path.join(input_directory, json_filename)
    if not os.path.exists(full_json_filename):
        return 0, 0, None, None
    with open(full_json_filename) as f_in:
        json_data = json.load(f_in)
    names = json_data["names"]
    people_degree = defaultdict(int)
    hypergraph_sets = []
    hypergraph = []
    for thread in json_data['emails']:
        thread_participants = set()
        for email in thread:
            thread_participants.add(names[email['sender']])
            for recipient in email["recipients"]:
                thread_participants.add(names[recipient])
        if len(thread_participants) > MAX_THREAD_PARTICIPANTS:
            continue
        for p in thread_participants:
            people_degree[p] += 1
        hypergraph_sets.append(thread_participants)
    hypergraph_sets = [e for e in hypergraph_sets if any(people_degree[node] > 1 for node in e)]
    people_degree = defaultdict(int)
    for e in hypergraph_sets:
        for p in e:
            people_degree[p] += 1
    node_dict = {i + 1: name for i, name in enumerate(sorted(people_degree.keys()))}
    name_dict = {name: i + 1 for i, name in enumerate(sorted(people_degree.keys()))}
    hypergraph = [sorted([name_dict[name] for name in s]) for s in hypergraph_sets]
    n = len(node_dict)
    m = len(hypergraph)
    degree = [people_degree[k] for k in sorted(people_degree.keys())]
    original_fauci_name_dict = {name: i for i, name in enumerate(json_data['names'])}
    labels = [json_data['clusters'][original_fauci_name_dict[name]] - 1 for name in sorted(people_degree.keys())]
    label_names = json_data['cluster_names']
    if verbose > 1:
        average_rank = sum([len(e) for e in hypergraph]) / len(hypergraph)
        average_degree = sum(people_degree.values()) / len(people_degree)
        print(f'Fauce Email: |V| = {n}, |E| = {m}, Average Rank = {average_rank:.3f}, Average Degree = {average_degree:.3f}')
    return n, m, degree, hypergraph, None, node_dict, labels, label_names


def generate_grid(input_directory, verbose=0):
    """Grid Hypergraph

    Use input_directory to pass in `n` and `m`, the two dimensions of the grid.
    Create a grid n x m and then treat each node as a hyperedge of its incident edges.
    """
    try:
        n, m = [int(i) for i in input_directory.split()]
    except ValueError:
        raise ValueError(f'Expected integer dimensions of grid, instead got {input_directory}.')
    hypergraph = []
    degree = defaultdict(int)
    name_dict = OrderedDict()
    for i, j in product(range(n), range(m)):
        nodes = []
        if i > 0:
            node = j + (i - 1) * (2 * m - 1) + (m - 1) + 1
            degree[node] += 1
            nodes.append(node)
        if j > 0:
            node = (j - 1) + i * (2 * m - 1) + 1
            degree[node] += 1
            nodes.append(node)
        if j < m - 1:
            name_dict[i * (2 * m - 1) + j + 1] = f'{i / n} {j / m + 1 / (2 * m)}'
            node = j + i * (2 * m - 1) + 1
            degree[node] += 1
            nodes.append(node)
        if i < n - 1:
            name_dict[i * (2 * m - 1) + j + m] = f'{i / n + 1 / (2 * n)} {j / m}'
            node = j + i * (2 * m + 1) + (m - 1) + 1
            degree[node] += 1
            nodes.append(node)
        hypergraph.append(nodes)
    print(max(degree.keys()))
    node_dict = OrderedDict([(name_dict[k], k) for k in sorted(name_dict.keys())])
    node_labels = []
    for k in sorted(name_dict.keys()):
        i, j = [float(f) for f in name_dict[k].split()]
        node_labels.append((i + j) / 2)
    degree = [degree[k] for k in sorted(degree.keys())]
    return (n - 1) * (2 * m - 1) + m - 1, n * m, degree, hypergraph, None, node_dict, node_labels, None


def generate_nodeGrid(input_directory, verbose=0):
    """Node Grid Hypergraph

    Use input_directory to pass in `n` and `m`, the two dimensions of the grid.
    Create a grid n x m and then treat each node neighborhood as a hyperedge.
    """
    def coord2ind(i, j):
        return i * m + j + 1

    try:
        n, m = [int(i) for i in input_directory.split()]
    except ValueError:
        raise ValueError(f'Expected integer dimensions of grid, instead got {input_directory}.')
    hypergraph = []
    degree = defaultdict(int)
    name_dict = OrderedDict()
    weights = {}
    center_id = {}
    hypergraph_node_weights = {}
    for i, j in product(range(n), range(m)):
        node = coord2ind(i, j)
        degree[node] += 1
        nodes = [node]
        name_dict[node] = f'{i / n} {j / m}'
        hyperedge_node_weights = [1]
        w = 1
        if i > 0:
            node = coord2ind(i - 1, j)
            nodes.append(node)
            w += 1
            hyperedge_node_weights.append(np.exp(-1 / n))
        if j > 0:
            node = coord2ind(i, j - 1)
            nodes.append(node)
            w += 1
            hyperedge_node_weights.append(np.exp(-1 / m))
        if i < n - 1:
            node = coord2ind(i + 1, j)
            nodes.append(node)
            w += 1
            hyperedge_node_weights.append(np.exp(-1 / n))
        if j < m - 1:
            node = coord2ind(i, j + 1)
            nodes.append(node)
            w += 1
            hyperedge_node_weights.append(np.exp(-1 / m))
        for i, node in enumerate(nodes):
            degree[node] += w * hyperedge_node_weights[i]
        nodes = tuple(nodes)
        hypergraph.append(nodes)
        weights[nodes] = w
        center_id[nodes] = coord2ind(i, j)
        hypergraph_node_weights[nodes] = hyperedge_node_weights
    node_dict = OrderedDict([(name_dict[k], k) for k in sorted(name_dict.keys())])
    node_labels = np.zeros([n, m])
    node_labels[n // 5:(4 * n) // 5, m // 5: (4 * m) // 5] = 1
    node_labels = node_labels.reshape(-1)
    #for k in sorted(name_dict.keys()):
    #    i, j = [float(f) for f in name_dict[k].split()]
    #    node_labels.append((i + j) / 2)
    degree = [degree[k] for k in sorted(degree.keys())]
    return n * m, n * m, degree, hypergraph, weights, node_dict, node_labels, None, center_id, hypergraph_node_weights


def generate_clique(input_directory, verbose=0):
    """Connected Cliques

    Use input_directory to pass in `n` and `k`, the number of nodes and clusters.
    Create `k` hyperedges of `n` // `k` nodes each with weight `n` // `k` and
    a hyperedge over all nodes with weight 1.
    """
    try:
        n, k = [int(i) for i in input_directory.split()]
    except ValueError:
        raise ValueError(f'Expected integer dimensions of grid, instead got {input_directory}.')
    hypergraph = []
    weights = {}
    degree = []
    name_dict = OrderedDict()
    node_labels = []
    p = 0
    cat = 1
    for i in np.linspace(n / k, n, k):
        end = int(i)
        nodes = tuple([j + 1 for j in range(p, end)])
        node_labels.extend([cat] * (end - p))
        hypergraph.append(nodes)
        w = end - p
        weights[nodes] = w
        m_x, m_y = np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)
        pos = np.random.normal((m_x, m_y), size=(end - p, 2))
        for j, v in enumerate(pos):
            name_dict[p + j] = f'{v[0]:.6f} {v[1]:.6f}'
            degree.append(w + 1)
        p = end
        cat += 1
    nodes = tuple(range(1, n+1))
    hypergraph.append(nodes)
    weights[nodes] = 1
    node_dict = OrderedDict([(name_dict[k], k) for k in sorted(name_dict.keys())])
    label_names = [str(i) for i in range(1, k + 1)]
    return n, k+1, degree, hypergraph, weights, node_dict, node_labels, label_names


CONVERSION_FUNCTION = {
    'fauci_email': convert_fauci_email,
    'grid': generate_grid,
    'nodeGrid': generate_nodeGrid,
    'clique': generate_clique,
}


def write_hypergraph(filename, n, m, degree, hypergraph, weights=None, center_id=None, hypergraph_node_weights=None):
    """Writes `hypergraph` to filename."""
    with open(filename, 'w') as f_out:
        print(f'{m} {n}', end=' ', file=f_out)
        print(int(center_id is not None), end='', file=f_out)
        print(int(hypergraph_node_weights is not None), end='', file=f_out)
        print(1, end='', file=f_out)
        print(int(weights is not None), end='\n', file=f_out)
        for e in hypergraph:
            if weights is not None:
                print(weights[e], end=' ', file=f_out)
            if center_id is not None:
                print(center_id[e], end=' ', file=f_out)
            if hypergraph_node_weights is not None:
                print(' '.join([f'{node} {w}' for node, w in zip(e, hypergraph_node_weights[e])]), file=f_out)
            else:
                print(*e, file=f_out)
        for d in degree:
            print(d, file=f_out)


def write_node_dict(filename, node_dict, verbose=0):
    """Writes in every line which actual node each hmetis node in [n] translates to."""
    if node_dict is None:
        if verbose > 0:
            print(f'There is no node_dict to be written in {filename}.')
        return

    with open(filename, 'w') as f_out:
        print('\n'.join([str(v) for v in node_dict]), file=f_out)


def write_labels(filename, label_names, labels, verbose=0):
    """Write the labels, one in each line"""
    with open(filename, 'w') as f_out:
        if label_names is not None:
            print(' '.join(label_names), file=f_out)
        for label in labels:
            print(label, file=f_out)


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Convert the varying hypergraph formats into the uniform hMETIS format.')
    parser.add_argument('-i', '--input_directory', help='Directory where the raw data is stored.', type=str, default=SRC_DIR)
    parser.add_argument('-o', '--output_directory', help='Directory to write the processed results.', type=str, default=SRC_DIR)
    parser.add_argument('-n', '--names', help='Choose which hypergraphs you want to process.', nargs='+', choices=CONVERSION_FUNCTION.keys(), default=CONVERSION_FUNCTION.keys())
    parser.add_argument('-v', '--verbose', help='Verbose mode. Prints out useful information. Higher levels print more information, including basic hypergraph stats.', action='count', default=0)
    parser.add_argument('-f', '--force', help='Force reprocessing even if .hmetis file is present.', action='store_true')
    args = parser.parse_args()
    return args.input_directory, args.output_directory, args.names, args.verbose, args.force


def main():
    input_directory, output_directory, names, verbose, force = parse_args()
    print(input_directory)
    for name in names:
        hmetis_filename = os.path.join(output_directory, f'{name}.hmetis')
        dict_filename = os.path.join(output_directory, f'{name}.dict')
        label_filename = os.path.join(output_directory, f'{name}.label')
        if not force and os.path.exists(hmetis_filename):
            if verbose > 0:
                print(f'File {hmetis_filename} exists.')
                print(f'Skipping processing for {name}.')
            continue
        if name not in CONVERSION_FUNCTION:
            if verbose > 0:
                print(f'There is no conversion function registered for hypergraph {name}.', file=sys.stderr)
            continue
        function = CONVERSION_FUNCTION[name]
        n, m, degree, hypergraph, weights, node_dict, labels, label_names, center_id, hypergraph_node_weights = function(input_directory, verbose=verbose)
        if hypergraph is None:
            print(f'Unable to load {name} from {input_directory}. Check files.')
            continue
        write_hypergraph(hmetis_filename, n, m, degree, hypergraph, weights, center_id, hypergraph_node_weights)
        write_node_dict(dict_filename, node_dict)
        write_labels(label_filename, label_names, labels)


def __init__():
    pass


if __name__ == '__main__':
    main()
