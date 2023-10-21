import os
import sys
import argparse

import json
from collections import defaultdict, OrderedDict
from itertools import product

import numpy as np
import pandas as pd
from scipy.io import loadmat

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def convert_fauci_email(input_directory, verbose=0, **kwargs):
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


def generate_grid(input_directory, verbose=0, **kwargs):
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


def generate_nodeGrid(input_directory, verbose=0, **kwargs):
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


def generate_graph_grid(input_directory, verbose=0, **kwargs):
    """Grid Graph

    Use input_directory to pass in `n`, `m`, the two dimensions of the grid, and optionally `k`,
    whether the graph is 4 or 8 connected.
    """
    def coord2ind(i, j):
        return i * m + j + 1

    def add_edge(u, v):
        edge = (u, v)
        hypergraph.append(edge)
        degree[u] += 1
        degree[v] += 1
        total_edges[0] += 1

    input_directory_tokens = input_directory.split()
    try:
        n = int(input_directory_tokens[0])
        m = int(input_directory_tokens[1])
        k = int(input_directory_tokens[2]) if len(input_directory_tokens) > 2 else 4
    except ValueError:
        raise ValueError(f'Expected integer dimensions of grid, instead got {input_directory}.')

    hypergraph = []
    degree = defaultdict(int)
    name_dict = OrderedDict()
    # Make it a list so it is passed by reference to local function
    total_edges = [0]
    node_labels = np.zeros([n, m])
    for i, j in product(range(n), range(m)):
        node = coord2ind(i, j)
        node_labels[i, j] = np.exp(-16 * np.log(2)**2 * ((i - n/2)**2 + (j - m/2)**2)/n**2)
        name_dict[node] = f'{i / n} {j / m}'
        if i < n - 1:
            add_edge(node, coord2ind(i + 1, j))
        if j < m - 1:
            add_edge(node, coord2ind(i, j + 1))
        if k == 8:
            # Diagonal edges
            if i < n - 1:
                if j > 0:
                    add_edge(node, coord2ind(i + 1, j - 1))
                if j < m - 1:
                    add_edge(node, coord2ind(i + 1, j + 1))
    node_dict = OrderedDict([(name_dict[k], k) for k in sorted(name_dict.keys())])
    node_labels = node_labels.reshape(-1)
    D = [degree[k] for k in sorted(degree)]
    return n * m, total_edges[0], D, hypergraph, None, node_dict, node_labels, None, None, None


def generate_graph_ring(input_directory, verbose=0, **kwargs):
    """Grid Graph

    Use input_directory to pass in `n` the number of nodes in the ring.
    """
    try:
        n = int(input_directory)
    except ValueError:
        raise ValueError(f'Expected integer number of nodes in the ring, instead got {input_directory}.')

    hypergraph = []
    degree = defaultdict(int)
    name_dict = OrderedDict()
    node_labels = np.zeros(n)
    for i in range(1, n + 1):
        theta = 2 * np.pi * i / n
        node_labels[i - 1] = np.exp(-(np.sin(theta) - 1)**2)
        name_dict[i] = f'{np.cos(theta)} {np.sin(theta)}'
        j = i % n + 1
        hypergraph.append((i, j))
        degree[i] += 1
        degree[j] += 1
    node_dict = OrderedDict([(name_dict[k], k) for k in sorted(name_dict.keys())])
    D = [degree[k] for k in sorted(degree)]
    return n, n, D, hypergraph, None, node_dict, node_labels, None, None, None


def generate_graph_line(input_directory, verbose=0, **kwargs):
    """Line Graph

    Use input_directory to pass in `n` the number of nodes on the line
    """
    try:
        n = int(input_directory)
    except ValueError:
        raise ValueError(f'Expected integer number of nodes in the ring, instead got {input_directory}.')

    hypergraph = []
    degree = defaultdict(int)
    name_dict = OrderedDict()
    node_labels = np.zeros(n)
    node_labels[n // 2 - 4:n // 2 + 4] = 1
    for i in range(1, n + 1):
        # node_labels[i - 1] = np.exp(-16 * np.log(2)**2 * ((i - n/2)**2)/n**2)# np.cos(np.pi * i / n) + 1
        name_dict[i] = f'{i / n}'
        if i < n:
            hypergraph.append((i, i + 1))
            degree[i] += 1
            degree[i + 1] += 1
    node_dict = OrderedDict([(name_dict[k], k) for k in sorted(name_dict.keys())])
    D = [degree[k] for k in sorted(degree)]
    return n, n - 1, D, hypergraph, None, node_dict, node_labels, None, None, None


def generate_bolas(input_directory, verbose=0, **kwargs):
    """Dumbbell Graph

    Use input_directory to pass in `n` the number of nodes in the graph
    """
    input_directory_tokens = input_directory.split()
    try:
        n = int(input_directory_tokens[0])
        k = int(input_directory_tokens[1])
    except ValueError:
        raise ValueError(f'Expected sizes of each clique and length of path, instead got {input_directory}.')

    degree = defaultdict(int)
    node_labels = np.zeros(2 * n + k, int)
    half = n // 2
    hypergraph = [(n, n + 1), (n + k, n + k + 1)]
    degree[n] += 1
    degree[n + 1] += 1
    degree[n + k] += 1
    degree[n + k + 1] += 1
    for i in range(n):
        node_labels[i] = 0
        node_labels[i + n + k] = 2
        for j in range(i+1, n):
            hypergraph.append((i + 1, j + 1))
            degree[i + 1] += 1
            degree[j + 1] += 1

            hypergraph.append((i + n + k + 1, j + n + k + 1))
            degree[i + n + k + 1] += 1
            degree[j + n + k + 1] += 1
    for j in range(n, n + k - 1):
        node_labels[j] = 1
        node_labels[j + 1] = 1
        hypergraph.append((j + 1, j + 2))
        degree[j + 1] += 1
        degree[j + 2] += 1
    hypergraph = sorted(hypergraph)
    D = [degree[k] for k in sorted(degree)]
    return 2 * n + k, len(hypergraph), D, hypergraph, None, None, node_labels, None, None, None


def generate_li_synthetic(input_directory, verbose=0, **kwargs):
    """Li Synthetic dataset

    Synthetic dataset appearing in Li et al (https://www.jmlr.org/papers/volume21/18-790/18-790.pdf)
    Use input directory to pass in `n`, `m1`, `m2` and `r`, the number of nodes,
    number of hyperedges inside each cluster, number of hyperedges involving the whole graph
    and rank of each hyperedge.
    """
    input_directory_tokens = input_directory.split()
    try:
        n = int(input_directory_tokens[0])
        m1 = int(input_directory_tokens[1])
        m2 = int(input_directory_tokens[2])
        k = int(input_directory_tokens[3])
    except ValueError:
        raise ValueError(f'Expected integer dimensions of grid, instead got {input_directory}.')
    half = n // 2
    nodes = list(range(n))
    degree = np.zeros(n, int)
    hypergraph = []
    node_labels = np.ones(n, int)
    node_labels[:half] = 0
    for i in range(m1):
        k1_hyperedge = np.random.choice(nodes[:half], size=k, replace=False)
        hypergraph.append(tuple(k1_hyperedge + 1))
        degree[k1_hyperedge] += 1

    for j in range(m1):
        k2_hyperedge = np.random.choice(nodes[half:], size=k, replace=False)
        hypergraph.append(tuple(k2_hyperedge + 1))
        degree[k2_hyperedge] += 1

    for i in range(m2):
        cross_hyperedge = np.random.choice(nodes, size=k, replace=False)
        hypergraph.append(tuple(cross_hyperedge + 1))
        degree[cross_hyperedge] += 1

    hypergraph = sorted(hypergraph)
    return n, len(hypergraph), degree, hypergraph, None, None, node_labels, None, None, None


def generate_clique(input_directory, verbose=0, **kwargs):
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


def generate_UCI(id, verbose=0):
    """UCI datasets

    Generate a hypergraph from the appropriate ucimlrepo id
    """
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=id)
    X = dataset.data.features
    labels = dataset.data.targets.T.values[0]
    label_names = {k: i for i, k in enumerate(sorted(set(labels)))}
    labels = np.array([label_names[k] for k in labels])
    variables = dataset['variables']

    n = labels.shape[0]
    hypergraph = []
    degree = np.zeros(n, int)

    for k in variables.index:
        if variables.loc[k, 'role'] != 'Feature':
            continue

        # Ignore columns with missing values
        if variables.loc[k, 'missing_values'] == 'yes':
            continue
        variable_name = variables.loc[k, 'name']
        if verbose > 0:
            print(f'Variable {variable_name} values:', end='')
        if variables.loc[k, 'type'] in ['Categorical', 'Binary']:
            for v in sorted(X[variable_name].unique()):
                if verbose > 0:
                    print(f' {v}', end='')
                ind = np.where(X[variable_name] == v)[0]
                # print(ind)
                hypergraph.append(ind + 1)
                degree[ind] += 1
        print()
    return n, len(hypergraph), degree, hypergraph, None, None, labels, label_names, None, None


def generate_zoo(input_directory, verbose=0, **kwargs):
    """Zoo

    Dataset of 101 animals classified into 7 categories using 16 features.
    All features have distinct values.
    The hyperedges are over the same values in each feature.

    Dataset can be found on https://archive.ics.uci.edu/dataset/111/zoo
    """
    return generate_UCI(id=111, verbose=verbose)


def generate_mushroom(input_directory, verbose=0, **kwargs):
    """Mushroom

    Dataset of 8124 mushrooms classified into poisonous and edible.
    All features have distinct categorical values.
    There are 22 features, but feature 11 has missing values and is ignored.
    The hyperedges are over the same values in each feature.

    Dataset can be found on https://archive.ics.uci.edu/dataset/73/mushroom
    """
    return generate_UCI(id=73, verbose=verbose)


def generate_newsgroups(input_directory, verbose=0, **kwargs):
    """Newsgroups

    An abreviation of the 20 newsgroups dataset (https://archive.ics.uci.edu/dataset/113/twenty+newsgroups)
    compiled by the late Sam Roweis, but still hosted on his NYT page (https://cs.nyu.edu/~roweis/data.html)
    For 16252 messages in 'comp*', 'rec*', 'sci*' and 'talk*' newsgroups, there are 100 binary features whether
    certain words appear.

    The target is to identify which newsgroup each message belongs to.
    """
    mat = loadmat(os.path.join(input_directory, '20news_w100.mat'))
    messages = mat['documents'].T
    wordlist = mat['wordlist']
    labels = mat['newsgroups'][0]
    label_names = [v[0] for v in mat['groupnames'][0]]
    n = messages.shape[0]
    hypergraph = []
    degree = np.zeros(n, int)
    for k in range(messages.shape[1]):
        ind = (messages.getcol(k) == 1).tocoo().row
        hypergraph.append(ind + 1)
        degree[ind] += 1
    return n, len(hypergraph), degree, hypergraph, None, None, labels, label_names, None, None


def generate_covertype(input_directory, types=(4, 5), verbose=0, **kwargs):
    """Covertype

    Predicting forest cover type from cartographic variables only
    (no remotely sensed data). The actual forest cover type for
    a given observation (30 x 30 meter cell) was determined from
    US Forest Service (USFS) Region 2 Resource Information System
    (RIS) data.  Independent variables were derived from data
    originally obtained from US Geological Survey (USGS) and
    USFS data.  Data is in raw form (not scaled) and contains
    binary (0 or 1) columns of data for qualitative independent
    variables (wilderness areas and soil types).

    The dataset contains 12 numerical features, wilderness area
    as 4 binary columns and soil type as 40 binary columns.
    """
    buckets = 10

    df = pd.read_csv(os.path.join(input_directory, 'covertype', 'covtype.data'), header=None)
    df = df.loc[df.loc[:, 54].isin(types)]
    n, f = df.shape
    hypergraph = []
    degree = np.zeros(n, int)
    labels = df.loc[:, 54].values
    label_names = types
    for c in range(10):
        column_data = df.loc[:, c]
        limits = np.percentile(column_data.values, np.linspace(0, 100, buckets + 1))
        for b in range(buckets):
            ind = np.where((limits[b] <= column_data) & (column_data <= limits[b + 1]))[0]
            hypergraph.append(ind + 1)
            degree[ind] += 1
    for c in range(10, 54):
        ind = np.where(df.loc[:, c])[0]
        hypergraph.append(ind + 1)
        degree[ind] += 1
    return n, len(hypergraph), degree, hypergraph, None, None, labels, label_names, None, None


CONVERSION_FUNCTION = {
    'fauci_email': convert_fauci_email,
    'grid': generate_grid,
    'nodeGrid': generate_nodeGrid,
    'clique': generate_clique,
    'graph_grid': generate_graph_grid,
    'graph_ring': generate_graph_ring,
    'graph_line': generate_graph_line,
    'dumbbell': generate_bolas,
    'li_synthetic': generate_li_synthetic,
    'zoo': generate_zoo,
    'mushroom': generate_mushroom,
    'newsgroups': generate_newsgroups,
    'covertype': generate_covertype,
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
    if labels is None:
        if verbose > 0:
            print(f'There are no labels to be written in {filename}.')
        return
    with open(filename, 'w') as f_out:
        if label_names is None:
            label_names = sorted(set(labels))
        print(' '.join([str(i) for i in label_names]), file=f_out)
        for label in labels:
            print(label, file=f_out)


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Convert the varying hypergraph formats into the uniform hMETIS format.',
                                     epilog='Konstantinos Ameranis, University of Chicago 2023')
    parser.add_argument('-i', '--input_directory', help='Directory where the raw data is stored.', type=str, default=SRC_DIR)
    parser.add_argument('-o', '--output_directory', help='Directory to write the processed results.', type=str, default=SRC_DIR)
    parser.add_argument('-n', '--names', help='Choose which hypergraphs you want to process.', nargs='+', choices=CONVERSION_FUNCTION.keys(), default=CONVERSION_FUNCTION.keys())
    parser.add_argument('-v', '--verbose', help='Verbose mode. Prints out useful information. Higher levels print more information, including basic hypergraph stats.', action='count', default=0)
    parser.add_argument('-f', '--force', help='Force reprocessing even if .hmetis file is present.', action='store_true')
    parser.add_argument('--suffix', help='Add at the end of the filenames to not overwrite for different settings', default='', type=str)
    parser.add_argument('-t', '--types', help='Covertypes that should be kept for 1-7. Popular datasets are from (4, 5) and (6, 7)',
                        default=(4, 5), type=int, nargs='+')
    args = parser.parse_args()
    return args.input_directory, args.output_directory, args.names, args.verbose, args.force, args.suffix, args.types


def main():
    input_directory, output_directory, names, verbose, force, suffix, types = parse_args()
    print(input_directory)
    for name in names:
        hmetis_filename = os.path.join(output_directory, f'{name}{suffix}.hmetis')
        dict_filename = os.path.join(output_directory, f'{name}{suffix}.dict')
        label_filename = os.path.join(output_directory, f'{name}{suffix}.label')
        if not force and os.path.exists(hmetis_filename):
            if verbose > 0:
                print(f'File {hmetis_filename} exists.')
                print(f'Skipping processing for {name}{suffix}.')
            continue
        if name not in CONVERSION_FUNCTION:
            if verbose > 0:
                print(f'There is no conversion function registered for hypergraph {name}.', file=sys.stderr)
            continue
        function = CONVERSION_FUNCTION[name]
        n, m, degree, hypergraph, weights, node_dict, labels, label_names, center_id, hypergraph_node_weights = function(input_directory, verbose=verbose, types=types)
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
