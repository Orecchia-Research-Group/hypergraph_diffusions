"""Input/Output Module

Helper functions for reading and writing files.
"""

from collections import Counter
import numpy as np


def read_seed(filename, labels, dimensions, D):
    """Read seed from a file where each line has the seed for the corresponding node"""
    p = None
    D = np.array(D)
    if filename is None:
        return None
    try:
        p = float(filename)
    except ValueError:
        pass
    if p is not None:
        n = len(labels)
        number_labels = len(set(labels))
        if number_labels != dimensions:
            raise ValueError(f'When using percentage seed, dimension space should equal number of labels. Labels: {number_labels}. Dimensions: {dimensions}.')
        ch = np.random.rand(n) < p
        idx = np.where(ch)[0]
        train_examples = ch.sum()
        s = np.zeros([n, number_labels])
        s[ch] = -1
        label_counter = Counter([labels[i] for i in idx])
        for l in set(labels):
            s[ch, l] /= (train_examples - label_counter[l])
        for i in idx:
            s[i, labels[i]] = 1 / label_counter[labels[i]]
        s -= s.sum(axis=0) / len(s)
        for i in range(dimensions):
            si = s[:, i]
            print(f'{i} {label_counter[i]:4d} {si[si < 0].sum():7.4f} {si[si > 0].sum():7.3f} {si.sum():7.4f}')

        return s
    with open(filename) as f:
        s = np.array([[float(i) for i in line.split()] for line in f])
    return s


def read_hypergraph(filename):
    """Read a hypergraph and return n, m and a list of participating nodes"""
    with open(filename) as f:
        weights = None
        first_line = [int(i) for i in f.readline().split()]
        m, n = first_line[:2]
        fmt = first_line[2] if len(first_line) > 2 else 0
        if fmt % 10 == 1:
            weights = {}
        hypergraph = []
        for _ in range(m):
            start = 0
            line = f.readline().split()
            if fmt % 10 == 1:
                w = float(line[0])
                start = 1
            hypergraph.append(tuple([int(i) - 1 for i in line[start:]]))
            if fmt % 10 == 1:
                weights[hypergraph[-1]] = w
        if fmt // 10 == 1:
            degree = [float(f.readline()) for _ in range(n)]
    return n, m, degree, hypergraph, weights


def read_labels(filename):
    """Groundtruth community labels"""
    if filename is None:
        return [], []
    with open(filename) as f:
        label_names = np.array(f.readline().split())
        labels = [int(i) for i in f]
    return label_names, labels


def read_positions(filename):
    """Manual positions"""
    if filename is None:
        return None
    with open(filename) as f:
        positions = np.array([[float(i) for i in line.split()] for line in f])
        return positions
