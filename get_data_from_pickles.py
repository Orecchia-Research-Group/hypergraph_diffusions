import os
import argparse
import pickle
from _collections import defaultdict

import numpy

DATASETS = {
    'Network Theory': 'network_theory',
    'Fauci Email': 'fauci_email',
    'Arxiv Cond Mat': 'arxiv_cond_mat',
    'DBPedia Writers': 'dbpedia_writer',
    'DBLP': 'hyperExtractedDblp',
    'Youtube': 'youtube',
    'Citeseer': 'citeseer',
    'DBPedia Record Labels': 'dbpedia_recordlabel',
    'DBPedia Genre': 'dbpedia_genre'
}


def data_from_pickle(directory='results', cut_function=None, regularizer=None, alpha=None):
    results = defaultdict(dict)
    possible_files = [filename for filename in os.listdir(directory) if filename.endswith('.pickle')]
    for name, code in DATASETS.items():
        for filename in possible_files:
            if not filename.startswith(f'Ikeda_{code}'):
                continue
            filename_tokens = os.path.splitext(filename)[0].split('_')
            a = int(filename_tokens[-1]) / 100
            reg = filename_tokens[-2]
            cut_func = filename_tokens[-3]
            if alpha is not None and a != alpha:
                continue
            if regularizer is not None and reg != regularizer:
                continue
            if cut_function is not None and cut_func != cut_function:
                continue

            with open(os.path.join(directory, filename), 'rb') as f:
                r = pickle.load(f)
                r['alpha'] = a
                r['regularizer'] = reg
                r['cut_function'] = cut_func
                r['dimensions'] = len(r['fx'][-1])
                r['iterations'] = len(r['fx'])
                results[name][cut_func] = r
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, default='results', help='Directory where .pickle files are')
    parser.add_argument('-c', '--cut-function', type=str, default=None, help='Restrict to only one cut function')
    parser.add_argument('-r', '--regularizer', default=None, type=str, help='Restrict to specific regularizer')
    parser.add_argument('-a', '--alpha', default=None, type=float, help='Restrict to specific alpha')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    results = data_from_pickle(directory=args.directory, cut_function=args.cut_function,
                               regularizer=args.regularizer, alpha=args.alpha)


if __name__ == '__main__':
    main()