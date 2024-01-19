import os
import sys
import argparse
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_FOLDER = 'data/Paper_results'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', type=str, default='zoo', help='Base name of the graph(s) to process.', nargs='+')
    parser.add_argument('--result_folder', type=str, default=RESULT_FOLDER, help='')
    parser.add_argument('-p', '--prefix', type=str, default='', help='Prefix before graph name in the results folder.')
    parser.add_argument('-v', '--verbose', help='Verbose mode. Prints out useful information. Higher levels print more information.', action='count', default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    df = {}

    fig_seeds, ax_seeds = plt.subplots()
    for graph_name in args.graph:
        graph_result_filename = os.path.join(args.result_folder, f'{args.prefix}{graph_name}.csv')
        df[graph_name] = pd.read_csv(graph_result_filename)
        df[graph_name]['Graph Name'] =[os.path.basename(path) for path in df[graph_name]['Graph Name']]

        # Grouping
        groups = df[graph_name].groupby(['Graph Name', 'buckets', 'seeds', 'iteration'])[['time', 'error']]
        group_mean = groups.mean().reset_index()
        group_median = groups.median().reset_index()
        group_min = groups.min().reset_index()
        group_max = groups.max().reset_index()
        group_std = groups.std().reset_index()

        # Iteration results
        for b, s in product(group_mean['buckets'].unique(), group_mean['seeds'].unique()):
            idx = (group_mean['buckets'] == b) & (group_mean['seeds'] == s)
            selected_data_mean = group_mean.loc[idx, ['iteration', 'time', 'error']]
            selected_data_std = group_std.loc[idx, ['iteration', 'time', 'error']]

            for column in ['time', 'error']:
                fig = plt.figure()
                plt.plot(selected_data_mean['iteration'] + 1, selected_data_mean[column])
                plt.fill_between(selected_data_mean['iteration'] + 1,
                                 selected_data_mean[column] - selected_data_std[column],
                                 selected_data_mean[column] + selected_data_std[column], alpha=0.5)
                figure_name = os.path.join(args.result_folder, f'{args.prefix}{graph_name}_iteration_{column}_{b}_{s}.png')
                plt.xlabel('# iteration')
                plt.ylabel('column')
                plt.savefig(figure_name, dpi=300, bbox_inches='tight')
                plt.close(fig)

        # Seed results
        idx = group_mean['iteration'] == 199
        selected_data_mean = group_mean.loc[idx, ['seeds', 'time', 'error']]
        selected_data_std = group_std.loc[idx, ['seeds', 'time', 'error']]
        ax_seeds.plot(selected_data_mean['seeds'], selected_data_mean['error'], label=graph_name)
        ax_seeds.fill_between(selected_data_mean['seeds'], selected_data_mean['error'] - selected_data_std['error'],
                              selected_data_mean['error'] + selected_data_std['error'], alpha=0.5)
        ax_seeds.set_xlim(selected_data_mean['seeds'].min(), selected_data_mean['seeds'].max())

    ax_seeds.set_ylim(ymin=0)
    ax_seeds.set_xlabel('# of revealed labels')
    ax_seeds.set_ylabel('Error (%)')
    ax_seeds.legend()
    figure_name = os.path.join(args.result_folder, f'{args.prefix}seeds.png')
    for side in ['top', 'right']:
        ax_seeds.spines[side].set_visible(False)
    plt.savefig(figure_name, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
