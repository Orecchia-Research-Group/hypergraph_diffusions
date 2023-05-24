from collections import defaultdict
import dill

import numpy as np

import reading
from get_data_from_pickles import data_from_pickle, DATASETS
from diffusion_functions import diffusion_functions, diffusion, sweep_cut, compute_hypergraph_matrices, all_sweep_cuts
import matplotlib.pyplot as plt


results = data_from_pickle(directory='paper_results/', regularizer='degree')
cut_results = defaultdict(lambda: defaultdict(dict))

# Median Improvement - Table 2
print('Median Improvement - Table 2')
for name, r in results.items():
    print(f"{name:21s}", end='')
    our25, our50, our75 = np.quantile(r['infinity']['fx_cs'], [0.25, 0.5, 0.75])
    their25, their50, their75 = np.quantile(r['infinity']['fx'], [0.25, 0.5, 0.75])
    print(f" & {(our50 / their50 - 1) * 100:+7.2f}\\%", end='')
    print(" \\\\")

# Timings - Table 3
print('Timings - Table 3')
for name, r in results.items():
    print(f"{name:21s}", end='')
    for func_name, a in sorted(r.items()):
        print(f" & {len(a['t'])-1:3d} & {a['t'][-1]:8.2f} & {a['t'][-1] / len(a['t']):7.3f}", end='')
    print(" \\\\")

# Conductances
print('Compute conductances')
for name, r in results.items():
    print(name)
    n, m, node_weights, hypergraph, weights, center_id, hypergraph_node_weights = reading.read_hypergraph(f'Hypergraphs/{DATASETS[name]}.hmetis')
    for cut_func, saved in r.items():
        cut_results[name][cut_func]['value last'], cut_results[name][cut_func]['volume last'], cut_results[name][cut_func]['conductance last'] = all_sweep_cuts(saved['x'], n, m, node_weights, hypergraph, 1)
        cut_results[name][cut_func]['value avg'], cut_results[name][cut_func]['volume avg'], cut_results[name][cut_func]['conductance avg'] = all_sweep_cuts(saved['x_cs'], n, m, node_weights, hypergraph, 1)

print('That took a while. Better save that for next time')
with open('paper_results/conductances.dill', 'wb') as f:
    dill.dump(cut_results, f)

print('Sweep Cuts - Table 6')
for name, r in cut_results.items():
    print(f"{name:21s}", end='')
    for cut_func, a in sorted(r.items()):
        ours = a['conductance avg'].min()
        theirs = a['conductance last'].min()
        print(f" & {ours:.4f} & {theirs:.4f} & {(ours / theirs- 1) * 100:5.2f}\\%", end='')
    print(' \\\\')

print('Printing Figure 4 in paper_results.')
for name, r in results.items():
    print(name)
    for cut_func, saved in r.items():
       plt.figure()
       plt.plot(saved['fx'].min(axis=1), label='Takai et al.')
       plt.plot(saved['fx_cs'].min(axis=1), label='This paper')
       plt.xlabel('Iterations')
       plt.ylabel(r'$\mathcal{U}(\mathbf{x}) + \frac{\lambda}{2} \|\mathbf{x}\|_D^2 - \langle \mathbf{s}, \mathbf{x} \rangle$')
       plt.legend(loc='upper right', prop={'size': 24})
       plt.savefig(f'paper_results/function_{name}_{cut_func}.png', dpi=300)
       plt.close()

