from collections import defaultdict, Counter
import scipy as sp
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx

from diffusion_functions import diffusion, diffusion_functions, compute_hypergraph_matrices


HEIGHT = 0.07


def graph_electrical_flow(n, m, hypergraph, u, v):
    """Given a hypergraph, construct a graph and hope
     that solving on that will give something similar"""
    edges = []
    for i, h in enumerate(hypergraph):
        edges.extend([(i + n, v) for v in h])
    G = nx.Graph(edges)
    L = nx.laplacian_matrix(G)
    print(L.shape)
    i_ext = np.zeros(n + m)
    i_ext[u] = 1
    i_ext[v] = -1
    x = sp.sparse.linalg.spsolve(L, i_ext)
    x = x - x.min()
    return x


def electrical_flow(n, m, hypergraph, weights, u, v):
    x = cp.Variable(n)
    y_min = cp.Variable(m)
    y_max = cp.Variable(m)
    constraints = [x[u] == 1, x[v] == 0]
    constraint_names = [(-1, u), (v, -1)]
    for j, h in enumerate(hypergraph):
        constraints.extend([y_min[j] <= x[i] for i in h])
        constraint_names.extend([(i, j + n) for i in h])
        constraints.extend([y_max[j] >= x[i] for i in h])
        constraint_names.extend([(j + n, i) for i in h])
    obj = cp.Minimize(sum([weights[h] * (y_max[j] - y_min[j])**2 for j, h in enumerate(hypergraph)]))
    prob = cp.Problem(obj, constraints)
    result = prob.solve()
    return result, prob, x, y_min, y_max, constraints, constraint_names


def flow_laplacian_solver(n, m, D, hypergraph, weights, s, T=100):
    x0 = np.zeros((n, 1), float)
    t, x, y, fx = diffusion(x0, n, m, D, hypergraph, weights, s, T=T, verbose=1)
    W, sparse_h, rank = compute_hypergraph_matrices(n, m, hypergraph, weights)
    x_final = np.mean(x, axis=0)
    x_final = (x_final - x_final.min()) / (x_final.max() - x_final.min())
    _, y_final, fx_final = diffusion_functions['infinity'](x_final, sparse_h, rank, W, D)
    y_min = np.zeros(m)
    y_max = np.zeros(m)
    for j, h in enumerate(hypergraph):
        y_min[j] = x_final[h, :].min()
        y_max[j] = x_final[h, :].max()
    return x_final, y_final, y_min, y_max, fx_final


def visualize_solution(n, m, hypergraph, x, y_min, y_max, names, sol_values, filename=None):
    if filename is None:
        filename = 'simple_hypergraph'
    x = x.round(2)
    y_min = y_min.round(2)
    y_max = y_max.round(2)
    y = y_min + (y_max - y_min) / 2
    nodeCounter = Counter(x)
    edgeCounter = Counter(y)
    node_pos_counter = defaultdict(int)
    edge_pos_counter = defaultdict(int)
    edge_names = {j: ''.join([names[i] for i in h]) for j, h in enumerate(hypergraph)}

    edges = []
    for j, h in enumerate(hypergraph):
        edges.extend([(names[i], edge_names[j]) for i in h])
    G = nx.from_edgelist(edges)
    G_ef = nx.from_edgelist(sol_values, create_using=nx.DiGraph)
    G_ef.add_nodes_from(names)
    G_ef.add_nodes_from(edge_names.values())
    print(G_ef.edges(data=True))
    pos = {}
    for i in range(n):
        pos[names[i]] = (x[i], HEIGHT * (node_pos_counter[x[i]] - (nodeCounter[x[i]] / 2 + 0.5)))
        node_pos_counter[x[i]] += 1
    for j in range(m):
        pos[edge_names[j]] = (y[j], 0.2 + HEIGHT * (edge_pos_counter[y[j]] - (edgeCounter[y[j]] / 2 + 0.5)))
        edge_pos_counter[y[j]] += 1
    nx.draw(G, pos=pos, font_size=8, with_labels=True, font_color='w')
    plt.axis('equal')
    plt.savefig(f'{filename}_full.png', bbox_inches='tight', dpi=500)
    plt.close()

    # pos = nx.spectral_layout(G_ef)
    degrees = {v: d for v, d in G_ef.out_degree(names, weight='weight')}
    # print(degrees)
    # print(max(degrees.values()))
    total = max(degrees.values())
    weights = nx.get_edge_attributes(G_ef, 'weight')
    edge_weights = [[1-w / total] * 3 for w in weights.values()]
    # print(edge_weights)
    nx.draw(G_ef, pos=pos, font_size=8, with_labels=True, font_color='w', arrows=True,
            edge_color=edge_weights,)
    plt.axis('equal')
    plt.savefig(f'{filename}_sol.png', bbox_inches='tight', dpi=500)
    plt.close()


def main():
    T = 10000
    hypergraph = [
        (0, 1, 2),
        (1, 3, 4),
        (2, 4, 5),
        (2, 5, 6),
        (5, 7, 8),
        (5, 6, 8),
        (2, 3, 8),
    ]
    m = len(hypergraph)
    n = max([max(e) for e in hypergraph]) + 1
    weights = defaultdict(lambda: 1)
    degree = np.zeros(n)
    for e in hypergraph:
        for v in e:
            degree[v] += weights[e]
    names = [chr(ord("A") + i) for i in range(n)]
    edge_names = {j: ''.join([names[i] for i in h]) for j, h in enumerate(hypergraph)}
    result, prob, x, y_min, y_max, constraints, constraint_names = electrical_flow(n, m, hypergraph, defaultdict(lambda: 1), 8, 0)
    print(result)
    for i in range(n):
        print(f'{names[i]}: {x[i].value:.3f}')
    for j, h in enumerate(hypergraph):
        print(f'({"".join([names[i] for i in h])}): {y_max[j].value - y_min[j].value:.3f}')
    dual_names = []
    for a, b in constraint_names:
        def translate_number(i):
            if i < 0: return str(i)
            elif i < n: return names[i]
            else: return edge_names[i - n]

        dual_names.append((translate_number(a), translate_number(b)))
    sol_values = []
    for (u, v), f in zip(dual_names, [c.dual_value for c in constraints]):
        if abs(f) < 1e-3:
            continue
        print(f'{u:3s} -> {v:3s}: {f:9.3f}')
        sol_values.append((u, v, {'weight': round(abs(float(f)), 3)}))
    visualize_solution(n, m, hypergraph, x.value, y_min.value, y_max.value, names, sol_values[2:])

    s = np.zeros((n, 1))
    u = 0
    v = 8
    s[u] = -1
    s[v] = 1
    flow_x, flow_y, flow_y_min, flow_y_max, flow_fx = flow_laplacian_solver(n, m, degree, hypergraph, None, s, T=T)
    print(flow_x)
    print(x.value)
    print(f'y_min = {y_min.value}')
    print(f'flow_y_min = {flow_y_min}')
    print(f'y_max = {y_max.value}')
    print(f'flow_y_max = {flow_y_max}')
    print(sum(abs(y_min.value - flow_y_min)))
    print(sum(abs(y_max.value - flow_y_max)))
    visualize_solution(n, m, hypergraph, flow_x.flatten(), flow_y_min.flatten(), flow_y_max.flatten(), names, sol_values[2:], filename=f'laplacian_solver_{T:04d}')

    C_eff_pair = {}
    C_eff = {}
    C_eff_arg = {}
    for e in hypergraph:
        for i in range(len(e)):
            for j in range(i+1, len(e)):
                pair = tuple(sorted([e[i], e[j]]))
                if pair not in C_eff_pair:
                    result, prob, x, y_min, y_max, constraints, constraint_names = electrical_flow(n, m, hypergraph, defaultdict(lambda: 1), pair[1], pair[0])
                    C_eff_pair[pair] = result
                if (e not in C_eff) or (C_eff[e] > C_eff_pair[pair]):
                    C_eff[e] = C_eff_pair[pair]
                    C_eff_arg[e] = pair
    for i, e in enumerate(hypergraph):
        print(f"Hyperedge {edge_names[i]} has effective conductance {1 / C_eff[e]:.3f} for pair ({names[C_eff_arg[e][0]]}, {names[C_eff_arg[e][1]]})")


if __name__ == '__main__':
    main()
