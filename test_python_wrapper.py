import reading
import diffusion


n, m, node_weights, hypergraph, weights, center_id, hypergraph_node_weights = reading.read_hypergraph('data/Paper_datasets/zoo.hmetis')
# Turn hypergraph into list of lists
hypergraph = [list(e) for e in hypergraph]
label_names, labels = reading.read_labels('data/Paper_datasets/zoo.label')
gs = diffusion.GraphSolver(n, m, node_weights, hypergraph, len(label_names), labels, 0)
