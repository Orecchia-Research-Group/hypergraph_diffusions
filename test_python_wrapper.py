import reading
import diffusion


n, m, node_weights, hypergraph, weights, center_id, hypergraph_node_weights = reading.read_hypergraph('data/Paper_datasets/zoo.hmetis')
label_names, labels = reading.read_labels('data/Paper_datasets/zoo.label')
gs = diffusion.GraphSolver(n, m, node_weights, hypergraph, len(label_names), labels, 0)
