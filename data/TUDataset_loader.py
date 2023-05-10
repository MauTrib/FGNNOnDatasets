import dgl
import torch
import pandas as pd
import pickle
import os


def load_adj(adj_path):
    """Loads the adjacency matrix from the file and returns it as a dgl graph."""
    df = pd.read_csv(adj_path, sep=",", header=None)
    u = df[0].values - 1
    v = df[1].values - 1
    graph = dgl.graph((u, v))
    graph = dgl.to_bidirected(graph)  # Ensure graph is undirected
    return graph


def load_node_labels(node_label_path):
    """Loads the node labels and returns them as a tensor of size [n_nodes, n_features]."""
    df = pd.read_csv(node_label_path, sep=",", header=None)
    node_labels = df.values
    node_labels = torch.tensor(node_labels, dtype=torch.long)
    return node_labels


def load_node_features(node_feat_path):
    """Loads the node features and returns them as a tensor of size [n_nodes, n_features]."""
    df = pd.read_csv(node_feat_path, sep=",", header=None)
    node_features = df.values
    node_features = torch.tensor(node_features, dtype=torch.float)
    return node_features


def load_graph_indicator(graph_indicator_path):
    """Loads the graph indicator from the file and returns it as a tensor."""
    with open(graph_indicator_path, "r") as f:
        gi = f.read().splitlines()
    gi = [int(i) for i in gi]
    return gi


def unbatch(graph, gi):
    """Unbatch the sparse global graph into the smaller unconnected graphs."""
    gidx = [i for i in range(len(gi)) if gi[i] != gi[i - 1]]
    gi_nnodes = [gidx[i + 1] - gidx[i] + 1 for i in range(len(gidx) - 1)] + [
        len(gi) - gidx[-1] + 1
    ]
    # Determine which are the edges of each graph
    edges = []
    for i in range(len(gidx) - 1):
        edges.append(
            graph.subgraph([j for j in range(gidx[i], gidx[i + 1])]).num_edges()
        )
    edges.append(
        graph.subgraph([j for j in range(gidx[-1], graph.num_nodes())]).num_edges()
    )
    # Unbatch the result once we have node/edge correspondance
    graphs = dgl.unbatch(graph, torch.tensor(gi_nnodes) - 1, torch.tensor(edges))

    return graphs


def save_graphs(graphs, path):
    with open(path, "wb") as f:
        pickle.dump(graphs, f)


def read_graphs(path):
    graphs = pickle.load(open(path, "rb"))
    return graphs


def load_data(output_path, process_func=None):
    if not os.path.exists(output_path):
        if process_func is None:
            raise ValueError(
                "process_func must be provided if output_path does not exist."
            )
        process_func()
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Couldn't generate file {output_path}...")
    graphs = read_graphs(output_path)
    return graphs
