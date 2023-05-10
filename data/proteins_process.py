import os
import torch
import dgl

from data.TUDataset_loader import (
    load_adj,
    load_node_labels,
    load_node_features,
    load_graph_indicator,
    unbatch,
    save_graphs,
)
from data.TUDataset_loader import load_data as tu_load_data
from toolbox.conversions import (
    dgl_adj_to_fgnn_tensor,
    dgl_with_node_features_to_fgnn_tensor,
)

SUBFOLDER = "data/PROTEINS/"
RAW_DATA_FOLDER = "raw"
ADJ_FILENAME = "PROTEINS_A.txt"
GRAPH_INDICATOR_FILENAME = "PROTEINS_graph_indicator.txt"
GRAPH_LABEL_FILENAME = "PROTEINS_graph_labels.txt"
NODE_LABEL_FILENAME = "PROTEINS_node_labels.txt"
NODE_FEAT_FILENAME = "PROTEINS_node_attributes.txt"
OUTPUT_FILENAME = "processed/PROTEINS"
OUTPUT_DGL_EXTENSION = ".dgl"
OUTPUT_FGNN_EXTENSION = ".fgnn"


def load_PROTEINS_old():
    adj_path = os.path.join(SUBFOLDER, RAW_DATA_FOLDER, ADJ_FILENAME)
    adj_graph = load_adj(adj_path)
    node_label_path = os.path.join(SUBFOLDER, RAW_DATA_FOLDER, NODE_LABEL_FILENAME)
    node_labels = load_node_labels(node_label_path)
    node_feat_path = os.path.join(SUBFOLDER, RAW_DATA_FOLDER, NODE_FEAT_FILENAME)
    node_features = load_node_features(node_feat_path)

    adj_graph.ndata["label"] = node_labels
    adj_graph.ndata["feat"] = node_features

    graph_indicator_path = os.path.join(
        SUBFOLDER, RAW_DATA_FOLDER, GRAPH_INDICATOR_FILENAME
    )
    gi = load_graph_indicator(graph_indicator_path)
    graphs = unbatch(adj_graph, gi)
    return graphs


def load_PROTEINS():
    graphs = dgl.data.TUDataset(
        "PROTEINS", raw_dir=os.path.join(SUBFOLDER, RAW_DATA_FOLDER)
    )
    graphs = [g for g in graphs]
    for g, _ in graphs:
        g.ndata["feat"] = g.ndata["node_attr"]
    return graphs


def dgl_to_fgnn(graphs):
    input_graphs = []
    for g in graphs:
        t = dgl_with_node_features_to_fgnn_tensor(g)
        input_graphs.append(t)
    return input_graphs


def prepare_fgnn_graphs(graphs):
    graphs, targets = zip(*graphs)
    input_graphs = dgl_to_fgnn(graphs)
    targets = [t.to(torch.float) for t in targets]
    return [(input_g, target_g) for input_g, target_g in zip(input_graphs, targets)]


def process():
    graphs_dgl = load_PROTEINS()
    output_path_dgl = os.path.join(
        SUBFOLDER, OUTPUT_FILENAME + OUTPUT_DGL_EXTENSION + ".pkl"
    )
    save_graphs(graphs_dgl, output_path_dgl)
    # Now, FGNN version
    graphs_fgnn = prepare_fgnn_graphs(graphs_dgl)
    output_path_fgnn = os.path.join(
        SUBFOLDER, OUTPUT_FILENAME + OUTPUT_FGNN_EXTENSION + ".pkl"
    )
    save_graphs(graphs_fgnn, output_path_fgnn)


def load_data(use_dgl):
    extension = OUTPUT_DGL_EXTENSION if use_dgl else OUTPUT_FGNN_EXTENSION
    output_path = os.path.join(SUBFOLDER, OUTPUT_FILENAME + extension + ".pkl")
    return tu_load_data(output_path, process)


if __name__ == "__main__":
    process()
