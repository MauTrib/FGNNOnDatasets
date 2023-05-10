import dgl
import torch


def dgl_to_dense(graph: dgl.graph):
    """Converts a dgl graph to a dense adjacency matrix."""
    N = graph.num_nodes()
    dense_adj = torch.zeros((N, N))
    src, dst = graph.edges()
    dense_adj[src, dst] = 1
    return dense_adj


def is_sym_matrix(A):
    """
    Checks if a matrix is symmetric
    """
    return torch.allclose(A, A.transpose(-2, -1))


def is_sym_graph(graph: dgl.graph):
    """
    Checks if a graph is symmetric
    """
    return is_sym_matrix(dgl_to_dense(graph))


def is_sym(object):
    """
    Checks if an object is symmetric
    """
    if isinstance(object, dgl.DGLGraph):
        return is_sym_graph(object)
    else:
        return is_sym_matrix(object)


def dgl_adj_to_fgnn_tensor(graph: dgl.graph):
    """Converts a dgl graph to a fgnn-usable tensor B :
    - B[:,:,1] = dense adjacency matrix
    - B[:,:,0] = 0 everywhere except for the diagonal where it is the degree of the node
    """
    assert is_sym(
        graph
    ), "The graph is assumed to be symmetric, which is not the case here"
    dense_adj = dgl_to_dense(graph)

    degrees = torch.sum(dense_adj, dim=1)
    degrees = torch.diag(degrees)
    return torch.stack((degrees, dense_adj), dim=2)


def dgl_with_node_features_to_fgnn_tensor(graph: dgl.graph):
    """Converts a dgl graph to a fgnn-usable tensor B :
    - B[:,:,0] = 0 everywhere except for the diagonal where it is the degree of the node
    - B[:,:,1] = dense adjacency matrix
    - B[:,:,2:] = features (node features on diagonal). Node features are assumed to be in graph.ndata["feat"].
    """
    assert is_sym(
        graph
    ), "The graph is assumed to be symmetric, which is not the case here"
    base_tensor = dgl_adj_to_fgnn_tensor(graph)
    keys = graph.ndata.keys()
    node_features = graph.ndata["feat"]
    # node_features is of shape [n_nodes, n_features]
    # We want to put it on the diagonal (of the first two dimensions) of a tensor of shape [n_nodes, n_nodes, n_features]
    # All other values must be 0
    N = graph.num_nodes()
    node_features_tensor = torch.zeros((N, N, node_features.shape[1])).to(
        node_features.dtype
    )
    node_features_tensor[torch.arange(N), torch.arange(N)] = node_features

    return torch.cat((base_tensor, node_features_tensor), dim=2)
