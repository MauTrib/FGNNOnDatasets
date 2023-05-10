import torch

from data.proteins_process import load_data as load_data_proteins

from data.dataloader import get_loader_from_graphs

USE_DGL = {"rsfgnn": False}


def get_graphs(config):
    problem = config["problem"].lower()
    use_dgl = USE_DGL[config["model"]["pipeline"]]
    if problem[:6] == "dummy-":
        problem = problem[6:]
    if problem == "proteins":
        graphs = load_data_proteins(use_dgl=use_dgl)
    else:
        raise NotImplementedError(f"Problem {problem} not implemented yet.")
    return graphs


def get_loader(config):
    graphs = get_graphs(config)
    loader = get_loader_from_graphs(graphs, **config["loader"])
    return loader


def split_data(data, config):
    train_split = config["train"]["train_split"]
    val_split = config["train"]["val_split"]
    test_split = 1 - train_split - val_split
    if config["problem"][:6] == "dummy-":  # If testing
        train_split = 1
        val_split = 1
        test_split = len(data) - train_split - val_split
    data_train, data_val, data_test = torch.utils.data.random_split(
        data, [train_split, val_split, test_split]
    )

    print(
        f"Splits (train/val/test) are : {len(data_train)}, {len(data_val)}, {len(data_test)}, total : {len(data_train) + len(data_val) + len(data_test)} ({len(data)})"
    )
    return data_train, data_val, data_test


def get_loaders(config):
    graphs = get_graphs(config)
    data_train, data_val, data_test = split_data(graphs, config)
    return data_train, data_val, data_test
