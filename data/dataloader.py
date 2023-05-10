import copy
from torch.utils.data import DataLoader

LOADER_ARGS = {"shuffle": True}


def get_loader_config(config):
    d = copy.deepcopy(LOADER_ARGS)
    d["batch_size"] = config["train"]["batch_size"]
    return d


def get_loader_from_graphs(graphs, config):
    add_config = get_loader_config(config)
    return DataLoader(graphs, **add_config)
