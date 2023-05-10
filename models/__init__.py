from models.fgnn import RSFGNN
from models.fgnn_model.fgnn import RS_Graph_Embedding, RS_Node_Embedding


def get_model(config):
    model = None
    name = config["model"]["name"]
    embedding = config["model"]["embedding"]
    if name == "rsfgnn":
        if embedding == "node":
            model = RS_Node_Embedding
        elif embedding == "graph":
            model = RS_Graph_Embedding
    if model is not None:
        model_params = config["model"]["params"][name]
        return model(**model_params)
    raise NotImplementedError(
        f"Model {name} with embedding {embedding} not implemented."
    )


def get_pipeline(config):
    pipeline = config["model"]["pipeline"]
    if pipeline == "rsfgnn":
        return RSFGNN
    raise NotImplementedError(f"Pipeline {pipeline} not implemented.")


def config_pipeline_from_model(pipeline, model, config):
    optim_args = config["train"]["optim"]
    return pipeline(model, optim_args)


def configure_pipeline(config):
    model = get_model(config)
    pipeline = get_pipeline(config)
    return config_pipeline_from_model(pipeline, model, config)
