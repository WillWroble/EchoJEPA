"""
Federated EchoJEPA — Flower ClientApp (deployment runtime)

Thin wrapper around JEPATrainer/EchoJEPAClient from
app.vjepa.train_federated. Each SuperNode passes its site YAML via
--node-config 'config-path="..."'. The trainer is built once per run
and cached at module level; client_fn is invoked every round.
"""
import torch
import yaml
from flwr.client import ClientApp

from app.vjepa.train_federated import JEPATrainer, EchoJEPAClient

_CLIENT = None


def client_fn(context):
    global _CLIENT
    if _CLIENT is None:
        fname = context.node_config["config-path"]
        with open(fname) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        cfgs_fl = config.get("federated")
        trainer = JEPATrainer(config, torch.device("cuda:0"))
        _CLIENT = EchoJEPAClient(
            trainer,
            cfgs_fl.get("mode"),
            cfgs_fl.get("local_steps"),
            cfgs_fl.get("sync_momentum", 0.998),
        )
    return _CLIENT.to_client()


app = ClientApp(client_fn=client_fn)
