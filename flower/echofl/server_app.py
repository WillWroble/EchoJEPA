"""
Federated EchoJEPA — Flower ServerApp (deployment runtime)
Mode-agnostic: averages whatever parameter arrays clients send.
Mode selection (what to send/receive) is handled client-side.
Replaces app/fl/server_federated.py. num_rounds comes from
[tool.flwr.app.config] in pyproject.toml (override with
`flwr run ... --run-config num-server-rounds=N`).
"""
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


def server_fn(context):
    num_rounds = int(context.run_config["num-server-rounds"])
    strategy = FedAvg(
        min_fit_clients=2,
        min_available_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))


app = ServerApp(server_fn=server_fn)
