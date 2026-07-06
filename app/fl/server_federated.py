"""
Federated EchoJEPA — Flower Aggregation Server
Mode-agnostic: averages whatever parameter arrays clients send.
Mode selection (what to send/receive) is handled client-side.
Usage:
    python server_federated.py --num_rounds 50 --min_clients 2 --port 8080
"""
import argparse
from logging import INFO

import flwr as fl
from flwr.common.logger import log


def main():
    parser = argparse.ArgumentParser(description="EchoJEPA FL Server")
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--min_clients", type=int, default=2)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
    )

    log(INFO, f"Starting FL server on port {args.port}")
    log(INFO, f"Expecting {args.min_clients} clients, running {args.num_rounds} rounds")

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
