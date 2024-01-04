import flwr as fl
from client.strategy import strategy
from client.dataset import trainloaders, valloaders, testloader
from client.client import FlowerClient


def generate_client_fn(trainloaders, valloaders):
    def client_fn(cid: str):
        """Returns a FlowerClient containing the cid-th data partition"""

        return FlowerClient(
            trainloader=trainloaders[int(cid)], vallodaer=valloaders[int(cid)]
        )

    return client_fn


if __name__ == "__main__":
    
    client_fn_callback = generate_client_fn(trainloaders, valloaders)

    history = fl.simulation.start_simulation(
        client_fn=client_fn_callback,  # a callback to construct a client
        num_clients=100,  # total number of clients in the experiment
        config=fl.server.ServerConfig(num_rounds=10),  # let's run for 10 rounds
        strategy=strategy,  # the strategy that will orchestrate the whole FL pipeline
    )