import torch.distributed.rpc as rpc
from datasets.remote_dataloader import RemoteDataLoader  # Import the remote DataLoader class

def run_cpu_server():

    # Instantiate the remote DataLoader
    remote_loader = RemoteDataLoader()

    rpc.init_rpc("cpu_server", rank=-1, world_size=None)  # Start RPC on CPU machine
    print("CPU Data Server is running...")

    rpc.shutdown()  # Keep running until shutdown

if __name__ == "__main__":
    run_cpu_server()