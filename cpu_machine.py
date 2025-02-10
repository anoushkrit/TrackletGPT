import torch
import torch.distributed.rpc as rpc

# Check if CUDA is available outside of the JIT function
def check_cuda_availability():
    return torch.cuda.is_available()

# JIT-compiled function does not check for CUDA anymore
# @torch.jit.script
def my_script_add(tensor: torch.Tensor, scalar: int):
    # You can use `is_cuda_available` inside the JIT function
    return check_cuda_availability()

def run_worker1():
    
    # rpc.init_rpc("worker1", rank=1, world_size=2
    #              ,rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
    #     init_method="tcp://172.10.2.4:29500",  # Use CPU machine's IP and a free port
    # ))
    rpc.init_rpc("worker1", rank=1, world_size=2
                 ,rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        init_method="tcp://10.8.1.100:29500"
    ))
    print(1)
    
    rpc.shutdown()

if __name__ == "__main__":
    run_worker1()
