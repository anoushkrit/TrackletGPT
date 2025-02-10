import torch
import torch.distributed.rpc as rpc

# def check_cuda_availability():
#     return torch.cuda.is_available()

# # JIT-compiled function does not check for CUDA anymore
# @torch.jit.script
def my_script_add(tensor: torch.Tensor, scalar: int):
    # You can use `is_cuda_available` inside the JIT function
    return torch.cuda.is_available()

def run_worker0():
    # Initialize RPC with the rank and world size
    
    print(1)
    rpc.init_rpc("worker0", rank=0, world_size=2)
    print(1)

    # Perform RPC call to worker1
    for i in range(10):
        ret = rpc.rpc_sync("worker1", my_script_add, args=(torch.ones(2), 3))
        print(ret)

    rpc.shutdown()

if __name__ == "__main__":
    run_worker0()
