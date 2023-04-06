import madrona_python
import madrona_simple_example_python
import torch
import time
import sys

def error_usage():
    print("interactive.py (CUDA | CPU)")
    sys.exit(1)

if len(sys.argv) < 2:
    error_usage()

exec_mode_str = sys.argv[1]

if exec_mode_str == "CUDA":
    exec_mode = madrona_simple_example_python.ExecMode.CUDA
elif exec_mode_str == "CPU":
    exec_mode = madrona_simple_example_python.ExecMode.CPU
else:
    error_usage()

sim = madrona_simple_example_python.SimpleSimulator(
        exec_mode = exec_mode,
        gpu_id = 0,
        num_worlds = 16,
        num_agents_per_world = 5,
)

actions = sim.action_tensor().to_torch()
positions = sim.position_tensor().to_torch()
resets = sim.reset_tensor().to_torch()
print(actions.shape, actions.dtype)
print(positions.shape, actions.dtype)
print(resets.shape, resets.dtype)

while True:
    actions += torch.randn_like(positions)

    sim.step()

    print(positions)

    time.sleep(1)
