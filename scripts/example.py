import madrona_python
import madrona_simple_example_python
import torch
import time

sim = madrona_simple_example_python.SimpleSimulator(
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
