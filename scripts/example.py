import madrona_python
import madrona_simple_example_python
import torch
import time

sim = madrona_simple_example_python.SimpleSimulator(
        gpu_id = 0,
        num_worlds = 4,
        num_rectangles_per_world = 5,
        bounds_min_x = -5,
        bounds_max_x = 5,
        bounds_min_y = -5,
        bounds_max_y = 5,
        min_width = 1,
        max_width = 3,
        min_height = 1,
        max_height = 3,
)

# Only deal with x, y components
actions = sim.action_tensor().to_torch()[:, :, 0:2]
positions = sim.position_tensor().to_torch()[:, :, 0:2]

overlaps = sim.num_overlaps_tensor().to_torch()
resets = sim.reset_tensor().to_torch()
print(actions.shape, actions.dtype)
print(positions.shape, actions.dtype)
print(resets.shape, resets.dtype)

while True:
    actions += torch.randn_like(positions)

    sim.step()

    print(positions)
    print(overlaps)

    time.sleep(1)
