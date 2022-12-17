import sys 
import numpy as np
import torch
from madrona_simple_example import GridWorld

num_worlds = int(sys.argv[1])

enable_gpu_sim = False
if len(sys.argv) >= 3 and sys.argv[2] == '--gpu':
    enable_gpu_sim = True

array_shape = [5,6]
walls = np.zeros(array_shape)
rewards = np.zeros(array_shape)
walls[3, 2:] = 1
start_cell = np.array([4,5])
end_cell = np.array([[4,5]])
rewards[4, 0] = -1
rewards[4, 5] = 1

grid_world = GridWorld(num_worlds, start_cell, end_cell, rewards, walls, enable_gpu_sim, 0)
#grid_world.vis_world()

print(grid_world.observations.shape)

for i in range(5):
    print("Obs:")
    print(grid_world.observations)

    # "Policy"
    grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))
    #grid_world.actions[:, 0] = 3 # right to win given (4, 4) start

    print("Actions:")
    print(grid_world.actions)

    # Advance simulation across all worlds
    grid_world.step()
    
    print("Rewards: ")
    print(grid_world.rewards)
    print("Dones:   ")
    print(grid_world.dones)
    print()
