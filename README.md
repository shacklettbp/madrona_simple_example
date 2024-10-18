Madrona Simple Example
============================

This repository contains an implementation of a simple [GridWorld](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html) environment written in the [Madrona Engine](https://madrona-engine.github.io).

 The purpose of this repository is to provide a bare-minimum starting point for users to build their own simulators with minimal existing logic to get in their way (only one ECS archetype and system). Simply fork this repository, and start adding your own custom state and logic via new ECS components and systems.

Note that this repository doesn't include integration with Madrona's physics or rendering functionality. If you're interested in those features, or are unfamiliar with how to use the engine, please start with the [Madrona Escape Room](https://github.com/shacklettbp/madrona_escape_room) repository instead for a 3D environment example with plenty of documentation.

Build and Test
==============
First, make sure you have all the dependencies listed [here](https://github.com/shacklettbp/madrona#dependencies) (briefly, recent python and cmake, as well as Xcode or Visual Studio on MacOS or Windows respectively).

Next, fetch the repo (don't forget `--recursive`!):
```bash
git clone --recursive https://github.com/shacklettbp/madrona_simple_example.git
cd madrona_simple_example
```

Next, for Linux and MacOS: Run `cmake` and then `make` to build the simulator:
```bash
mkdir build
cd build
cmake ..
make -j # cores to build with
cd ..
```

Or on Windows, open the cloned repository in Visual Studio and build
the project using the integrated `cmake` functionality.

Now, setup the python components of the repository with `pip`:
```bash
pip install -e . # Add -Cpackages.madrona_simple_example.ext-out-dir=PATH_TO_YOUR_BUILD_DIR on Windows
```

You can test the simulator as follows (first, [install pytorch](https://pytorch.org/get-started/locally/)):
```bash
python scripts/run.py 32 # Simulate 32 worlds on the CPU
python scripts/run.py 32 --gpu # Simulate 32 worlds on the GPU
```
