# TinyMPCReference
A reference implementation of TinyMPC in pure Python. The original C++ implementation can be found [here](https://github.com/TinyMPC/TinyMPC) . 

## Features

- Quadrotor hover control
- Trajectory tracking with figure-8 pattern
- Adaptive MPC with penalty parameter (rho) adaptation
- Wind disturbance simulation
- Performance metrics and visualization tools

## Installation

### Prerequisites
```bash
# Required Python packages
numpy
matplotlib
scipy
autograd
```

### Setup
```bash
git clone https://github.com/A2R-Lab/TinyMPCReference.git
cd TinyMPCReference
```

## Usage

Basic usage examples:
```bash
# Run hover control
python3 examples/hover/hover.py

# Run hover with rho adaptation
python3 examples/hover/hover.py --adapt

# Run trajectory tracking
python3 examples/traj/traj.py

# Run trajectory with rho adaptation
python3 examples/traj/traj.py --adapt

# Run with wind disturbance
python3 examples/traj/traj.py --wind
```

### Command Line Arguments

- `--adapt`: Enable rho adaptation
- `--wind`: Enable wind disturbance (functionality enabled only for trajectory tracking for now) 

## Project Structure
```
├── src/
│   ├── quadrotor.py         # Quadrotor dynamics
│   ├── tinympc.py          # MPC implementation
│   └── rho_adapter.py      # Adaptive parameter handling
├── utils/
│   ├── visualization.py    # Plotting utilities
│   ├── traj_simulation.py  # Simulation tools
│   ├── hover_simulation.py  # Simulation tools
│   └── reference_trajectories.py
├── examples/
│   ├── hover/
│   │   └── hover.py        # Hover control example
│   └── traj/
│       └── traj.py         # Trajectory tracking example
└── data/                   # Simulation results and metrics
```



## Contact

For any questions/concerns, free free to open issues/PRs, or contact the [developers](mailto:iam2141@columbia.edu) of this repository. 
