<p align="center">
 <img height=350px src="./simulation-output.png" alt="Simulation output">
</p>

<h1 align="center">Basic Traffic Intersection Simulation</h1>

<div align="center">

[![Python version](https://img.shields.io/badge/python-3.1+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<h4>A simulation developed from scratch using Pygame to simulate the movement of vehicles across a traffic intersection having traffic lights with a timer.</h4>

</div>

-----------------------------------------
### Description

* It contains a 4-way traffic intersection with traffic signals controlling the flow of traffic in each direction. 
* Each signal has a timer on top of it which shows the time remaining for the signal to switch from green to yellow, yellow to red, or red to green. 
* Vehicles such as cars, bikes, buses, and trucks are generated, and their movement is controlled according to the signals and the vehicles around them. 
* This simulation can be further used for data analysis or to visualize AI or ML applications. 

Find a step-by-step guide to build this simulation [here](https://towardsdatascience.com/traffic-intersection-simulation-using-pygame-689d6bd7687a).

------------------------------------------
### Demo

The video below shows the final output of the simulation.

<p align="center">
    <img src="./Demo.gif">
</p>

------------------------------------------
### Prerequisites

[Python 3.1+](https://www.python.org/downloads/)

------------------------------------------
### Installation

 * Step I: Clone the Repository
```sh
      $ git clone https://github.com/mihir-m-gandhi/Basic-Traffic-Intersection-Simulation
```
  * Step II: Install the required packages
```sh
      # On the terminal, move into Basic-Traffic-Intersection-Simulation directory
      $ cd Basic-Traffic-Intersection-Simulation
      $ pip install pygame
```
* Step III: Run the code
```sh
      # To run simulation
      $ python simulation.py
```

------------------------------------------
### Author

Mihir Gandhi - [mihir-m-gandhi](https://github.com/mihir-m-gandhi)

------------------------------------------
### License
This project is licensed under the MIT - see the [LICENSE](./LICENSE) file for details.

## Advanced State Representation

The simulation now supports an advanced state representation using dimensionality reduction techniques:

### Features

- **Data Collection**: Automatically collects state data during simulation runs
- **Autoencoder Compression**: Uses a neural network autoencoder with a bottleneck architecture to compress the state space when TensorFlow is available
- **PCA Fallback**: Falls back to PCA-based compression when TensorFlow is not available
- **Rich State Information**: Captures detailed information about vehicles including:
  - Counts by vehicle type (car, bus, truck, bike) in each zone
  - Average speed of vehicles in each zone
  - Distance metrics to the intersection

### Architecture

The system uses a modular approach:

1. **State Compressor Module**: The `state_compressor.py` file handles data collection, model training, and state compression
2. **Separation of Concerns**: Training happens outside the simulation, providing a clean architecture
3. **Progressive Learning**: Data is collected during regular simulation runs, allowing the model to improve over time
4. **GPU Acceleration**: The autoencoder automatically utilizes GPU acceleration when available for faster training and inference

### GPU Acceleration

The state compressor is optimized to take advantage of GPU acceleration when available:

- **Automatic Detection**: The system automatically detects and configures available GPUs
- **Mixed Precision**: Uses mixed precision (float16/float32) training on compatible GPUs for faster performance
- **Memory Growth**: Enables memory growth to prevent TensorFlow from allocating all GPU memory at once
- **Optimized Architecture**: Uses a deeper network architecture that benefits from GPU acceleration
- **Efficient Inference**: Optimized prediction path for compressed state representation

### How to Use

To enable the advanced state representation:

```python
# In your main function or script
reward = simulate(model, USE_ADVANCED_STATE=True)
```

To train a new compressor model:

```bash
# Run the training utility with default settings
python state_compressor.py --train

# Force retraining even if a model exists
python state_compressor.py --train --force

# Specify minimum number of samples required for training
python state_compressor.py --train --min-samples 2000

# Optimize for GPU performance with larger batch size
python state_compressor.py --train --batch-size 128
```

### Requirements

- NumPy (required)
- TensorFlow 2.x (optional, for autoencoder-based compression and GPU acceleration)

If TensorFlow is not available, the system will automatically fall back to a NumPy-based PCA implementation.