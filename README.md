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

# Traffic Intersection Simulation

This project simulates a traffic intersection with AI-controlled traffic lights. The goal is to optimize traffic flow and minimize CO2 emissions.

## Project Structure

- `simulation.py`: Main simulation logic for traffic and visualization
- `simulation_config.py`: Configuration parameters for the simulation
- `base_model.py`: Base class for traffic control models
- `td_learning.py`: TD Learning model for traffic control
- `advanced_traffic_agent.py`: Advanced Deep Q-Learning model for traffic control
- `run_advanced_agent.py`: CLI tool to train and run the advanced agent
- `vehicle.py`: Vehicle class and related functions
- `traffic_signal.py`: Traffic signal implementation

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Running the Simulation

### Basic Simulation

```bash
python simulation.py
```

### Advanced Deep Q-Learning Agent

The project includes an advanced Deep Q-Learning agent with neural networks for traffic control.

#### Train a new agent

```bash
python run_advanced_agent.py train --episodes 100 --ticks 36000 --save-interval 10
```

#### Evaluate a trained agent

```bash
python run_advanced_agent.py evaluate --model-path models/advanced_agent_final.pt --num-evals 5
```

#### Run simulation with a trained agent

```bash
python run_advanced_agent.py simulate --model-path models/advanced_agent_final.pt --ticks-per-second 60
```

## Agent Types

### Flipping Model (Base Implementation)
A simple model that alternates traffic light states at fixed intervals.

### TD Learning Model
A model that uses Temporal Difference learning to optimize traffic flow.

### Advanced Deep Q-Learning Agent
The most advanced model using:
- Deep Q-Learning with neural networks
- Experience replay buffer for efficient learning
- Target network for stable updates
- Batch normalization and dropout for better generalization
- Traffic pattern analysis
- Detailed state representation

## Customization

The advanced agent can be customized with various hyperparameters:

```bash
python run_advanced_agent.py train --learning-rate 0.001 --gamma 0.99 --epsilon 1.0 --epsilon-decay 0.995 --min-epsilon 0.05 --batch-size 64 --action-duration 300
```

## Performance Tracking

The advanced agent tracks and plots:
- Loss over training
- Rewards per episode
- Exploration rate changes
- Traffic patterns

## License

This project is open source under the MIT license.
