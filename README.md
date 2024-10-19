# RLHF Driving Simulator

## Overview

The RLHF Driving Simulator is a Reinforcement Learning (RL) project designed to simulate driving dynamics and optimize control strategies using Deep Q-Learning (DQN). The goal is to create an intelligent agent that can navigate a virtual environment while learning from its interactions and improving its performance over time.

## Features

- **Reinforcement Learning Framework**: Utilizes the DQN algorithm to train an agent in a simulated driving environment.
- **Custom Driving Environment**: Built using OpenAI's Gym to provide a flexible interface for reinforcement learning tasks.
- **Continuous Action Space**: The agent can learn to control acceleration within a continuous range, allowing for more nuanced driving behavior.
- **Episodic Training**: The training process is structured in episodes, providing the agent with repeated opportunities to learn and improve.

## Getting Started

### Prerequisites

Ensure you have the following installed on your machine:

- Python 3.7 or higher
- [pip](https://pip.pypa.io/en/stable/) (Python package installer)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/arastuthakur/RLHF-DRIVING.git
   cd RLHF-DRIVING
   ```
### Create a virtual environment (optional but recommended):

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
```
### Install the required packages:

```bash
pip install -r requirements.txt
```
### Folder Structure

```plaintext
RLHF-DRIVING/
├── agent/
│   ├── dqn.py             # DQN agent implementation
│   └── __init__.py        # Initialization file for the agent module
├── environment/
│   ├── driving_env.py     # Custom driving environment using OpenAI Gym
│   └── __init__.py        # Initialization file for the environment module
├── requirements.txt        # Python package dependencies
├── train.py                # Main training script
└── README.md               # Project documentation
```
### Usage

To train the DQN agent in the driving simulator, run the following command:

```bash
python train.py
```
This will initiate the training process over a specified number of episodes. You can monitor the total rewards obtained by the agent in each episode.

### How It Works

- **Driving Environment**: The `DrivingEnv` class simulates the driving dynamics, including state representation (position and velocity), action space (acceleration), and a reward mechanism based on distance from the goal.

- **DQN Agent**: The `DQNAgent` class implements the Deep Q-Learning algorithm, using experience replay and a target network to stabilize training. It learns to optimize its actions to maximize cumulative rewards.

- **Training Loop**: The training loop in `train.py` iteratively updates the agent based on its experiences in the environment. It captures the state, action, reward, and next state for each time step and trains the agent using mini-batch updates.

### Contributing

Contributions are welcome! If you would like to contribute to the project, please fork the repository and submit a pull request. Ensure your code adheres to the project's coding standards and includes appropriate documentation.

### License

This project is not licensed under any specific license. Feel free to use the code as you see fit, but attribution to the original authors is appreciated.

### Acknowledgements

- [OpenAI Gym](https://gym.openai.com/) - For providing a toolkit for developing and comparing reinforcement learning algorithms.
- [PyTorch](https://pytorch.org/) - For its flexibility and performance in building deep learning models.

### Contact

For any questions or inquiries regarding this project, please contact me at arustuthakur@gmail.com.
