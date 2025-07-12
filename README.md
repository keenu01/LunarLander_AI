# 🚀 Lunar Lander AI

A Deep Q-Learning agent implemented in PyTorch that learns to solve OpenAI Gym's LunarLander-v3 environment.

---

## 🌙 Project Overview

This project trains a reinforcement learning agent using Deep Q-Network (DQN) with experience replay to land a lunar module safely in the LunarLander-v3 environment.

---

## 🏗️ Structure

- `model.py`: Defines the neural network architecture.
- `main.py`: Contains the training loop, replay buffer, epsilon-greedy policy, and interaction with the environment.
- Trains the agent over 500 episodes with target network updates and epsilon decay.
- Plots training reward progress after completion.

---

## 🧠 Neural Network Architecture

- Simple fully connected feed-forward network with 2 hidden layers using ReLU activation.
- Input size: 8 (state space size of LunarLander-v3)
- Hidden layer size: 150 units
- Output size: 4 (action space size)

---

## 🚀 Getting Started

1. Clone the repository

```bash
git clone https://github.com/your-username/LunarLander_AI.git
cd LunarLander_AI
```

2. Install dependencies

```bash
pip install torch gymnasium matplotlib numpy
```

3. Run training

```bash
python main.py
```

The training progress will be displayed in the console, and a plot of total rewards per episode will be shown at the end.

---

## 🔧 Hyperparameters

- Gamma (discount factor): 0.99
- Epsilon start: 1.0 (for exploration)
- Epsilon end: 0.05
- Epsilon decay: 500 frames
- Learning rate: 0.001
- Batch size: 64
- Replay memory size: 10000
- Number of episodes: 500
- Max steps per episode: 400
- Target network update frequency: every 1000 frames

---

## 📈 Results

The agent learns to successfully land the lunar module over time. Training progress is monitored via episode reward plots.

---

## 🧾 Requirements

- Python 3.7 or later
- torch
- gymnasium
- matplotlib
- numpy

---

## 📁 Project Files

```
├── model.py       # Neural network definition
├── main.py        # Training and environment interaction loop
├── README.md      # Project overview and instructions
└── requirements.txt # (Optional) dependencies list
```

---

## 🧑‍💻 Credits

Based on the OpenAI Gym LunarLander environment and classic DQN reinforcement learning methods.

---

## 📜 License

MIT License

---
