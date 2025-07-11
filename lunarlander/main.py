import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch
from model import Network
import torch.optim as optim
import random
import numpy as np
from collections import deque
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 500
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
NUM_EPISODES = 500
MAX_STEPS = 400
TARGET_UPDATE_FREQ = 1000
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool)
        )
    
    def __len__(self):
        return len(self.buffer)

# Epsilon decay function
def epsilon_by_frame(frame_idx):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * frame_idx / EPSILON_DECAY)

# Main training loop
def train():
    net = Network(input_layer=8, hidden_layer=150, output_layer=4)
    target_net = Network(input_layer=8, hidden_layer=150, output_layer=4)
    target_net.load_state_dict(net.state_dict())  
    target_net.eval()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)
    frame_idx = 0
    all_rewards = []

    env = gym.make("LunarLander-v3", render_mode='human')  # create env once

    for episode in range(NUM_EPISODES):
        state, info = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            epsilon = epsilon_by_frame(frame_idx)
            frame_idx += 1

            state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = net(state_v)
                action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, info = env.step(action)

           

            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(replay_buffer) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

               
                q_values = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                action = torch.argmax(q_values).item()
                next_q_values = target_net(next_states).max(1)[0]
                expected_q_values = rewards + GAMMA * next_q_values * (~dones)
                loss_fn = nn.MSELoss()
                loss = loss_fn(q_values, expected_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            env.render()  # render after each step so you can see the agent training

            if done:
                break

        all_rewards.append(total_reward)
        print(f"Episode {episode + 1}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
        print(f"Loss: {loss.item():.4f}, Q: {q_values.mean().item():.2f}")

        if frame_idx % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(net.state_dict())

    env.close()  # close environment once training is fully done
    return all_rewards


if __name__ == "__main__":
    
    all_rewards= train()
    
    import matplotlib.pyplot as plt
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()
