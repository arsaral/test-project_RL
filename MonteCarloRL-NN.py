# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 18:33:57 2025

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Monte Carlo Value Prediction using a Neural Network (Fast Version)
Environment: 1D world with 5 states (0-4), goal at 4.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

# ---------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------
n_states = 5
goal_state = 4

def step(state, action):
    """
    action = 0 → left
    action = 1 → right
    Returns: next_state, reward, done
    """
    if action == 0:
        next_state = max(0, state - 1)
    else:
        next_state = min(n_states - 1, state + 1)

    reward = 1 if next_state == goal_state else 0
    done = next_state == goal_state
    return next_state, reward, done

# ---------------------------------------------------------
# 2. Neural Network for V(s)
# ---------------------------------------------------------
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),   # input = state (scalar)
            nn.ReLU(),
            nn.Linear(32, 1)    # output = V(s)
        )
    def forward(self, x):
        return self.layers(x)

value_net = ValueNet()
optimizer = optim.Adam(value_net.parameters(), lr=0.01)
criterion = nn.MSELoss()

# ---------------------------------------------------------
# 3. Monte Carlo return calculation
# ---------------------------------------------------------
def compute_mc_returns(rewards, gamma=0.99):
    """
    For an episode with rewards [r0, r1, ..., rT],
    returns are G_t = r_t + gamma*r_{t+1} + ...
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

# ---------------------------------------------------------
# 4. Generate one episode (using a random policy)
# ---------------------------------------------------------
def generate_episode():
    states = []
    rewards = []

    state = 0
    while True:
        action = random.choice([0, 1])  # random policy
        next_state, reward, done = step(state, action)

        states.append(state)
        rewards.append(reward)

        if done:
            break
        state = next_state

    return states, rewards

# ---------------------------------------------------------
# 5. Training loop (fast Monte Carlo using batch updates)
# ---------------------------------------------------------
n_episodes = 300

for episode in range(n_episodes):
    states, rewards = generate_episode()
    returns = compute_mc_returns(rewards)

    # Convert to tensors
    states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(1)
    returns_tensor = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)

    # Neural network update
    optimizer.zero_grad()
    predictions = value_net(states_tensor)
    loss = criterion(predictions, returns_tensor)
    loss.backward()
    optimizer.step()

    if (episode+1) % 50 == 0:
        print(f"Episode {episode+1}, Loss = {loss.item():.4f}")

# ---------------------------------------------------------
# 6. Print learned value function
# ---------------------------------------------------------
print("\nEstimated state values V(s):")
for s in range(n_states):
    s_tensor = torch.tensor([[s]], dtype=torch.float32)
    v = value_net(s_tensor).item()
    print(f"V({s}) = {v:.3f}")