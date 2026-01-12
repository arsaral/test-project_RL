# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 15:44:39 2025

@author: USER
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

# -----------------------------
# 1. Environment
# -----------------------------
n_states = 5
goal = 4

def step(state, action):
    """Take action in 1D world."""
    if action == 0:  # left
        next_state = max(0, state - 1)
    else:  # right
        next_state = min(n_states - 1, state + 1)

    reward = 1.0 if next_state == goal else 0.0
    done = next_state == goal
    return next_state, reward, done

# -----------------------------
# 2. Policy Network (softmax)
# -----------------------------
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 2)     # two actions: left/right
        )

    def forward(self, state):
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)

policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# -----------------------------
# 3. Generate episode using current policy
# -----------------------------
def generate_episode():
    log_probs = []
    rewards = []

    state = 0  # start at position 0

    for t in range(20):  # max 20 steps
        s = torch.tensor([[state]], dtype=torch.float32)
        probs = policy(s)

        m = torch.distributions.Categorical(probs)
        action = m.sample()

        log_prob = m.log_prob(action)
        log_probs.append(log_prob)

        next_state, reward, done = step(state, action.item())
        rewards.append(reward)

        state = next_state
        if done:
            break

    return log_probs, rewards

# -----------------------------
# 4. REINFORCE update
# -----------------------------
def update_policy(log_probs, rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns)

    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss += -log_prob * G  # maximize expected return

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# -----------------------------
# 5. Train the policy
# -----------------------------
for episode in range(300):
    log_probs, rewards = generate_episode()
    loss = update_policy(log_probs, rewards)

    if episode % 50 == 0:
        print(f"Episode {episode}, loss={loss:.3f}")

print("Training finished!")