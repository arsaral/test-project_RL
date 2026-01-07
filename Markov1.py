# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 18:36:29 2025

@author: USER
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

# --- Environment setup ---
n_states = 5       # positions 0 to 4
n_actions = 2      # left, right
goal = 4

# --- Q-network ---
class QNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(QNet, self).__init__()
        self.fc = nn.Linear(n_states, n_actions)
    def forward(self, x):
        return self.fc(x)

# --- Helper function to one-hot encode state ---
def one_hot(state, n):
    vec = torch.zeros(n)
    vec[state] = 1.0
    return vec

# --- Initialize Q-network ---
q_net = QNet(n_states, n_actions)
optimizer = optim.Adam(q_net.parameters(), lr=0.01)
criterion = nn.MSELoss()

# --- Hyperparameters ---
gamma = 0.9   # discount factor
epsilon = 0.1 # exploration rate
n_episodes = 50
max_steps = 10

# --- Q-learning loop ---
for episode in range(n_episodes):
    state = 0
    for step in range(max_steps):
        state_vec = one_hot(state, n_states)
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, n_actions-1)
        else:
            with torch.no_grad():
                action = torch.argmax(q_net(state_vec)).item()
        
        # Take action
        if action == 0:
            next_state = max(0, state - 1)
        else:
            next_state = min(n_states-1, state + 1)
        
        reward = 1.0 if next_state == goal else 0.0
        
        # Q-learning target
        next_state_vec = one_hot(next_state, n_states)
        with torch.no_grad():
            target = reward + gamma * torch.max(q_net(next_state_vec))
        
        # Update Q-network
        output = q_net(state_vec)[action]
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        if state == goal:
            break

# --- Test trained policy ---
state = 0
path = [state]
while state != goal:
    state_vec = one_hot(state, n_states)
    action = torch.argmax(q_net(state_vec)).item()
    state = max(0, state-1) if action == 0 else min(n_states-1, state+1)
    path.append(state)

print("Learned path to goal:", path)