# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 13:28:48 2025

@author: ARS
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

# --- One-hot encoding ---
def one_hot(state, n):
    vec = torch.zeros(n)
    vec[state] = 1.0
    return vec

# --- Initialize network ---
q_net = QNet(n_states, n_actions)
optimizer = optim.Adam(q_net.parameters(), lr=0.01)
criterion = nn.MSELoss()

# --- Hyperparameters ---
gamma = 0.9
epsilon = 0.1
n_episodes = 50
max_steps = 10

# --- Training loop ---
for episode in range(n_episodes):
    state = 0
    print(f"\n=== Episode {episode} ===")
    
    for step in range(max_steps):
        state_vec = one_hot(state, n_states)
        
        # -------- PRINT Q-VALUES BEFORE TAKING ACTION --------
        q_vals = q_net(state_vec)
        print(f"Step {step} | State: {state} | Q-values: {q_vals.tolist()}")
        
        # Epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, n_actions-1)
        else:
            with torch.no_grad():
                #action = torch.argmax(q_net(state_vec)).item()
                action = torch.argmax(q_vals).item()
        
        # Take action
        next_state = max(0, state-1) if action == 0 else min(n_states-1, state+1)
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
        
        # --- Print transition ---
        print(f"   Action: {action}, Reward: {reward}, Next state: {next_state}, Loss: {loss.item():.4f}")

        state = next_state
        if state == goal:
            break

# --- Step-by-step test ---
state = 0
print("\nStep-by-step test of learned policy:")
step_count = 0
while state != goal:
    # Print grid
    grid = ['.']*n_states
    grid[state] = 'A'   # Agent
    grid[goal] = 'G'    # Goal
    print(f"Step {step_count}: " + ' '.join(grid))
    
    # Take action
    state_vec = one_hot(state, n_states)
    action = torch.argmax(q_net(state_vec)).item()
    state = max(0, state-1) if action == 0 else min(n_states-1, state+1)
    step_count += 1

# Final step
grid = ['.']*n_states
grid[state] = 'A'
grid[goal] = 'G'
print(f"Step {step_count}: " + ' '.join(grid))
print("Agent reached the goal!")