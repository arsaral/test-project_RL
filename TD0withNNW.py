# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 18:49:27 2025

@author: USER
"""

import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------------------------------
# 1. Very small 1D environment
# -------------------------------------------------------
n_states = 5
terminal_state = 4

def step(state):
    """Agent always moves right."""
    next_state = min(state + 1, terminal_state)
    reward = 1.0 if next_state == terminal_state else 0.0
    done = (next_state == terminal_state)
    return next_state, reward, done

# -------------------------------------------------------
# 2. Neural network: V(s)
# -------------------------------------------------------
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # simplest possible NN

    def forward(self, s):
        return self.linear(s)

V = ValueNet()
optimizer = optim.Adam(V.parameters(), lr=0.01)

gamma = 0.99

# -------------------------------------------------------
# 3. TD(0) Learning Loop
# -------------------------------------------------------
for episode in range(200):
    state = 0

    while True:
        s_tensor = torch.tensor([[state]], dtype=torch.float32)

        next_state, reward, done = step(state)
        ns_tensor = torch.tensor([[next_state]], dtype=torch.float32)

        # TD target: r + γ V(next_state)
        with torch.no_grad():
            target = reward + gamma * V(ns_tensor) * (0.0 if done else 1.0)

        # Prediction V(s)
        value = V(s_tensor)

        loss = (value - target)**2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        if done:
            break

# -------------------------------------------------------
# 4. Print learned values
# -------------------------------------------------------
for s in range(n_states):
    #print(f"V({s}) =", float(V(torch.tensor([[s]], dtype=torch.float32))))
    state_tensor = torch.tensor([[float(s)]], dtype=torch.float32)
    value = V(state_tensor).detach().item()
    print(f"V({s}) = {value}")