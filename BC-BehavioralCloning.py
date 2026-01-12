# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 19:20:45 2025

@author: USER
"""

import torch
import torch.nn as nn
import torch.optim as optim
#2. Define the Expert
#This is the ground truth behavior.

def expert_policy(x):
    # x is a tensor
    return (x > 0).float()

#3. Generate Expert Demonstrations
#This is the offline dataset.

torch.manual_seed(0)

N = 1000

# Observations
obs = torch.randn(N, 1)

# Expert actions
actions = expert_policy(obs)

#At this point we have:
#(obs[i], actions[i])
#These are the only thing BC will ever use.

#4. Define the Policy Network (Learner)
#A minimal neural network:

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
#Interpretation:
#Output is a probability of action = 1

#5. Train with Behavioral Cloning
#This is supervised learning, nothing more.

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

for epoch in range(200):
    pred_actions = policy(obs)
    loss = loss_fn(pred_actions, actions)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.4f}")
# =============================================================================
# Key philosophical point:
# No environment
# No reward
# No rollout
# No exploration
# 
# Just imitation.
# =============================================================================

#6. Test the Learned Policy
#Now we deploy the learner.

test_obs = torch.tensor([[-2.0], [-0.5], [0.2], [1.5]])

with torch.no_grad():
    probs = policy(test_obs)
    predicted_actions = (probs > 0.5).float()

print("Observations:", test_obs.squeeze().tolist())
print("Predicted actions:", predicted_actions.squeeze().tolist())

#Expected output:
#Observations: [-2.0, -0.5, 0.2, 1.5]
#Predicted actions: [0, 0, 1, 1]

# =============================================================================
# %runfile
# Epoch 0, Loss = 0.6989
# Epoch 50, Loss = 0.1699
# Epoch 100, Loss = 0.0726
# Epoch 150, Loss = 0.0516
# Observations: [-2.0, -0.5, 0.20000000298023224, 1.5]
# Predicted actions: [0.0, 0.0, 1.0, 1.0]
# =============================================================================
