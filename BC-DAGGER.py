# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 19:43:49 2025

@author: USER
"""

import torch
import torch.nn as nn
import torch.optim as optim

#2. Expert Policy (Same)
def expert_policy(x):
    return (x > 0).float()

#3. Policy Network (Same)
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
#Nothing new so far.
#DAgger does not require a different model.

#4. Initial Expert Dataset (Bootstrap)
#DAgger always starts with some BC.

torch.manual_seed(0)

N = 200
obs_dataset = torch.randn(N, 1)
act_dataset = expert_policy(obs_dataset)
#This prevents the learner from acting randomly at the start.

#5. Training Function (Reusable)
#We will retrain the policy repeatedly.

def train_policy(policy, obs, acts, epochs=100):
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    for _ in range(epochs):
        preds = policy(obs)
        loss = loss_fn(preds, acts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#6. DAgger Loop (Core Difference)
#This is the entire philosophy of DAgger in code.

policy = Policy()

for dagger_iter in range(5):

    # Step 1: Train on current dataset
    train_policy(policy, obs_dataset, act_dataset)

    # Step 2: Learner acts (produces its own data)
    new_obs = torch.randn(100, 1)

    with torch.no_grad():
        learner_actions = (policy(new_obs) > 0.5).float()

    # Step 3: Expert corrects learner's states
    expert_actions = expert_policy(new_obs)

    # Step 4: Aggregate dataset
    obs_dataset = torch.cat([obs_dataset, new_obs])
    act_dataset = torch.cat([act_dataset, expert_actions])

    print(f"DAgger iteration {dagger_iter}: dataset size = {len(obs_dataset)}")

#7. What Is Actually Happening (Conceptual Flow)
# =============================================================================
# Let’s slow this down.
# 
# Iteration Structure
# Each DAgger iteration does:
# 
# vbnet
# Kodu kopyala
# Train → Act → Correct → Aggregate
# Key philosophical change
# The learner is now inside the data-generation loop
# 
# The expert is responding to learner-induced situations
# 
# This is the moral correction discussed earlier.
# =============================================================================

#8. Why This Is Not Just “More BC”
# =============================================================================
# In BC:
# Dataset is fixed
# Learner is passive
# In DAgger:
# Dataset grows
# Learner shapes what it must learn
# The learner is trained on:
# “What I should have done, given what I actually did.”
# =============================================================================

#9. Final Test (Deployment)
#Same as BC — but behavior is more robust.

test_obs = torch.tensor([[-2.0], [-0.1], [0.1], [2.0]])

with torch.no_grad():
    probs = policy(test_obs)
    actions = (probs > 0.5).float()

print("Observations:", test_obs.squeeze().tolist())
print("Predicted actions:", actions.squeeze().tolist())

# =============================================================================
# %runfile 
# DAgger iteration 0: dataset size = 300
# DAgger iteration 1: dataset size = 400
# DAgger iteration 2: dataset size = 500
# DAgger iteration 3: dataset size = 600
# DAgger iteration 4: dataset size = 700
# Observations: [-2.0, -0.10000000149011612, 0.10000000149011612, 2.0]
# Predicted actions: [0.0, 0.0, 1.0, 1.0]
# =============================================================================
