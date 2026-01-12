# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 10:51:04 2025

@author: USER
"""

#1. Imports and setup
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

#2. Environment (1D random walk)

#State is a single scalar.

def step(s, a):
    noise = 0.1 * np.random.randn()
    return s + a + noise


#Actions will be continuous scalars.

#3. True (hidden) human reward

#This is never shown to the learner.

def true_reward(traj):
    return -np.sum(np.abs(traj))

#4. Policy network (very small)

#Maps state → action mean.

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, s):
        return self.net(s)


#We will add exploration manually.

#5. Reward network (neural reward model)

#Maps state → scalar reward.

class RewardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, s):
        return self.net(s)


#Trajectory reward = sum of per-state rewards.

#6. Rollout trajectories with current policy
def rollout(policy, start=2.0, horizon=10):
    s = start
    traj = []

    for _ in range(horizon):
        s_tensor = torch.tensor([[s]], dtype=torch.float32)
        a = policy(s_tensor).item()
        a += 0.3 * np.random.randn()   # exploration
        s = step(s, a)
        traj.append(s)

    return np.array(traj)

#7. Preference oracle (simulated human)
def human_pref(traj_a, traj_b):
    return true_reward(traj_a) > true_reward(traj_b)

#8. Preference loss (Bradley–Terry)
def preference_loss(reward_net, traj_a, traj_b, y):
    sa = torch.tensor(traj_a[:, None], dtype=torch.float32)
    sb = torch.tensor(traj_b[:, None], dtype=torch.float32)

    Ra = reward_net(sa).sum()
    Rb = reward_net(sb).sum()

    p = torch.sigmoid(Ra - Rb)
    target = torch.tensor(float(y))

    return nn.functional.binary_cross_entropy(p, target)

#9. Policy loss (REINFORCE-style, tiny)

#We maximize learned reward.

def policy_loss(policy, reward_net, traj):
    states = torch.tensor(traj[:, None], dtype=torch.float32)
    rewards = reward_net(states).detach().sum()
    actions = policy(states)

    # simple squared action penalty to stabilize
    return -rewards + 0.01 * (actions ** 2).mean()


#This is intentionally simple and biased — the goal is clarity.

#10. Training loop (reward ↔ policy)
policy = Policy()
reward_net = RewardNet()

policy_opt = optim.Adam(policy.parameters(), lr=1e-3)
reward_opt = optim.Adam(reward_net.parameters(), lr=1e-3)

for iteration in range(50):

    # --- collect data ---
    traj_a = rollout(policy)
    traj_b = rollout(policy)

    # >>> ADD THIS LINE <<<
    print(
        f"Iter {iteration:02d} | "
        f"True reward(A): {true_reward(traj_a):.2f}"
    )
    
    pref = human_pref(traj_a, traj_b)

    # --- update reward model ---
    reward_opt.zero_grad()
    r_loss = preference_loss(reward_net, traj_a, traj_b, pref)
    r_loss.backward()
    reward_opt.step()

    # --- update policy ---
    policy_opt.zero_grad()
    p_loss = policy_loss(policy, reward_net, traj_a)
    p_loss.backward()
    policy_opt.step()

    if iteration % 10 == 0:
        print(
            f"Iter {iteration:02d} | "
            f"Reward loss {r_loss.item():.3f} | "
            f"Policy loss {p_loss.item():.3f}"
        )