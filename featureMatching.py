# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 11:43:01 2025

@author: USER
"""

#Step 1 — Expert trajectories (fixed data)
import numpy as np

# Expert demonstrations (state, action)
expert_trajectories = [
    [(0, +1), (1, +1), (2, +1)],
    [(1, +1), (2, +1), (2, +1)],
    [(0, +1), (1, +1), (1, +1)],
]

#Step 2 — Feature definition
def phi(state, action):
    return np.array([state, action], dtype=float)

#Step 3 — Feature expectation of expert
def feature_expectation(trajectories):
    features = []
    for traj in trajectories:
        for (s, a) in traj:
            features.append(phi(s, a))
    return np.mean(features, axis=0)

mu_expert = feature_expectation(expert_trajectories)
print("Expert feature expectation:", mu_expert)


# =============================================================================
# Example output:
# Expert feature expectation: [1.33 1.  ]
# Interpretation:
# Expert stays in higher states
# Expert strongly prefers action +1
# =============================================================================

#Step 4 — Parametric policy (softmax over actions)
#This policy has no reward, only parameters.

actions = [-1, +1]

def policy(state, theta):
    logits = np.array([theta @ phi(state, a) for a in actions])
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs

#Step 5 — Sample trajectories from policy
def sample_policy_trajectories(theta, n_traj=20, T=3):
    trajectories = []
    for _ in range(n_traj):
        s = np.random.choice([0,1,2])
        traj = []
        for _ in range(T):
            probs = policy(s, theta)
            a = np.random.choice(actions, p=probs)
            traj.append((s, a))
            s = np.clip(s + a, 0, 2)
        trajectories.append(traj)
    return trajectories

#Step 6 — Feature matching loss
#This is the entire learning objective:
def feature_matching_loss(theta):
    trajs = sample_policy_trajectories(theta)
    mu_pi = feature_expectation(trajs)
    return np.linalg.norm(mu_pi - mu_expert)

#Step 7 — Policy learning = feature matching
#Simple gradient-free optimization (finite differences):
theta = np.random.randn(2)
lr = 0.1

for i in range(50):
    loss = feature_matching_loss(theta)

    grad = np.zeros_like(theta)
    eps = 1e-4
    for j in range(len(theta)):
        theta_eps = theta.copy()
        theta_eps[j] += eps
        grad[j] = (feature_matching_loss(theta_eps) - loss) / eps

    theta -= lr * grad

    if i % 10 == 0:
        print(f"Iter {i:02d} | loss {loss:.4f} | theta {theta}")