# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 12:33:15 2025

@author: ARS
"""

import random

# --- Environment ---
# States: 0,1,2,3
# Goal: 3 → reward +1
# Move right deterministically

def step(state, action):
    """Deterministic environment: only action=1 is allowed (go right)."""
    next_state = min(state + 1, 3)
    reward = 1 if next_state == 3 else 0
    done = (next_state == 3)
    return next_state, reward, done


# --- Policy: always go right ---
policy = {0: 1, 1: 1, 2: 1}  # state→action


# --- Monte Carlo: First-Visit Value Estimates ---
def generate_episode():
    """Generate one episode following the policy."""
    episode = []     # list of (state, reward)
    state = 0

    while True:
        action = policy[state]
        next_state, reward, done = step(state, action)
        episode.append((state, reward))
        state = next_state
        if done:
            break

    return episode


# First-visit MC evaluation
def mc_first_visit(num_episodes=50, gamma=1.0):
    returns = {0: [], 1: [], 2: [], 3: []}   # store returns for each state
    V = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}      # value estimates

    for _ in range(num_episodes):
        episode = generate_episode()

        # Extract states in order of first appearance
        visited = set()
        first_visits = []

        for i, (state, _) in enumerate(episode):
            if state not in visited:
                visited.add(state)
                first_visits.append((state, i))

        # For each first-visit state, calculate return G
        for state, first_idx in first_visits:
            G = 0
            discount = 1
            for _, reward in episode[first_idx:]:
                G += discount * reward
                discount *= gamma

            returns[state].append(G)

            # Update value as mean of returns
            V[state] = sum(returns[state]) / len(returns[state])

    return V


# --- Run ---
V = mc_first_visit(num_episodes=50)
print("Estimated state values:")
for s in V:
    print(f"V({s}) = {V[s]:.3f}")