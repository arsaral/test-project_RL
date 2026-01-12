# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 15:23:25 2025

@author: USER
"""

import random

# --- Toy environment ---
# States: 0,1,2,3,4 ; terminal = 4
def step(state):
    """Move right with probability 0.8, left with 0.2."""
    if state == 4:
        return 4, 0, True
    action = random.random()
    if action < 0.8:
        next_state = min(state + 1, 4)
    else:
        next_state = max(state - 1, 0)

    reward = 1 if next_state == 4 else 0
    done = next_state == 4
    return next_state, reward, done


# --- Monte Carlo value estimation (no bootstrap) ---
def monte_carlo_V(num_episodes=200):
    V = {s: 0.0 for s in range(5)}
    returns = {s: [] for s in range(5)}

    for _ in range(num_episodes):
        state = 0
        episode = []

        # Generate full episode
        while True:
            next_s, r, done = step(state)
            episode.append((state, r))
            if done:
                break
            state = next_s

        # Compute FULL return (no bootstrap)
        G = 0
        for t in reversed(range(len(episode))):
            s, r = episode[t]
            G = r + G  # full return
            returns[s].append(G)
            V[s] = sum(returns[s]) / len(returns[s])

    return V


# --- TD(0) value estimation (bootstrap) ---
def td0_V(num_episodes=200, alpha=0.1, gamma=1.0):
    V = {s: 0.0 for s in range(5)}

    for _ in range(num_episodes):
        state = 0

        while True:
            next_s, r, done = step(state)

            # TD(0) update: V(s) <- V(s) + alpha * (r + γ V(s') - V(s))
            target = r + gamma * V[next_s] * (0 if done else 1)
            V[state] += alpha * (target - V[state])

            if done:
                break
            state = next_s

    return V


# --- Run both methods ---
print("Monte Carlo estimate:")
print(monte_carlo_V())

print("\nTD(0) estimate:")
print(td0_V())