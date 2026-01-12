# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 18:10:34 2025

@author: ARS
"""

# Step-by-step demonstration of the three items the user requested:
# 1) Graph of J(θ)
# 2) Two-step full optimization (analytic gradient updates) shown step-by-step
# 3) Two-step sample-based REINFORCE updates step-by-step
#
# This code prints all intermediate values and shows a single plot of J(θ)
# with markers for the various theta positions.
import math
import numpy as np
import matplotlib.pyplot as plt

def sigma(x):
    return 1.0 / (1.0 + math.exp(-x))

def J(theta):
    return 1.0 + 2.0 * sigma(theta)

def analytic_grad_J(theta):
    s = sigma(theta)
    return 2.0 * s * (1.0 - s)  # d/dθ (1 + 2σ(θ)) = 2 σ'(θ) = 2 σ(1-σ)

def sample_reinforce_gradient(theta, action_reward):
    # action_reward: tuple (a, R)
    a, R = action_reward
    s = sigma(theta)
    # ∇θ log π(a) for Bernoulli with π(1)=σ(θ)
    if a == 1:
        grad_log = 1.0 - s
    else:
        grad_log = -s
    return grad_log * R

# --- Settings ---
theta0 = 0.0
alpha = 0.1  # learning rate for both methods (kept small for clarity)
np.random.seed(0)

# --- 1) Graph of J(θ) ---
thetas = np.linspace(-6, 6, 601)
Js = [J(t) for t in thetas]

# --- 2) Two-step analytic gradient ascent (exact gradient of J) ---
print("=== Analytic gradient (exact) two-step update ===")
theta_a = theta0
print(f"Initial θ = {theta_a:.6f}, σ(θ) = {sigma(theta_a):.6f}, J(θ) = {J(theta_a):.6f}")
for step in range(1, 3):
    g = analytic_grad_J(theta_a)
    theta_a = theta_a + alpha * g
    print(f"Step {step}: grad = {g:.6f}, updated θ = {theta_a:.6f}, σ(θ) = {sigma(theta_a):.6f}, J(θ) = {J(theta_a):.6f}")

# --- 3) Two-step sample-based REINFORCE updates ---
# Environment: one state, two actions.
# rewards: a=0 -> r=1, a=1 -> r=3
print("\n=== Sample-based REINFORCE two-step update ===")
theta_s = theta0
print(f"Initial θ = {theta_s:.6f}, σ(θ) = {sigma(theta_s):.6f}, J(θ) = {J(theta_s):.6f}")
for step in range(1, 3):
    # sample an action according to π_θ
    p = sigma(theta_s)
    a = 1 if np.random.rand() < p else 0
    R = 3 if a == 1 else 1
    # gradient estimate
    g_est = sample_reinforce_gradient(theta_s, (a, R))
    theta_s = theta_s + alpha * g_est
    print(f"Step {step}: sampled a={a}, R={R}, grad_est = {g_est:.6f}, updated θ = {theta_s:.6f}, σ(θ) = {sigma(theta_s):.6f}, J(θ) = {J(theta_s):.6f}")

# --- Plot J(θ) and mark points ---
plt.figure(figsize=(8,5))
plt.plot(thetas, Js)
# markers for initial and updated thetas
markers_thetas = [theta0, theta_a, # analytic final
                  theta0, theta_s]  # sample final (initial repeated for clarity)
markers_labels = ["initial (analytic)", "after 2 analytic steps", "initial (sample)", "after 2 sampled steps"]
y_vals = [J(t) for t in markers_thetas]
plt.scatter(markers_thetas, y_vals)
for x,y,lbl in zip(markers_thetas, y_vals, markers_labels):
    plt.annotate(lbl + f"\nθ={x:.3f}\nJ={y:.3f}", (x,y), textcoords="offset points", xytext=(5,5), fontsize=8)

plt.title("J(θ) = 1 + 2σ(θ) and update markers")
plt.xlabel("θ")
plt.ylabel("J(θ)")
plt.grid(True)
plt.tight_layout()
plt.show()