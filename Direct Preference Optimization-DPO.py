# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 21:01:02 2025

@author: USER
"""

import torch
import torch.nn.functional as F

# ----------------------------
# Hyperparameters
# ----------------------------
beta = 1.0
learning_rate = 0.1
num_steps = 10

# ----------------------------
# Trainable policy logits
# ----------------------------
# Two answers: [chosen, rejected]
policy_logits = torch.tensor([0.0, 0.0], requires_grad=True)

# ----------------------------
# Fixed reference policy logits
# ----------------------------
ref_logits = torch.tensor([0.0, 0.0])  # frozen

optimizer = torch.optim.SGD([policy_logits], lr=learning_rate)

# ----------------------------
# Training loop
# ----------------------------
for step in range(num_steps):
    optimizer.zero_grad()

    # Convert logits → log probabilities
    logp_policy = F.log_softmax(policy_logits, dim=0)
    logp_ref = F.log_softmax(ref_logits, dim=0)

    # Indices
    chosen = 0
    rejected = 1

    # DPO core quantity
    delta = (
        (logp_policy[chosen] - logp_policy[rejected])
        -
        (logp_ref[chosen] - logp_ref[rejected])
    )

    # DPO loss
    loss = -torch.log(torch.sigmoid(beta * delta))

    # Backprop
    loss.backward()
    optimizer.step()

    # Print diagnostics
    probs = torch.softmax(policy_logits.detach(), dim=0)
    print(
        f"Step {step:02d} | "
        f"Loss {loss.item():.4f} | "
        f"P(chosen) {probs[0]:.3f} | "
        f"P(rejected) {probs[1]:.3f}"
    )
    
    """
    %runfile 
    Step 00 | Loss 0.6931 | P(chosen) 0.525 | P(rejected) 0.475
    Step 01 | Loss 0.6444 | P(chosen) 0.549 | P(rejected) 0.451
    Step 02 | Loss 0.6004 | P(chosen) 0.571 | P(rejected) 0.429
    Step 03 | Loss 0.5606 | P(chosen) 0.592 | P(rejected) 0.408
    Step 04 | Loss 0.5247 | P(chosen) 0.611 | P(rejected) 0.389
    Step 05 | Loss 0.4922 | P(chosen) 0.630 | P(rejected) 0.370
    Step 06 | Loss 0.4627 | P(chosen) 0.647 | P(rejected) 0.353
    Step 07 | Loss 0.4359 | P(chosen) 0.663 | P(rejected) 0.337
    Step 08 | Loss 0.4115 | P(chosen) 0.678 | P(rejected) 0.322
    Step 09 | Loss 0.3892 | P(chosen) 0.692 | P(rejected) 0.308
    """