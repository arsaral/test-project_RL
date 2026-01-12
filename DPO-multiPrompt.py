# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 21:32:57 2025

@author: USER
"""

import torch
import torch.nn.functional as F

# ----------------------------
# Hyperparameters
# ----------------------------
beta = 1.0
lr = 0.1
steps = 20

# ----------------------------
# "Dataset": human preferences
# ----------------------------
# Each entry: (chosen_index, rejected_index)
# Indices are LOCAL to each prompt
preferences = [
    (0, 1),  # Prompt 0: answer 0 preferred
    (1, 0),  # Prompt 1: answer 1 preferred
    (0, 1),  # Prompt 2: answer 0 preferred
]

num_prompts = len(preferences)
num_answers = 2

# ----------------------------
# Policy parameters
# ----------------------------
# One logit vector per prompt (shared training, independent contexts)
policy_logits = torch.zeros(num_prompts, num_answers, requires_grad=True)

# ----------------------------
# Reference model (frozen)
# ----------------------------
ref_logits = torch.zeros(num_prompts, num_answers)

optimizer = torch.optim.SGD([policy_logits], lr=lr)

# ----------------------------
# Training loop
# ----------------------------
for step in range(steps):
    optimizer.zero_grad()
    total_loss = 0.0

    for i, (chosen, rejected) in enumerate(preferences):
        logp_policy = F.log_softmax(policy_logits[i], dim=0)
        logp_ref = F.log_softmax(ref_logits[i], dim=0)

        delta = (
            (logp_policy[chosen] - logp_policy[rejected])
            -
            (logp_ref[chosen] - logp_ref[rejected])
        )

        loss = -torch.log(torch.sigmoid(beta * delta))
        total_loss += loss

    total_loss.backward()
    optimizer.step()

    # Monitoring
    probs = torch.softmax(policy_logits.detach(), dim=1)
    print(f"\nStep {step:02d}")
    for i in range(num_prompts):
        print(
            f" Prompt {i}: "
            f"P(answer0)={probs[i,0]:.3f}, "
            f"P(answer1)={probs[i,1]:.3f}"
        )
"""
%runfile 
Reloaded modules: torch.ops, torch.classes

Step 00
 Prompt 0: P(answer0)=0.525, P(answer1)=0.475
 Prompt 1: P(answer0)=0.475, P(answer1)=0.525
 Prompt 2: P(answer0)=0.525, P(answer1)=0.475

Step 01
 Prompt 0: P(answer0)=0.549, P(answer1)=0.451
 Prompt 1: P(answer0)=0.451, P(answer1)=0.549
 Prompt 2: P(answer0)=0.549, P(answer1)=0.451

Step 02
 Prompt 0: P(answer0)=0.571, P(answer1)=0.429
 Prompt 1: P(answer0)=0.429, P(answer1)=0.571
 Prompt 2: P(answer0)=0.571, P(answer1)=0.429

Step 03
 Prompt 0: P(answer0)=0.592, P(answer1)=0.408
 Prompt 1: P(answer0)=0.408, P(answer1)=0.592
 Prompt 2: P(answer0)=0.592, P(answer1)=0.408

Step 04
 Prompt 0: P(answer0)=0.611, P(answer1)=0.389
 Prompt 1: P(answer0)=0.389, P(answer1)=0.611
 Prompt 2: P(answer0)=0.611, P(answer1)=0.389

Step 05
 Prompt 0: P(answer0)=0.630, P(answer1)=0.370
 Prompt 1: P(answer0)=0.370, P(answer1)=0.630
 Prompt 2: P(answer0)=0.630, P(answer1)=0.370

Step 06
 Prompt 0: P(answer0)=0.647, P(answer1)=0.353
 Prompt 1: P(answer0)=0.353, P(answer1)=0.647
 Prompt 2: P(answer0)=0.647, P(answer1)=0.353

Step 07
 Prompt 0: P(answer0)=0.663, P(answer1)=0.337
 Prompt 1: P(answer0)=0.337, P(answer1)=0.663
 Prompt 2: P(answer0)=0.663, P(answer1)=0.337

Step 08
 Prompt 0: P(answer0)=0.678, P(answer1)=0.322
 Prompt 1: P(answer0)=0.322, P(answer1)=0.678
 Prompt 2: P(answer0)=0.678, P(answer1)=0.322

Step 09
 Prompt 0: P(answer0)=0.692, P(answer1)=0.308
 Prompt 1: P(answer0)=0.308, P(answer1)=0.692
 Prompt 2: P(answer0)=0.692, P(answer1)=0.308

Step 10
 Prompt 0: P(answer0)=0.705, P(answer1)=0.295
 Prompt 1: P(answer0)=0.295, P(answer1)=0.705
 Prompt 2: P(answer0)=0.705, P(answer1)=0.295

Step 11
 Prompt 0: P(answer0)=0.717, P(answer1)=0.283
 Prompt 1: P(answer0)=0.283, P(answer1)=0.717
 Prompt 2: P(answer0)=0.717, P(answer1)=0.283

Step 12
 Prompt 0: P(answer0)=0.728, P(answer1)=0.272
 Prompt 1: P(answer0)=0.272, P(answer1)=0.728
 Prompt 2: P(answer0)=0.728, P(answer1)=0.272

Step 13
 Prompt 0: P(answer0)=0.739, P(answer1)=0.261
 Prompt 1: P(answer0)=0.261, P(answer1)=0.739
 Prompt 2: P(answer0)=0.739, P(answer1)=0.261

Step 14
 Prompt 0: P(answer0)=0.749, P(answer1)=0.251
 Prompt 1: P(answer0)=0.251, P(answer1)=0.749
 Prompt 2: P(answer0)=0.749, P(answer1)=0.251

Step 15
 Prompt 0: P(answer0)=0.758, P(answer1)=0.242
 Prompt 1: P(answer0)=0.242, P(answer1)=0.758
 Prompt 2: P(answer0)=0.758, P(answer1)=0.242

Step 16
 Prompt 0: P(answer0)=0.767, P(answer1)=0.233
 Prompt 1: P(answer0)=0.233, P(answer1)=0.767
 Prompt 2: P(answer0)=0.767, P(answer1)=0.233

Step 17
 Prompt 0: P(answer0)=0.775, P(answer1)=0.225
 Prompt 1: P(answer0)=0.225, P(answer1)=0.775
 Prompt 2: P(answer0)=0.775, P(answer1)=0.225

Step 18
 Prompt 0: P(answer0)=0.783, P(answer1)=0.217
 Prompt 1: P(answer0)=0.217, P(answer1)=0.783
 Prompt 2: P(answer0)=0.783, P(answer1)=0.217

Step 19
 Prompt 0: P(answer0)=0.790, P(answer1)=0.210
 Prompt 1: P(answer0)=0.210, P(answer1)=0.790
 Prompt 2: P(answer0)=0.790, P(answer1)=0.210

"""