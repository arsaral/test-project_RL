# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 20:37:55 2025

@author: USER
"""
# =============================================================================
# 
# An RNN maintains a hidden state:
# hidden_state = summary of the past
# 
# So the decision is based on:
# (current observation, remembered history)
# 
# Now the mapping is well-defined.
# =============================================================================

#5. Dataset Generation (Expert Demonstrations)
#We generate expert behavior that follows the rule.

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

# Generate sequences
def generate_data(num_sequences=500, seq_len=10):
    obs = torch.randint(0, 2, (num_sequences, seq_len, 1)).float()
    acts = torch.zeros_like(obs)

    for i in range(num_sequences):
        for t in range(1, seq_len):
            acts[i, t] = (obs[i, t] == obs[i, t-1]).float()

    return obs, acts

obs, actions = generate_data()
#Observations are partial by design.

#6. RNN Policy (Minimal and Honest)
class RNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.sigmoid(self.head(out))

#7. Training (Behavioral Cloning)
policy = RNNPolicy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

for epoch in range(200):
    preds = policy(obs)
    loss = loss_fn(preds, actions)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.4f}")

#8. Test on a Single Sequence
test_seq = torch.tensor([[[1.0],[1.0],[0.0],[0.0],[0.0],[1.0]]])

with torch.no_grad():
    probs = policy(test_seq)
    predicted = (probs > 0.5).int()

print("Observations:", test_seq.squeeze().tolist())
print("Predicted actions:", predicted.squeeze().tolist())

# =============================================================================
# Expected behavior:
# Observations: [1, 1, 0, 0, 0, 1]
# Predicted actions: [0, 1, 0, 1, 1, 0]
# =============================================================================
# %runfile 
# Epoch 0, Loss = 0.6972
# Epoch 50, Loss = 0.2639
# Epoch 100, Loss = 0.0095
# Epoch 150, Loss = 0.0035
# Observations: [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# Predicted actions: [0, 1, 0, 1, 1, 0]
# =============================================================================
