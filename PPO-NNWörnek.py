import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -----------------------------
# ENVIRONMENT
# -----------------------------

class LineWorld:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # 0 = Left, 1 = Right
        if action == 1:
            self.state = min(self.state + 1, 3)
        else:
            self.state = max(self.state - 1, 0)

        reward = 1 if self.state == 3 else 0
        done = self.state == 3
        return self.state, reward, done


# -----------------------------
# PPO NETWORK
# -----------------------------

class PPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 64)
        self.actor = nn.Linear(64, 2)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.shared(x))
        return self.actor(x), self.critic(x)


# -----------------------------
# HELPERS
# -----------------------------

def one_hot(state):
    vec = np.zeros(4)
    vec[state] = 1
    return torch.tensor(vec, dtype=torch.float32)


def compute_returns(rewards, gamma=0.9):
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


# -----------------------------
# TRAINING SETUP
# -----------------------------

env = LineWorld()
agent = PPOAgent()

optimizer = optim.Adam(agent.parameters(), lr=0.01)

gamma = 0.9
eps_clip = 0.2
ppo_epochs = 5
episodes = 300


# -----------------------------
# PPO TRAINING LOOP
# -----------------------------

for episode in range(episodes):

    states = []
    actions = []
    rewards = []
    old_log_probs = []

    state = env.reset()
    done = False

    # -------- COLLECT ONE EPISODE --------
    while not done:
        state_tensor = one_hot(state)

        logits, _ = agent(state_tensor)
        probs = torch.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done = env.step(action.item())

        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        old_log_probs.append(log_prob.detach())

        state = next_state

    # -------- RETURNS & ADVANTAGES --------

    returns = compute_returns(rewards, gamma)

    values = []
    for s in states:
        _, v = agent(s)
        values.append(v.squeeze())
    values = torch.stack(values)

    advantages = (returns - values.detach()).detach()

    old_log_probs = torch.stack(old_log_probs)

    # -------- PPO UPDATES --------

    for _ in range(ppo_epochs):

        new_log_probs = []
        new_values = []

        for s, a in zip(states, actions):
            logits, v = agent(s)
            probs = torch.softmax(logits, dim=0)
            dist = torch.distributions.Categorical(probs)

            new_log_probs.append(dist.log_prob(a))
            new_values.append(v.squeeze())

        new_log_probs = torch.stack(new_log_probs)
        new_values = torch.stack(new_values)

        ratios = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(new_values, returns)

        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # -------- PROGRESS PRINT --------
    if episode % 25 == 0:
        print(f"Episode {episode} | Total Reward = {sum(rewards)}")


# -----------------------------
# FINAL POLICY CHECK
# -----------------------------

print("\nFinal Policy Probabilities (Right Action):")
for s in range(3):
    state_tensor = one_hot(s)
    logits, _ = agent(state_tensor)
    probs = torch.softmax(logits, dim=0)
    print(f"State {s+1} → P(Right) = {probs[1].item():.3f}")