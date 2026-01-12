# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 14:47:16 2025

@author: USER
"""

# First-visit Monte Carlo örneği (Python)
# - Basit bir "Random Walk" episodic environment kullanır (states 0..4).
# - Başlangıç state'i 2'dir. 0 ve 4 terminaldir. Sağ terminale (4) ulaşmak +1 ödül verir, sol terminal (0) 0 ödül verir.
# - Önce: Policy Evaluation (rastgele policy ile first-visit MC) -> V(s) tahmini
# - Sonra: On-policy First-visit Monte Carlo Control (epsilon-greedy) -> Q(s,a) ve öğrenilmiş politika
# Çalıştırınca hem sayısal sonuçlar (V, Q) hem de öğrenilmiş politika yazdırılır.
# Not: Kullanıcı için adım adım kod içi açıklamalar (Türkçe) eklendi.

import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# --- Basit Random Walk ortamı ---
class RandomWalkEnv:
    def __init__(self):
        self.states = list(range(5))  # 0..4
        self.start_state = 2
        self.terminal_states = {0, 4}
        self.reset()
    
    def reset(self):
        self.state = self.start_state
        return self.state
    
    def step(self, action):
        # action: 0 -> left, 1 -> right
        if action == 0:
            self.state -= 1
        else:
            self.state += 1
        
        done = self.state in self.terminal_states
        reward = 1.0 if self.state == 4 else 0.0
        return self.state, reward, done, {}

# --- Yardımcı fonksiyonlar ---

def generate_episode(env, policy):
    """Verilen policy ile bir episode üretir. Döner: (states, actions, rewards)"""
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    while not done:
        action = policy[state]
        states.append(state)
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        state = next_state
    return states, actions, rewards

def returns_from_rewards(rewards):
    """Bir ödül dizisinden G_t'leri hesapla (discount = 1, yani toplam ödül)."""
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + G
        returns.insert(0, G)
    return returns

# --- 1) First-visit MC: Policy Evaluation (rastgele policy) ---
def mc_policy_evaluation(env, policy, num_episodes=5000):
    """
    First-visit Monte Carlo policy evaluation.
    policy: dict state->action (deterministic for simplicity)
    """
    # Gözlem toplama için veri yapıları
    returns_sum = defaultdict(float)   # toplam getiriler
    returns_count = defaultdict(int)   # ziyaret sayıları
    V = defaultdict(float)             # değer tahmini (ortalama)
    
    for ep in range(num_episodes):
        states, actions, rewards = generate_episode(env, policy)
        Gs = returns_from_rewards(rewards)
        
        visited = set()
        for t, state in enumerate(states):
            if state in visited:
                continue  # first-visit
            visited.add(state)
            G = Gs[t]
            returns_sum[state] += G
            returns_count[state] += 1
            V[state] = returns_sum[state] / returns_count[state]
            
    return V, returns_count

# Rastgele (random) policy: her state için eş olasılıkla left veya right
def random_policy_factory(env):
    policy = {}
    for s in env.states:
        if s in env.terminal_states:
            policy[s] = None
        else:
            policy[s] = random.choice([0,1])  # deterministic sample for generation; we will sample actions probabilistically when generating episodes
    return policy

# Daha doğru rastgele policy: state'e göre olasılık (0.5, 0.5)
def random_policy_prob_action(state):
    return np.random.choice([0,1])


# Üretilen episode'larda rastgele policy uygulamak için küçük değişiklik:
def generate_episode_random(env):
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    while not done:
        action = random_policy_prob_action(state)
        states.append(state)
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        state = next_state
    return states, actions, rewards

def mc_policy_evaluation_random(env, num_episodes=5000):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)
    for ep in range(num_episodes):
        states, actions, rewards = generate_episode_random(env)
        Gs = returns_from_rewards(rewards)
        visited = set()
        for t, state in enumerate(states):
            if state in visited:
                continue
            visited.add(state)
            G = Gs[t]
            returns_sum[state] += G
            returns_count[state] += 1
            V[state] = returns_sum[state] / returns_count[state]
    return V, returns_count

# --- 2) First-visit MC Control (On-policy, epsilon-soft) ---
def mc_on_policy_control(env, num_episodes=20000, epsilon=0.1):
    """
    On-policy first-visit Monte Carlo control with epsilon-soft (epsilon-greedy) policies.
    Returns Q, policy.
    """
    Q = defaultdict(lambda: np.zeros(2))  # Q[s][a]
    returns_sum = defaultdict(lambda: np.zeros(2))
    returns_count = defaultdict(lambda: np.zeros(2))
    
    # Başlangıçta rastgele (epsilon-soft) policy
    def get_action_probabilities(state):
        # epsilon-soft: diğer aksiyonlara epsilon/|A|, greedy action gets 1-epsilon + epsilon/|A|
        A = 2
        probs = np.ones(A) * (epsilon / A)
        q_vals = Q[state]
        best = np.argmax(q_vals)
        probs[best] += (1.0 - epsilon)
        return probs
    
    def generate_episode_with_policy():
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        while not done:
            probs = get_action_probabilities(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            states.append(state)
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            state = next_state
        return states, actions, rewards
    
    for ep in range(num_episodes):
        states, actions, rewards = generate_episode_with_policy()
        Gs = returns_from_rewards(rewards)
        
        visited = set()
        for t, state in enumerate(states):
            action = actions[t]
            sa = (state, action)
            if sa in visited:
                continue  # first-visit sa pair
            visited.add(sa)
            G = Gs[t]
            returns_sum[state][action] += G
            returns_count[state][action] += 1
            Q[state][action] = returns_sum[state][action] / returns_count[state][action]
        # policy implicitly improves because get_action_probabilities uses updated Q
    # Final deterministic greedy policy (from Q)
    policy = {}
    for s in env.states:
        if s in env.terminal_states:
            policy[s] = None
        else:
            policy[s] = int(np.argmax(Q[s]))
    return Q, policy

# --- Çalıştırma ve çıktı ---
# --- Çalıştırma ve çıktı ---
# --- Çalıştırma ve çıktı ---
env = RandomWalkEnv()

print("---- First-visit Monte Carlo Policy Evaluation (rastgele policy) ----")
V_est, counts = mc_policy_evaluation_random(env, num_episodes=5000)
for s in sorted(V_est.keys()):
    print(f"V({s}) ≈ {V_est[s]:.4f}  (visit_count={counts[s]})")

print("\n---- First-visit Monte Carlo Control (On-policy, epsilon-greedy) ----")
Q, learned_policy = mc_on_policy_control(env, num_episodes=20000, epsilon=0.1)
print("Q(s,a):")
for s in sorted(Q.keys()):
    print(f"s={s}: a=left({Q[s][0]:.4f}), a=right({Q[s][1]:.4f})")
print("\nÖğrenilmiş politika (0=left, 1=right):")
for s in env.states:
    print(f"state {s}: policy -> {learned_policy[s]}")

# Basit görselleştirme: state'lerin V tahminleri (policy evaluation kısmı) ve Q'dan türetilen greedy V
states = [0,1,2,3,4]
V_list = [V_est.get(s, 0.0) for s in states]
# Greedy value from Q
V_greedy = [0.0 if s in env.terminal_states else max(Q[s]) for s in states]

plt.figure(figsize=(8,4))
plt.plot(states, V_list, marker='o', label='MC eval V(s) (random policy)')
plt.plot(states, V_greedy, marker='s', label='V(s) from learned Q (greedy)')
plt.xlabel('State')
plt.ylabel('Value estimate')
plt.title('First-visit Monte Carlo - V estimates')
plt.legend()
plt.grid(True)
plt.show()

# Ayrıca Q tablosunu daha okunaklı gösterelim
import pandas as pd
df = pd.DataFrame(index=states, columns=['Q_left','Q_right'])
for s in states:
    df.loc[s,'Q_left'] = Q[s][0] if s not in env.terminal_states else np.nan
    df.loc[s,'Q_right'] = Q[s][1] if s not in env.terminal_states else np.nan

print("\nTamamlandı: İlk MC örneğini çalıştırdım. Yukarıdaki tablo ve grafik sonuçları gösteriyor.")

"""
---- First-visit Monte Carlo Policy Evaluation (rastgele policy) ----
V(1) ≈ 0.2460  (visit_count=3288)
V(2) ≈ 0.5042  (visit_count=5000)
V(3) ≈ 0.7582  (visit_count=3325)

---- First-visit Monte Carlo Control (On-policy, epsilon-greedy) ----
Q(s,a):
s=1: a=left(0.0000), a=right(0.9603)
s=2: a=left(0.5690), a=right(0.9982)
s=3: a=left(0.9656), a=right(1.0000)

Öğrenilmiş politika (0=left, 1=right):
state 0: policy -> None
state 1: policy -> 1
state 2: policy -> 1
state 3: policy -> 1
state 4: policy -> None

png=First-visit MC-V-estimates
"""