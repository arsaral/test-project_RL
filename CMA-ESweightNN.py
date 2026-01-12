# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 14:58:50 2025
Optimize a tiny neural network to fit  y=2x+1
@author: ARS
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cma
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define the target data
# -------------------------------
x_data = np.linspace(-1, 1, 20)
y_data = 2 * x_data + 1  # simple linear function

x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)

# -------------------------------
# 2. Define a tiny neural network
# -------------------------------
class TinyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)  # 1 input, 1 output

    def forward(self, x):
        return self.fc(x)

# -------------------------------
# 3. Flatten parameters for CMA-ES
# -------------------------------
def get_params_vector(model):
    return np.concatenate([p.detach().numpy().flatten() for p in model.parameters()])

def set_params_vector(model, vector):
    pointer = 0
    for p in model.parameters():
        shape = p.shape
        size = p.numel()
        new_values = vector[pointer:pointer+size].reshape(shape)
        p.data = torch.tensor(new_values, dtype=p.dtype)
        pointer += size

# -------------------------------
# 4. Define the fitness function
# -------------------------------
def fitness_function(params_vector):
    model = TinyNN()
    set_params_vector(model, params_vector)
    with torch.no_grad():
        y_pred = model(x_tensor)
        loss = ((y_pred - y_tensor)**2).mean().item()  # MSE
    return loss

# -------------------------------
# 5. Run CMA-ES
# -------------------------------
initial_model = TinyNN()
initial_params = get_params_vector(initial_model)

es = cma.CMAEvolutionStrategy(initial_params, 0.5)

while not es.stop():
    solutions = es.ask()
    fitnesses = [fitness_function(s) for s in solutions]
    es.tell(solutions, fitnesses)

best_params = es.result.xbest
print("Best parameters found:", best_params)

# -------------------------------
# 6. Test the best model
# -------------------------------
best_model = TinyNN()
set_params_vector(best_model, best_params)
with torch.no_grad():
    y_pred = best_model(x_tensor)

# -------------------------------
# 7. Plot the results
# -------------------------------
plt.figure(figsize=(6,4))
plt.scatter(x_data, y_data, color='blue', label='Target')
plt.plot(x_data, y_pred.numpy(), color='red', label='CMA-ES NN')
plt.title('CMA-ES Optimized Neural Network')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()