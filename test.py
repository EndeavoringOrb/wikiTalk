from scipy.optimize import minimize
import numpy as np

x = np.linspace(0, 0.5, 100000)
y = np.exp(x)

def objective_function(params):
    a1, a2, a3, a4, a5 = params
    pred = a1 + a2 * x + a3 * x ** 2 + a4 * x ** 3 + a5 * x ** 4
    loss = np.sum((y - pred) ** 2)
    return loss

initial_guess = [1, 1, 1, 1, 1]
result = minimize(objective_function, initial_guess)

print("Best parameters:", result.x)
print("Minimum value:", result.fun)
