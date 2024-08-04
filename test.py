from scipy.optimize import minimize
import numpy as np

x = np.linspace(-1, 1, 30)
y = np.exp(x)


def objective_function(params):
    a1, a2, a3, a4, a5 = params
    return np.sum((y - (a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + a5)) ** 2)


initial_guess = [1, 0.5, 0.17, 0.04, 1]
result = minimize(objective_function, initial_guess)

print("Best parameters:", result.x)
print("Minimum value:", result.fun)
