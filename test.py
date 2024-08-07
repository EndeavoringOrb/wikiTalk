from scipy.optimize import minimize
import numpy as np
import csv

# Lists to store the data
pages = []
loss = []

# Read the data from the file
with open('models/embedArticle/0/loss.txt', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        pages.append(int(row[0]))
        loss.append(float(row[2]))

x = np.asarray(pages)
y = np.asarray(loss)
#x = np.linspace(0, 0.5, 100000)
#y = np.exp(x)

def objective_function(params):
    a1, a2 = params
    pred = a1 * x + a2
    loss = np.sum((y - pred) ** 2)
    return loss

initial_guess = [1, 0]
result = minimize(objective_function, initial_guess)

print("Best parameters:", result.x)
print("Minimum value:", result.fun)
