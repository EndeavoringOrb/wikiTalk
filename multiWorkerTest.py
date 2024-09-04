import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tokenizeWiki import vocab, decode


def loadWeights(file):
    # Load from file
    with open(file, "rb") as f:
        weights, weightShapes = pickle.load(f)

    weights = [weights[i].reshape(weightShapes[i]) for i in range(len(weights))]

    return weights


def tanh(x):
    ex = np.exp(x)
    e_x = np.exp(-x)
    return (ex - e_x) / (ex + e_x)


def softmax(x):
    x = np.exp(x)
    x *= 1 / np.sum(x)
    return x


def generate(weights, nTokens):
    state = weights[0]
    for i in range(nTokens):
        out = state @ weights[3]
        probs = softmax(out)
        token = np.random.choice(np.arange(len(vocab.vocab)), p=probs.squeeze())
        yield token
        state = tanh(state @ weights[2] + weights[1][token])


folder = "multiWorkerModels/0"
paths = os.listdir(f"{folder}/checkpoints")
if len(paths) == 0:
    print(f"No checkpoints found in {folder}/checkpoints.")
    exit(0)
modelPath = os.listdir(f"{folder}/checkpoints")[-1]
print(f"Loading model from {folder}/checkpoints/{modelPath}")
weights = loadWeights(f"{folder}/checkpoints/{modelPath}")
print()

with open(f"{folder}/loss.txt", "r", encoding="utf-8") as f:
    text = f.read()
loss_vals = [float(item.split()[0]) for item in text.splitlines()]
sigma_vals = [float(item.split()[1]) for item in text.splitlines()]

# Creating subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plotting loss_vals
ax1.plot(loss_vals, marker="o", linestyle="-", color="b")
ax1.set_title("Loss Values")
ax1.set_xlabel("Step #")
ax1.set_ylabel("Loss")
ax1.grid(True)

# Plotting sigma_vals
ax2.plot(sigma_vals, marker="o", linestyle="-", color="r")
ax2.set_title("Sigma Values")
ax2.set_xlabel("Step #")
ax2.set_ylabel("Sigma")
ax2.grid(True)

# Adjusting layout
plt.tight_layout()
plt.show()


while True:
    nTokens = int(input("Enter number of tokens to generate: "))
    for token in generate(weights, nTokens):
        print(decode([token]), end="", flush=True)
    print()
    print()
