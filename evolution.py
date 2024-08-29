import numpy as np

solution = np.array([0.5, 0.1, -0.3])


def f(w):
    return -np.sum((w - solution) ** 2)


nIters = 300  # number of optimization iterations
npop = 50  # population size
sigma = 0.1  # noise standard deviation
alpha = 0.001  # learning rate
w = np.random.randn(3)  # initial guess
for i in range(nIters):
    N = np.random.randn(npop, 3)
    R = np.zeros(npop)
    for j in range(npop):
        w_try = w + sigma * N[j]
        R[j] = f(w_try)
    A = (R - np.mean(R)) / np.std(R)
    w = w + alpha / (npop * sigma) * np.dot(N.T, A)

# multi-worker code
nWorkers = 5
seed = 0

# send initial w to all workers

for i in range(nIters):
    # WORKER DO TRIALS
    # start workers, each has their own seed
    # each worker does N trials

    # ON MAIN THREAD
    # get rewards from each worker
    # A = (R - mean(R)) / std(R)

    # WORKER UPDATE W
    # send A, and all worker seeds to every worker
    # each worker does
        # N = []
        # for seed in workerSeeds
        # N.append(np.random.randn(nTrials, nParams))
        # w = w + alpha / (npop * sigma) * np.dot(N.T, A)
    pass
