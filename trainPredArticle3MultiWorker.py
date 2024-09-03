import numpy as np
from tokenizeWiki import loadTokens, decode
from tqdm import trange
from multiprocessing import Process, Queue, Pipe
from time import sleep, perf_counter
import pickle

from line_profiler import profile

profile.disable()

vocabSize = 95
hiddenSize = 16

model_initState = np.random.random((hiddenSize))
model_ih = np.random.random((vocabSize, hiddenSize))
model_hh = np.random.random((hiddenSize, hiddenSize))
model_out = np.random.random((hiddenSize, vocabSize))

tokenLoader = loadTokens("tokenData")
fileNum, title, tokens = next(tokenLoader)
tokens = tokens[:19]


@profile
def tanh(x):
    ex = np.exp(x)
    e_x = np.exp(-x)
    return (ex - e_x) / (ex + e_x)


@profile
def softmax(x):
    x = np.exp(x)
    x *= 1 / np.sum(x)
    return x


@profile
def f(tokens, weights):
    state = weights[0]
    losses = np.zeros(vocabSize)
    for token in tokens:
        out = state @ weights[3]
        out = softmax(out)
        out[token] = 1 - out[token]
        losses += out**2
        state = tanh(state @ weights[2] + weights[1][token])
    return -np.sum(losses) / (len(tokens) * vocabSize)


@profile
def generate(weights, nTokens):
    state = weights[0]
    generated_tokens = []
    for i in range(nTokens):
        out = state @ weights[3]
        probs = softmax(out)
        token = np.random.choice(np.arange(vocabSize), p=probs.squeeze())
        generated_tokens.append(token)
        state = tanh(state @ weights[2] + weights[1][token])
    return generated_tokens


@profile
def updateW(w, alpha, sigma, nTrials, nPop, seeds, A):
    wUpdate = [np.zeros_like(item) for item in w]

    for workerIdx, seed in enumerate(seeds):
        np.random.seed(seed)
        for trial in range(nTrials):
            for i in range(len(wUpdate)):
                wUpdate[i] += (
                    np.random.randn(*wUpdate[i].shape) * A[workerIdx * nTrials + trial]
                )

    for i in range(len(wUpdate)):
        w[i] += (alpha / (nPop * sigma)) * wUpdate[i]

    return w


@profile
def worker_process(worker_id, nTrials, nPop, w, tokens, alpha, pipe):
    w = pipe.recv()  # Receive the initial weights

    while True:
        done, seeds, sigma = pipe.recv()
        if done:  # If done is true, terminate the worker
            pipe.send(w)  # Send the updated weights back to the main process
            break

        # Set seed
        np.random.seed(seeds[worker_id])

        R = np.zeros(nTrials)

        for trial in range(nTrials):
            w_try = [item + sigma * np.random.randn(*item.shape) for item in w]
            R[trial] = f(tokens, w_try)

        pipe.send(R)  # Send the rewards and seed back to the main process

        A = pipe.recv()  # Receive A from the main process
        w = updateW(w, alpha, sigma, nTrials, nPop, seeds, A)


def main():
    nIters = 10000  # number of optimization iterations
    npop = 8000  # population size
    nWorkers = 8  # number of workers
    sigmaMin = 0.01
    sigmaMax = 0.01
    alpha = 0.001  # learning rate
    w = [model_initState, model_ih, model_hh, model_out]

    # Split npop among workers
    trials_per_worker = npop // nWorkers
    assert (
        npop == trials_per_worker * nWorkers
    ), f"npop must be divisible by nWorkers. {npop} is not divisible by {nWorkers}"

    # Create pipes for communication
    pipes = [Pipe() for _ in range(nWorkers)]
    workers = [
        Process(
            target=worker_process,
            args=(i, trials_per_worker, npop, w, tokens, alpha, pipe[1]),
        )
        for i, pipe in enumerate(pipes)
    ]

    # Start worker threads
    for worker in workers:
        worker.start()

    # Send the initial weights to each worker
    for parent_pipe in pipes:
        parent_pipe[0].send(w)

    # Initialize counters
    lastTime = perf_counter()

    for i in range(nIters):
        seeds = np.random.randint(0, 1_000_000_000, nWorkers)

        # Send (not done, seed) to workers
        sigma = (0.5 * np.sin(0.1 * i) + 0.5) * (sigmaMax - sigmaMin) + sigmaMin
        for idx in range(nWorkers):
            pipes[idx][0].send((False, seeds, sigma))

        # Receive the rewards from each worker
        all_R = []
        for parent_pipe in pipes:
            R = parent_pipe[0].recv()
            all_R.extend(R)

        # Normalize rewards
        R = np.array(all_R)
        mean = np.mean(R)
        A = (R - mean) / np.std(R)

        for parent_pipe in pipes:
            parent_pipe[0].send(A)  # Send A to each worker to update their weights

        # Logging
        time = perf_counter()
        print(f"Iter: {i}, Loss: {mean:.3e}, Sigma: {sigma:.3e}, Time: {time - lastTime:.4f}")
        lastTime = time

    # Terminate workers
    for parent_pipe in pipes:
        parent_pipe[0].send((True, None))
        w = parent_pipe[
            0
        ].recv()  # Receive the updated weights from each worker (this will overwrite, but they should all be the same so it shouldn't matter)

    for worker in workers:
        worker.join()
    
    # Saving the list to a file using pickle
    with open('models/tokenPredArticleEvolve/0.pkl', 'wb') as file:
        pickle.dump(w, file)
    
    print(decode(generate(w, 100)))

    sleep(2)  # So that the main thread profile wont overwrite the worker thread profile


if __name__ == "__main__":
    main()
