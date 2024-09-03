# https://openai.com/index/evolution-strategies/

import numpy as np
from tokenizeWiki import loadTokens, decode
from tqdm import tqdm, trange

from line_profiler import profile
profile.disable()

vocabSize = 95
hiddenSize = 2

model_initState = np.random.random((1, hiddenSize))
model_ih = np.random.random((vocabSize, hiddenSize))
model_hh = np.random.random((hiddenSize, hiddenSize))
model_out = np.random.random((hiddenSize, vocabSize))

tokenLoader = loadTokens("tokenData")
fileNum, title, tokens = next(tokenLoader)


@profile
def randWeights(w, N: list[np.ndarray], idx, sigma):
    return [w[i] + sigma * N[i][idx] for i in range(len(w))]


def relu(x):
    x[x < 0] = 0
    return x


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
    loss = 0
    for token in tokens:
        out = state @ weights[3]

        out = softmax(out)
        out[:, token] = 1 - out[:, token]
        loss -= np.sum(out**2)

        state = tanh(state @ weights[2] + weights[1][token])

    return loss / (len(tokens) * vocabSize)


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
def main():
    nIters = 1_000_000  # number of optimization iterations
    npop = 500  # population size
    sigma = 0.01  # noise standard deviation
    alpha = 0.001  # learning rate
    w = [model_initState, model_ih, model_hh, model_out]
    for i in range(nIters):
        N = [np.random.randn(npop, *item.shape) for item in w]
        R = np.zeros(npop)
        for j in trange(npop):
            w_try = randWeights(w, N, j, sigma)
            R[j] = f(tokens, w_try)

        mean = np.mean(R)
        A = (R - mean) * (1 / np.std(R))

        for j in range(len(w)):
            w[j] = w[j] + alpha / (npop * sigma) * np.dot(N[j].transpose(1, 2, 0), A)

        print(i, mean, decode(generate(w, 11)))


if __name__ == "__main__":
    main()
