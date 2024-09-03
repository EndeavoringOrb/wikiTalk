import socket
import struct
import pickle
import numpy as np
from tqdm import trange


def tanh(x):
    ex = np.exp(x)
    e_x = np.exp(-x)
    return (ex - e_x) / (ex + e_x)


def softmax(x):
    x = np.exp(x)
    x *= 1 / np.sum(x)
    return x


def f(tokens, weights, vocabSize):
    state = weights[0]
    losses = np.zeros(vocabSize)
    for token in tokens:
        out = state @ weights[3]
        out = softmax(out)
        out[token] = 1 - out[token]
        losses += out**2
        state = tanh(state @ weights[2] + weights[1][token])
    return -np.sum(losses) / (len(tokens) * vocabSize)


def generate(weights, nTokens, vocabSize):
    state = weights[0]
    generated_tokens = []
    for i in range(nTokens):
        out = state @ weights[3]
        probs = softmax(out)
        token = np.random.choice(np.arange(vocabSize), p=probs.squeeze())
        generated_tokens.append(token)
        state = tanh(state @ weights[2] + weights[1][token])
    return generated_tokens


def updateW(w, alpha, sigma, nTrials, seeds, A):
    wUpdate = [np.zeros_like(item) for item in w]

    for workerIdx, seed in enumerate(seeds):
        np.random.seed(seed)
        for trial in range(nTrials):
            for i in range(len(wUpdate)):
                wUpdate[i] += (
                    np.random.randn(*wUpdate[i].shape) * A[workerIdx * nTrials + trial]
                )

    nPop = len(A)

    for i in range(len(wUpdate)):
        w[i] += (alpha / (nPop * sigma)) * wUpdate[i]

    return w


def send_data(sock, data):
    # Encode data
    data_bytes = pickle.dumps(data)
    # Create the header with message length (4 bytes)
    header = struct.pack("!I", len(data_bytes))
    # Send header and message
    sock.sendall(header + data_bytes)


def receive_data(sock):
    # Receive the header
    header = sock.recv(4)
    if not header:
        raise ConnectionResetError()
    message_length = struct.unpack("!I", header)[0]

    # Receive the message
    message = sock.recv(message_length)
    if not message:
        raise ConnectionResetError()
    data = pickle.loads(message)

    return data


print("Connecting to server")
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.connect(("130.215.211.30", 8080))

try:
    print("Waiting for initial data")
    # Receive initial data
    weights, seeds, nTrials, alpha, sigma, vocabSize, firstClient = receive_data(server_socket)

    if not firstClient:
        # Receive normalized results
        A = receive_data(server_socket)

        # Update weights
        w = updateW(weights, alpha, sigma, nTrials, seeds, A)

    while True:
        print("Waiting for data")
        # Get data
        done, send_weights, seeds, nTrials, client_id, tokens, alpha, sigma = (
            receive_data(server_socket)
        )

        # Do trials
        ## Initialize
        np.random.seed(seeds[client_id])  # Set seed
        R = np.zeros(nTrials)  # Init reward array

        for trial in trange(nTrials, desc="Doing trials"):
            w_try = [item + sigma * np.random.randn(*item.shape) for item in weights]
            R[trial] = f(tokens, w_try, vocabSize)

        # Send results
        print("Sending results")
        if send_weights:
            send_data(server_socket, [R, weights])
        else:
            send_data(server_socket, R)

        # Receive normalized results
        print("Waiting for normalized results")
        A = receive_data(server_socket)

        # Update weights
        print("Updating weights")
        weights = updateW(weights, alpha, sigma, nTrials, seeds, A)
        print()

finally:
    server_socket.close()
