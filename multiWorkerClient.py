import socket
import struct
import pickle
import numpy as np
from tqdm import trange


def clearLines(numLines):
    for _ in range(numLines):
        print("\033[F\033[K", end="")


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


def receive_nparrays(sock):
    # Receive the header
    header = sock.recv(8)
    if not header:
        raise ConnectionResetError()
    num_items = struct.unpack("Q", header)[0]

    data = []

    for i in range(num_items):
        # Receive the header
        header = sock.recv(8)
        if not header:
            raise ConnectionResetError()
        item_len = struct.unpack("Q", header)[0]

        # Receive the message
        chunks = []
        while item_len > 0:
            chunkLen = min(CHUNK_SIZE, item_len)
            chunk = sock.recv(chunkLen)
            if not chunk:
                raise ConnectionResetError()
            chunks.append(chunk)
            item_len -= chunkLen

        item = np.frombuffer(b"".join(chunks), dtype=np.float32)
        data.append(item)

    return data


def receive_data(sock):
    # Receive the header
    header = sock.recv(8)
    if not header:
        raise ConnectionResetError()
    message_length = struct.unpack("Q", header)[0]

    # Receive the message
    chunks = []
    while message_length > 0:
        chunkLen = min(CHUNK_SIZE, message_length)
        chunk = sock.recv(chunkLen)
        if not chunk:
            raise ConnectionResetError()
        chunks.append(chunk)
        message_length -= chunkLen

    data = pickle.loads(b"".join(chunks))

    return data


def send_data(sock, data):
    # Encode data
    data_bytes = pickle.dumps(data)
    # Create the header with message length (8 bytes)
    data_len = len(data_bytes)
    header = struct.pack("Q", data_len)
    # Send header
    sock.sendall(header)

    # Send chunks
    chunks = []
    while len(data_bytes) > 0:
        chunkLen = min(CHUNK_SIZE, len(data_bytes))
        chunks.append(data_bytes[:chunkLen])
        data_bytes = data_bytes[chunkLen:]
    for chunk in chunks:
        sock.sendall(chunk)


def send_nparrays(sock, data: np.ndarray):
    # Encode data
    data_bytes = [item.tobytes() for item in data]
    # Create the header with number of arrays (8 bytes)
    data_len = len(data_bytes)
    header = struct.pack("Q", data_len)
    # Send header
    sock.sendall(header)

    for item_bytes in data_bytes:
        # Create the header with number of arrays (8 bytes)
        item_len = len(item_bytes)
        header = struct.pack("Q", item_len)
        # Send header
        sock.sendall(header)

        # Send chunks
        chunks = []
        while len(item_bytes) > 0:
            chunkLen = min(CHUNK_SIZE, len(item_bytes))
            chunks.append(item_bytes[:chunkLen])
            item_bytes = item_bytes[chunkLen:]
        for chunk in chunks:
            sock.sendall(chunk)


print("Connecting to server")
CHUNK_SIZE = 64
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.connect(("130.215.211.30", 8080))
server_socket.settimeout(10)

try:
    clearLines(1)
    print("Waiting for initial data")
    # Receive initial data
    weights = receive_nparrays(server_socket)
    seeds, nTrials, alpha, sigma, vocabSize, firstClient = receive_data(server_socket)

    if not firstClient:
        # Receive normalized results
        A = receive_nparrays(server_socket)[0]

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

        clearLines(1)
        for trial in trange(nTrials, desc="Doing trials"):
            w_try = [item + sigma * np.random.randn(*item.shape) for item in weights]
            R[trial] = f(tokens, w_try, vocabSize)

        # Send results
        clearLines(1)
        print("Sending results")
        if send_weights:
            send_nparrays(server_socket, [R])
            send_nparrays(server_socket, weights)
        else:
            send_nparrays(server_socket, [R])

        # Receive normalized results
        clearLines(1)
        print("Waiting for normalized results")
        A = receive_nparrays(server_socket)[0]

        # Update weights
        clearLines(1)
        print("Updating weights")
        weights = updateW(weights, alpha, sigma, nTrials, seeds, A)
        clearLines(1)

finally:
    server_socket.close()
