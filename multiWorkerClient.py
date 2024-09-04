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


def updateW(w, alpha, sigma, A, workerInfo):
    wUpdate = [np.zeros_like(item) for item in w]

    AIndex = 0

    for workerIdx, seed, nTrials in workerInfo:
        np.random.seed(seed)
        for trial in range(nTrials):
            for i in range(len(wUpdate)):
                wUpdate[i] += np.random.randn(*wUpdate[i].shape) * A[AIndex + trial]
        AIndex += nTrials

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


def send_nparrays(sock, data: list[np.ndarray]):
    # Encode data
    data_bytes = [item.astype(np.float32).tobytes() for item in data]
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
CHUNK_SIZE = 256
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.connect(("130.215.211.30", 8080))
clearLines(1)
print("Connected to server")

iterNum = 0

try:
    print("Waiting for initial data")
    # Receive initial data
    weights: list[np.ndarray] = receive_nparrays(server_socket)
    alpha, sigma, vocabSize, weightShapes, firstClient = receive_data(server_socket)
    for i in range(len(weights)):
        weights[i] = weights[i].reshape(weightShapes[i]).copy()
    clearLines(1)

    if not firstClient:
        # Receive normalized results
        print("Waiting for normalized results")
        success, workerInfo = receive_data(server_socket)
        if success:
            A = receive_nparrays(server_socket)[0]

            # Update weights
            print("Updating weights")
            weights = updateW(weights, alpha, sigma, A, workerInfo)
            clearLines(1)
        clearLines(1)

    while True:
        print(f"Iter #: {iterNum}")
        print("Waiting for data")
        # Get data
        done, send_weights, seed, nTrials, client_id, tokens, alpha, sigma = (
            receive_data(server_socket)
        )

        # Do trials
        ## Initialize
        np.random.seed(seed)  # Set seed
        R = np.zeros(nTrials)  # Init reward array

        for trial in trange(nTrials, desc="Doing trials"):
            w_try = [item + sigma * np.random.randn(*item.shape) for item in weights]
            R[trial] = f(tokens, w_try, vocabSize)

        # Send results
        print("Sending results")
        if send_weights:
            send_nparrays(server_socket, [R])
            print("Sending weights")
            send_nparrays(server_socket, weights)
        else:
            send_nparrays(server_socket, [R])

        # Receive normalized results
        print("Waiting for normalized results")
        success, workerInfo = receive_data(server_socket)
        if not success:
            continue
        A = receive_nparrays(server_socket)[0]

        # Update weights
        print("Updating weights")
        weights = updateW(weights, alpha, sigma, A, workerInfo)

        iterNum += 1
        clearLines(6 + send_weights)

finally:
    server_socket.close()
