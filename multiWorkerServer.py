import os
import socket
import select
import struct
import pickle
import numpy as np
from time import perf_counter
from datetime import datetime
from tokenizeWiki import loadTokens, vocab


def clearLines(numLines):
    for _ in range(numLines):
        print("\033[F\033[K", end="")


def receive_nparrays(sock):
    global log
    global sockets_list
    global clients
    global numClients
    try:
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
    except Exception as e:
        log.append(f"EXCEPTION in receive_nparrays: {e}")
        log.append(f"Connection closed from {clients[sock]}")
        sockets_list.remove(sock)
        del clients[sock]
        sock.close()
        numClients -= 1
        updateLog()


def receive_data(sock):
    global log
    global sockets_list
    global clients
    global numClients
    try:
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
    except Exception as e:
        log.append(f"EXCEPTION in receive_data: {e}")
        log.append(f"Connection closed from {clients[sock]}")
        sockets_list.remove(sock)
        del clients[sock]
        sock.close()
        numClients -= 1
        updateLog()


def send_data(sock, data):
    global log
    global sockets_list
    global clients
    global numClients
    try:
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
    except Exception as e:
        log.append(f"EXCEPTION in send_data: {e}")
        log.append(f"Connection closed from {clients[sock]}")
        sockets_list.remove(sock)
        del clients[sock]
        sock.close()
        numClients -= 1
        updateLog()


def send_nparrays(sock, data: list[np.ndarray]):
    global log
    global sockets_list
    global clients
    global numClients
    try:
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
    except Exception as e:
        log.append(f"EXCEPTION in send_nparrays: {e}")
        log.append(f"Connection closed from {clients[sock]}")
        sockets_list.remove(sock)
        del clients[sock]
        sock.close()
        numClients -= 1
        updateLog()


def saveWeights(folder, weights):
    # Get the current date and time
    now = datetime.now()

    # Convert to a string
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Make sure folder exists
    folderpath = f"{folder}/checkpoints"
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    # Write to file
    with open(f"{folderpath}/{date_time_str}.bin", "wb") as f:
        pickle.dump(weights, f)


# Training setup
nTrials = 100
alpha = 2e-4
sigma = 0.01
hiddenSize = 16
checkPointSeconds = 1 * 60
savePath = "multiWorkerModels/0"

tokenLoader = loadTokens("tokenData")
fileNum, title, tokens = next(tokenLoader)
vocabSize = len(vocab.vocab)

model_initState = np.random.random((hiddenSize)).astype(np.float32)
model_ih = np.random.random((vocabSize, hiddenSize)).astype(np.float32)
model_hh = np.random.random((hiddenSize, hiddenSize)).astype(np.float32)
model_out = np.random.random((hiddenSize, vocabSize)).astype(np.float32)
weights = [model_initState, model_ih, model_hh, model_out]
weightShapes = [item.shape for item in weights]

# Server setup
CHUNK_SIZE = 64
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 8080))
server_socket.listen(10000)
server_socket.settimeout(10)

sockets_list = [server_socket]
clients = {}
numClients = 0
new_clients_list = []
receivingWeightsFrom = -1

# Trackers setup
mean = "N/A"
log = []
lastCheckpointTime = perf_counter()
totalIters = 0
totalSamples = 0


print("Server started and listening")
print("\nLOG:")
print("\n" * 3)


def updateLog():
    global log, lastCheckpointTime, checkPointSeconds
    clearLines(6)
    for line in log:
        print(line)
    log = []
    print()
    print(f"Iter #: {totalIters:,}")
    print(f"Total # Samples Taken: {totalSamples:,}")
    print(f"# Clients: {numClients}")
    print(
        f"Time until next checkpoint: {checkPointSeconds - (perf_counter() - lastCheckpointTime):,.2f}"
    )
    print(f"Mean Reward: {mean}")


while True:
    # Handle any new connections
    read_sockets, _, exception_sockets = select.select(
        sockets_list, [], sockets_list, 0.1
    )

    if server_socket in read_sockets:
        client_socket, client_address = server_socket.accept()
        log.append(f"New connection from {client_address}")
        sockets_list.append(client_socket)
        clients[client_socket] = client_address
        numClients += 1
        updateLog()

        if numClients == 1:
            send_nparrays(client_socket, weights)
            send_data(
                client_socket,
                [alpha, sigma, vocabSize, weightShapes, True],
            )
        else:
            new_clients_list.append(client_socket)

        updateLog()

    # Get new tokens
    if len(sockets_list) > len(new_clients_list) + 1:
        try:
            fileNum, title, tokens = next(tokenLoader)
        except StopIteration:
            tokenLoader = loadTokens("tokenData")
            fileNum, title, tokens = next(tokenLoader)
    tokens = tokens[:200]

    # Check if we want to get a checkpoint
    requestCheckpoint = (perf_counter() - lastCheckpointTime) > checkPointSeconds

    # Broadcast done, weight request, seeds and nTrials for each client
    nSeeds = len(sockets_list) - 1 - len(new_clients_list)
    seeds = np.random.randint(0, 1_000_000_000, nSeeds)
    workerInfo = {}

    for i, client_socket in enumerate(sockets_list):
        if client_socket == server_socket or client_socket in new_clients_list:
            continue

        need_weights = (
            True
            if (
                receivingWeightsFrom == -1
                and (len(new_clients_list) > 0 or requestCheckpoint)
            )
            else False
        )

        workerID = len(workerInfo)
        workerInfo[client_socket] = (workerID, seeds[workerID])

        if need_weights:
            receivingWeightsFrom = workerID

        send_data(
            client_socket,
            [False, need_weights, seeds, nTrials, workerID, tokens, alpha, sigma],
        )
        workerID += 1

    # Receive the rewards from each client
    for client_socket in sockets_list:
        if client_socket == server_socket or client_socket in new_clients_list:
            continue

        R = receive_nparrays(client_socket)
        if R is not None:
            workerInfo[client_socket] = (
                workerInfo[client_socket][0],
                workerInfo[client_socket][1],
                R[0],
            )
        else:
            if client_socket in workerInfo:
                del workerInfo[client_socket]
            continue

        if workerInfo[client_socket][0] == receivingWeightsFrom:
            # If recieving weights, handle getting the weights
            weights = receive_nparrays(client_socket)
            if weights is None:
                if client_socket in workerInfo:
                    del workerInfo[client_socket]
                continue

            if requestCheckpoint:
                lastCheckpointTime = perf_counter()
                saveWeights(savePath, weights)

            # send weights to new clients
            for new_client in new_clients_list:
                send_nparrays(new_client, weights)
                send_data(
                    new_client,
                    [alpha, sigma, vocabSize, weightShapes, False],
                )

    # reset
    new_clients_list = []
    receivingWeightsFrom = -1

    # Normalize rewards
    all_R = list([item[2] for item in workerInfo.values()])

    success = len(all_R) != 0
    if not success:
        A = np.array([])
        # Send A to each client to update their weights
        for client_socket in sockets_list:
            if client_socket == server_socket or client_socket in new_clients_list:
                continue

            send_data(client_socket, (success, []))
    else:
        R = np.concatenate(all_R)
        mean = np.mean(R)
        std = np.std(R)
        A = (R - mean) / std

        info = []
        for k, v in workerInfo.items():
            info.append((v[0], v[1], len(v[2])))
        info = sorted(info, key=lambda x: x[0])

        # Send A to each client to update their weights
        for client_socket in sockets_list:
            if client_socket == server_socket or client_socket in new_clients_list:
                continue

            send_data(client_socket, (success, info))
            send_nparrays(client_socket, [A])

        # Increment counter
        totalIters += 1
        totalSamples += len(A)
        updateLog()
