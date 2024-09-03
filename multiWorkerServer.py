import socket
import select
import struct
import pickle
import numpy as np
from tokenizeWiki import loadTokens, vocab

def clearLines(numLines):
    for _ in range(numLines):
        print("\033[F\033[K", end="")

def send_data(sock, data):
    # Encode data
    data_bytes = pickle.dumps(data)
    # Create the header with message length (4 bytes)
    header = struct.pack("!I", len(data_bytes))
    # Send header and message
    sock.sendall(header + data_bytes)


# Training setup
nTrials = 100
alpha = 2e-4
sigma = 0.01
hiddenSize = 16

tokens = [0, 1, 2, 3, 4, 5, 6, 7]
vocabSize = len(vocab.vocab)

model_initState = np.random.random((hiddenSize))
model_ih = np.random.random((vocabSize, hiddenSize))
model_hh = np.random.random((hiddenSize, hiddenSize))
model_out = np.random.random((hiddenSize, vocabSize))
weights = [model_initState, model_ih, model_hh, model_out]

# Server setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 8080))
server_socket.listen(5)

sockets_list = [server_socket]
clients = {}
numClients = 0
new_clients_list = []
receivingWeightsFrom = -1

print("Server started and listening")
print()

while True:
    # Handle any new connections
    print("Checking for new connections")
    read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list, 0.1)

    # Set seeds
    seeds = np.random.randint(0, 1_000_000_000, len(sockets_list) - 1)

    if server_socket in read_sockets:
        client_socket, client_address = server_socket.accept()
        print(f"New connection from {client_address}")
        sockets_list.append(client_socket)
        clients[client_socket] = client_address
        numClients += 1

        if numClients == 1:
            # Set seeds again with new length
            seeds = np.random.randint(0, 1_000_000_000, len(sockets_list) - 1)
            send_data(client_socket, [weights, seeds, nTrials, alpha, sigma, vocabSize, True])
        else:
            new_clients_list.append(client_socket)

    # Broadcast done, weight request, seeds and nTrials for each client
    for i, client_socket in enumerate(sockets_list):
        if client_socket == server_socket or client_socket in new_clients_list:
            continue

        print(f"Sending data to {clients[client_socket]}")
        need_weights = (
            True if receivingWeightsFrom == -1 and len(new_clients_list) > 0 else False
        )

        if need_weights:
            receivingWeightsFrom = i

        send_data(
            client_socket,
            [False, need_weights, seeds, nTrials, i - 1, tokens, alpha, sigma],
        )
        clearLines(1)

    # Receive the rewards from each client
    all_R = []
    for i, client_socket in enumerate(sockets_list):
        if client_socket == server_socket or client_socket in new_clients_list:
            continue

        print(f"Receiving data from {clients[client_socket]}")
        try:
            header = client_socket.recv(4)
            if not header:
                raise ConnectionResetError()
            message_length = struct.unpack("!I", header)[0]
            message = client_socket.recv(message_length)
            if not message:
                raise ConnectionResetError()
            data = pickle.loads(message)
            if i == receivingWeightsFrom:
                # If recieving weights, handle getting the weights
                R, weights = data
                all_R.append(R)

                # send weights to new clients
                for new_client in new_clients_list:
                    print(f"Sending data to new client {new_client}")
                    send_data(
                        new_client, [weights, seeds, nTrials, alpha, sigma, vocabSize, False]
                    )
                    clearLines(1)

                # reset
                new_clients_list = []
                receivingWeightsFrom = -1
            else:
                all_R.append(data)
            clearLines(1)
        except Exception as e:
            clearLines(1)
            print(e)
            print(f"Connection closed from {clients[client_socket]}")
            sockets_list.remove(client_socket)
            del clients[client_socket]
            client_socket.close()
            numClients -= 1

    # Normalize rewards
    clearLines(1)
    if len(all_R) == 0:
        A = np.array([])
    else:
        print(f"Calculating normalized rewards")
        R = np.concatenate(all_R)
        mean = np.mean(R)
        std = np.std(R)
        A = (R - mean) / std
        clearLines(1)
        print(f"Mean Reward: {mean}")

    # Send A to each client to update their weights
    for client_socket in sockets_list:
        if client_socket == server_socket or client_socket in new_clients_list:
            continue

        print(f"Sending result to {clients[client_socket]}")
        send_data(client_socket, A)
        clearLines(1)

    """
    for notified_socket in read_sockets:
        if notified_socket == server_socket:
            client_socket, client_address = server_socket.accept()
            print(f"New connection from {client_address}")
            sockets_list.append(client_socket)
            clients[client_socket] = client_address
        else:
            try:
                header = notified_socket.recv(4)
                if not header:
                    raise ConnectionResetError()
                message_length = struct.unpack("!I", header)[0]
                message = notified_socket.recv(message_length)
                if not message:
                    raise ConnectionResetError()
                data = pickle.loads(message)
                print(f"Received message from {clients[notified_socket]}: {data}")
                send_data(notified_socket, f"Echo: {data}")
            except:
                print(f"Connection closed from {clients[notified_socket]}")
                sockets_list.remove(notified_socket)
                del clients[notified_socket]
                notified_socket.close()

    for notified_socket in exception_sockets:
        sockets_list.remove(notified_socket)
        del clients[notified_socket]
        notified_socket.close()
    """
