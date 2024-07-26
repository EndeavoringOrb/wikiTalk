import torch
import torch.optim as optim
import torch.nn as nn
import json
import os
from model import *
from tqdm import tqdm
from helperFuncs import *

modelSavePath = "models/embed/0.pt"

# Initialize the model, loss function, and optimizer
print("Loading model...")
model = torch.load(modelSavePath)

dataFolder = "conversationData"


def getNumSteps(folder):
    files: list[str] = os.listdir(folder)
    files = [file for file in files if file.endswith(".json")]
    files = sorted(files, key=lambda x: int(x.split(".")[0]))

    numSteps = 0

    for file in files:
        with open(f"{folder}/{file}", "r", encoding="utf-8") as f:
            data: list[str] = json.load(f)
        for chunk in data:
            numSteps += len(chunk)

    return numSteps


def dataLoader(folder):
    files: list[str] = os.listdir(folder)
    files = [file for file in files if file.endswith(".json")]
    files = sorted(files, key=lambda x: int(x.split(".")[0]))

    for file in files:
        with open(f"{folder}/{file}", "r", encoding="utf-8") as f:
            data: list[str] = json.load(f)
        yield (None, False, -1)
        for chunk in data:
            chunk_tensor = torch.tensor(
                [mainVocab[character] for character in chunk.lower()]
            )
            if chunk.startswith("User: "):
                train = False
                startTrain = -1
            elif chunk.startswith("Assistant: "):
                train = True
                startTrain = 11
            elif chunk.startswith("-"):
                train = False
                startTrain = -1
            elif chunk == "+talk" or chunk == "+search" or chunk == "+get article":
                train = True
                startTrain = 1
            elif chunk.startswith("Enter query: "):
                train = True
                startTrain = 13
            elif chunk.startswith("Enter Article #: "):
                train = True
                startTrain = 17

            yield (chunk_tensor, train, startTrain)


# Create vocab
vocabChars = "abcdefghijklmnopqrstuvwxyz0123456789 ():.-',/?!&+#{}|\n=_[]<>;\"%*@$"
vocabChars += "—–ł"
mainVocab = {character: idx for idx, character in enumerate(vocabChars)}

if __name__ == "__main__":
    # Hyperparameters
    vocab_size = len(vocabChars)
    embeddingDim = 32
    hiddenDim = 16
    numEpochs = 100
    learningRate = 0.001

    # Settings
    modelSavePath = "models/main/0.pt"

    # Initialize the model, loss function, and optimizer
    print("Initializing model...")
    model = RNNLanguage(vocab_size, embeddingDim, hiddenDim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    stepsPerEpoch = getNumSteps("conversationData")
    print(f"{stepsPerEpoch:,} steps per epoch.")

    # Training loop
    for epoch in range(numEpochs):
        total_loss = 0
        # Prepare your data
        dataGen = dataLoader("conversationData")

        # Init state
        state = torch.zeros(hiddenDim)

        numTrainSteps = 0
        numSteps = 0

        loss = 0

        with tqdm(total=stepsPerEpoch, desc=f"Epoch [{epoch+1}/{numEpochs}]") as pbar:
            for sequence, train, trainStart in dataGen:
                if sequence == None:
                    state = torch.zeros(hiddenDim)

                    if loss > 0:
                        # Compute loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()

                    loss = 0
                    continue

                if train:
                    for i, token in enumerate(sequence):
                        if i >= trainStart:
                            logits = model.logits(state)
                            loss += criterion(logits, token)
                            numTrainSteps += 1
                        state = model(state, token)
                        pbar.update(1)

                    numSteps += len(sequence)
                else:
                    for i, token in enumerate(sequence):
                        state = model(state, token)
                        pbar.update(1)

        print(
            f"Epoch [{epoch+1}/{numEpochs}], Loss: {total_loss/numTrainSteps if numTrainSteps > 0 else 'nan':.4f}"
        )

        # Save the trained model
        torch.save(model, modelSavePath)
