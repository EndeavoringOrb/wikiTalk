import os
import json
import math
import torch
import pickle
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from helperFuncs import *
from tqdm import tqdm, trange
from model import *
from vocab import *
from tokenizeWiki import loadTokens, loadTitles, countNumTokens
from time import perf_counter
import copy

from line_profiler import profile
import os

os.environ["LINE_PROFILE"] = "0"


def getNumPages(folder):
    with open(f"{folder}/info.txt", "r", encoding="utf-8") as f:
        text = f.read()
        numPages = int(text.split(" ")[0].strip())
    return numPages


@torch.no_grad()
def getNewModel(model, std=0.1):
    # Create a clone of the original model
    newModel = copy.deepcopy(model)

    # Iterate through the parameters of the copied model
    for param in newModel.parameters():
        # Generate random normal noise with the same shape as the parameter, and with specified std
        noise = torch.randn_like(param) * std

        # Add the noise to the parameter
        param.data += noise

    return newModel


@profile
def main():
    # Hyperparameters
    vocabSize = len(vocab)
    hiddenSize = 64
    numEpochs = 1_000_000
    learningRate = 2e-4
    batchSize = 1
    nHead = 4
    headSize = 16
    nLayer = 4

    nPop = 150  # population size
    sigma = 0.01  # noise standard deviation

    # Settings
    modelLoadPath = "models/tokenPredArticle/current"
    modelSavePath = "models/tokenPredArticle/current"
    saveInterval = 1
    tokenFolder = "tokenData"

    device = torch.device(
        "cpu"
    )  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training Parameters:")
    print(f"# of Epochs: {numEpochs:,}")
    print(f"Learning Rate: {learningRate:,}")
    print(f"Batch Size: {batchSize:,}")
    print(f"Training Device: {device}")
    print()

    # Initialize the model, loss function, and optimizer
    print("Initializing model...")
    #model: RecurrentTransformer = torch.load(
    #   f"{modelLoadPath}/model.pt", weights_only=False, map_location=device
    #)
    model: RecurrentTransformer = RecurrentTransformer(
        vocabSize, hiddenSize, nHead, headSize, nLayer, device
    )

    criterion = nn.CrossEntropyLoss()
    R = torch.zeros(nPop)
    A = torch.zeros(nPop)
    wUpdate = [torch.zeros_like(param) for param in model.parameters()]

    clearLines(1)
    print(f"Model Parameter Information:")
    print(f"Vocab Size: {model.vocabSize:,}")
    print(f"Hidden Dim: {model.hiddenSize:,}")
    print(f"Model Total # Params: {sum([p.numel() for p in model.parameters()]):,}")
    print()

    # Get all titles
    print(f"Loading all page titles...")
    titles = []
    for _, _, titleTokens in loadTitles(tokenFolder):
        titles.append(titleTokens)
    print(f"Finished loading")
    print(f"Shuffling titles...")
    random.shuffle(titles)
    clearLines(3)
    print(f"Loaded {len(titles):,} titles")

    # Get number of pages
    print(f"Counting # of pages and # of tokens per epoch...")
    numPagesPerEpoch, numTokensPerEpoch = countNumTokens(tokenFolder)
    clearLines(1)
    print(f"{numPagesPerEpoch:,} pages per epoch")
    print(f"{numTokensPerEpoch:,} tokens per epoch")

    # Save model info
    print(f"Saving model info...")
    os.makedirs(f"{modelSavePath}", exist_ok=True)
    with open(f"{modelSavePath}/info.txt", "w", encoding="utf-8") as f:
        infoText = f"Vocab Size: {vocabSize}\n"
        infoText += f"Hidden Dim: {hiddenSize}\n"
        infoText += f"Learning Rate: {learningRate}\n"
        infoText += f"# Pages Per Epoch: {numPagesPerEpoch}\n"
        infoText += f"# Tokens Per Epoch: {numTokensPerEpoch}\n"
        f.write(infoText)

    with open(f"{modelSavePath}/loss.txt", "w", encoding="utf-8") as f:
        f.write("# Steps Trained, # Tokens Trained On, Loss\n")

    clearLines(1)
    print("Training...\n")

    totalStepNum = 0
    totalNumTokens = 0

    # Training loop
    for epoch in range(numEpochs):
        totalLoss = 0
        windowLoss = 0
        windowSteps = 0
        lastLoss = "N/A"
        lastTokSec = 0
        numPages = 0
        numTokens = 0

        tokenLoader = loadTokens(tokenFolder)

        numBatches = int(math.ceil(numPagesPerEpoch / batchSize))

        stepNum = 0

        for _ in range(numBatches):
            # Print progress, loss and tok/sec
            print(
                f"Epoch [{epoch+1}/{numEpochs}], Batch [{stepNum + 1}/{numBatches}] ({100.0 * (stepNum + 1) /numBatches:.4f}%), Last Loss: {lastLoss}, Last Tok/Sec: {lastTokSec}"
            )

            start = perf_counter()
            batch = []
            adjustedBatchSize = min(batchSize, numPagesPerEpoch - numPages)
            for i in range(adjustedBatchSize):
                fileIndex, titleTokens, textTokens = next(tokenLoader)
                batch.append((titleTokens, textTokens))
            numPages += adjustedBatchSize

            # sort batch by article length, and get lengths
            batch = sorted(batch, key=lambda x: len(x[1]))
            lengths = [len(item[1]) for item in batch]
            lengths.insert(0, 0)

            newModels = []

            with torch.no_grad():
                for trialIdx in range(nPop):
                    newModel: RecurrentTransformer = getNewModel(model, sigma)
                    # Prepare for forward pass
                    newModel.preCompute()  # Pre-Compute variables for a faster forward pass

                    # Reset the states
                    states = newModel.initState.expand(adjustedBatchSize, -1)

                    loss = 0

                    # Batch process articles
                    clearLines(trialIdx > 0)
                    with tqdm(
                        total=lengths[-1], desc=f"Forward Pass {trialIdx + 1}/{nPop}"
                    ) as pbar:
                        for i in range(len(lengths) - 1):
                            # init states and tokens for this length
                            tokens = []
                            for item in batch[i:]:
                                tokens.append(item[1][lengths[i] : lengths[i + 1]])
                            tokens = torch.tensor(
                                tokens, device=device, dtype=torch.int64
                            )

                            # train
                            states, newLoss = newModel.trainPreComputed(
                                states, tokens, criterion
                            )

                            # update
                            loss += newLoss
                            pbar.update(lengths[i + 1] - lengths[i])
                            states = states[1:]

                    loss = loss / lengths[-1]

                    R[trialIdx] = (
                        -loss
                    )  # negative because our evolution maximizes the function

                    newModels.append(newModel)

                A = (R - R.mean()) / R.std()

                for popIdx, newModel in enumerate(newModels):
                    for paramIdx, newParam in enumerate(newModel.parameters()):
                        wUpdate[paramIdx] += newParam * A[popIdx]

                for paramIdx, param in enumerate(model.parameters()):
                    param += (learningRate / (nPop * sigma)) * wUpdate[paramIdx]

            # Track loss and counters
            stop = perf_counter()
            lossValue = -R.mean().item()
            totalLoss += lossValue
            lastLoss = lossValue
            windowLoss += lossValue
            windowSteps += 1
            numPages += adjustedBatchSize

            batchNumTokens = sum([len(item[1]) for item in batch])
            lastTokSec = int(batchNumTokens / (stop - start))
            numTokens += batchNumTokens

            totalStepNum += 1
            totalNumTokens += batchNumTokens

            # Save the trained model
            if stepNum % saveInterval == 0 and stepNum != (numBatches - 1):
                print("Saving model...")
                torch.save(model, f"{modelSavePath}/model.pt")
                with open(f"{modelSavePath}/loss.txt", "a", encoding="utf-8") as f:
                    f.write(
                        f"{totalStepNum}, {totalNumTokens}, {windowLoss / windowSteps}\n"
                    )
                windowLoss = 0
                windowSteps = 0
                clearLines(1)

            # Clear logging so we are ready for the next step
            clearLines(2)

            stepNum += 1

            if stepNum == 1:
                break

        print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {totalLoss/stepNum:.4f}")

        # Save the trained model
        torch.save(model, f"{modelSavePath}/model_E{epoch + 1}.pt")


if __name__ == "__main__":
    main()
