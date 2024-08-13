import os
import json
import math
import torch
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


def getNumPages(folder):
    with open(f"{folder}/info.txt", "r", encoding="utf-8") as f:
        text = f.read()
        numPages = int(text.split(" ")[0].strip())
    return numPages


@profile
def main():
    # Hyperparameters
    vocabSize = len(vocab)
    hiddenDim = 128
    numEpochs = 1_000_000
    learningRate = 0.001
    batchSize = 128

    # Settings
    modelSavePath = "models/embedArticle/0"
    saveInterval = 10
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
    # model: RNNEmbedder = torch.load(
    #    f"{modelSavePath}/model.pt", map_location=device, weights_only=False
    # )
    model = RNNEmbedder(vocabSize, hiddenDim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    clearLines(1)
    print(f"Sub-Model Parameter Information:")
    print(f"Vocab Size: {vocabSize:,}")
    print(f"Hidden Dim: {hiddenDim:,}")
    print(f"# Embedding Params: {vocabSize * hiddenDim:,}")
    print(f"# Input->Hidden Params: {hiddenDim * hiddenDim:,}")
    print(f"# Hidden->Hidden Params: {hiddenDim * hiddenDim:,}")
    print(f"# Hidden Bias Params: {hiddenDim:,}")
    print(
        f"Sub-Model Total # Params: {vocabSize * hiddenDim + 2 * hiddenDim * hiddenDim + hiddenDim:,}"
    )
    print(
        f"Embedder Total # Params (2 x subModel): {sum([p.numel() for p in model.parameters()]):,}"
    )
    print()

    # Get all titles
    print(f"Loading all page titles...")
    titles = []
    for _, _, titleTokens in loadTitles(tokenFolder):
        titles.append(titleTokens)
    clearLines(1)
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
        infoText += f"Hidden Dim: {hiddenDim}\n"
        infoText += f"Learning Rate: {learningRate}\n"
        infoText += f"# Pages Per Epoch: {numPagesPerEpoch}\n"
        infoText += f"# Tokens Per Epoch: {numTokensPerEpoch}\n"
        f.write(infoText)

    with open(f"{modelSavePath}/loss.txt", "w", encoding="utf-8") as f:
        f.write("# Pages Trained On, # Tokens Trained On, Loss\n")

    clearLines(1)
    print("Training...\n")

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

        titleStates = torch.zeros(batchSize, hiddenDim, device=device)
        otherTitleStates = torch.zeros(batchSize, hiddenDim, device=device)
        articleStates = torch.zeros(batchSize, hiddenDim, device=device)

        for stepNum in range(numBatches):
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
            stepNum += 1

            # sort batch by article length, and get lengths
            batch = sorted(batch, key=lambda x: len(x[1]), reverse=False)
            lengths = [len(item[1]) for item in batch]
            lengths.insert(0, 0)

            # Prepare for forward pass
            optimizer.zero_grad()  # Zero all gradients in the model
            model.preCompute()  # Pre-Compute variables for a faster forward pass

            # Reset the states
            titleStates = titleStates.detach()
            otherTitleStates = otherTitleStates.detach()
            articleStates = articleStates.detach()

            # Non-batched processing of titles cause titles are short
            for i in trange(adjustedBatchSize, desc="Getting Title Embeddings"):
                # Get text embedding for title
                state = torch.zeros(model.hiddenDim, device=device)
                for token in batch[i][0]:
                    state = model.titleModel.forwardEmbedded(state, token)
                titleStates[i] = state

                # Get random "wrong" title that is not equal to correct title
                otherTitleTokens = random.choice(titles)
                while otherTitleTokens == batch[i][0]:
                    otherTitleTokens = random.choice(titles)

                # Get text embedding for "wrong" title
                otherState = torch.zeros(model.hiddenDim, device=device)
                for token in otherTitleTokens:
                    otherState = model.titleModel.forwardEmbedded(otherState, token)
                otherTitleStates[i] = otherState

            # Batch process articles
            with tqdm(total=lengths[-1], desc="Getting Article Embeddings") as pbar:
                for i in range(len(lengths) - 1):
                    for j in range(lengths[i], lengths[i + 1]):
                        tokens = [item[1][j] for item in batch[i:]]
                        tokens = torch.tensor(tokens, device=device, dtype=torch.int64)
                        newArticleStates = model.textModel.forwardEmbedded(
                            articleStates[i:], tokens
                        )
                        articleStates = torch.cat(
                            [articleStates[:i], newArticleStates], dim=0
                        )
                    pbar.update(lengths[i + 1] - lengths[i])

            # Normalize embeddings
            titleStates = titleStates * (
                1 / titleStates.norm(dim=-1).unsqueeze(-1)
            )  # correct title
            otherTitleStates = otherTitleStates * (
                1 / otherTitleStates.norm(dim=-1).unsqueeze(-1)
            )  # wrong title
            articleStates = articleStates * (
                1 / articleStates.norm(dim=-1).unsqueeze(-1)
            )  # article

            # Get dot products
            correctDot = torch.sum(titleStates * articleStates, dim=-1)
            wrongDot = torch.sum(otherTitleStates * articleStates, dim=-1)

            # Get loss (best possible loss value is 0)
            # minimize wrong dot, maximize correctDot
            loss = wrongDot - correctDot  # [-2, 2]
            loss = (-2 - loss) ** 2  # [0, 16]
            loss = loss[
                :adjustedBatchSize
            ]  # if we are on the last batch, we only use part of the states
            loss = torch.mean(loss)

            # Backpropogation
            print("Doing backpropagation...")
            loss.backward()
            optimizer.step()

            # Track loss and numPages
            stop = perf_counter()
            lossValue = loss.item()
            totalLoss += lossValue
            lastLoss = lossValue
            windowLoss += lossValue
            windowSteps += 1
            numPages += adjustedBatchSize

            print(f"Title Model Embedding Grad: {model.titleModel.embedding.grad.norm()}")
            print(f"Title Model I->H Grad: {model.titleModel.ih.grad.norm()}")
            print(f"Title Model H->H Grad: {model.titleModel.hh.grad.norm()}")
            print(f"Title Model Bias Grad: {model.titleModel.hh.grad.norm()}")

            print(f"Text Model Embedding Grad: {model.textModel.embedding.grad.norm()}")
            print(f"Text Model I->H Grad: {model.textModel.ih.grad.norm()}")
            print(f"Text Model H->H Grad: {model.textModel.hh.grad.norm()}")
            print(f"Text Model Bias Grad: {model.textModel.hh.grad.norm()}")

            batchNumTokens = sum([len(item[0]) + len(item[1]) for item in batch])
            lastTokSec = int(batchNumTokens / (stop - start))
            numTokens += batchNumTokens

            # Save the trained model
            if stepNum % saveInterval == 0:
                print("Saving model...")
                torch.save(model, f"{modelSavePath}/model.pt")
                with open(f"{modelSavePath}/loss.txt", "a", encoding="utf-8") as f:
                    f.write(
                        f"{epoch * numPagesPerEpoch + stepNum}, {epoch * numTokensPerEpoch + numTokens}, {windowLoss / windowSteps}\n"
                    )
                windowLoss = 0
                windowSteps = 0
                clearLines(1)

            # Clear logging so we are ready for the next step
            clearLines(4+8)

        print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {totalLoss/numPages:.4f}")

        # Save the trained model
        torch.save(model, f"{modelSavePath}/model.pt")
        with open(f"{modelSavePath}/loss.txt", "a", encoding="utf-8") as f:
            f.write(
                f"{epoch * numPagesPerEpoch + stepNum}, {epoch * numTokensPerEpoch + numTokens}, {lastLoss}\n"
            )


if __name__ == "__main__":
    main()
