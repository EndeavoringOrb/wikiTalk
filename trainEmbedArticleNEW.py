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

from line_profiler import profile
import os

os.environ["LINE_PROFILE"] = "0"


def getNumPages(folder):
    with open(f"{folder}/info.txt", "r", encoding="utf-8") as f:
        text = f.read()
        numPages = int(text.split(" ")[0].strip())
    return numPages

def getEmbeddingInit(rows, cols, numSteps):
    data = torch.normal(0, 0.02, (rows, cols))
    vectors = torch.nn.Parameter(data)
    optimizer = torch.optim.Adam([vectors], lr=0.001)
    mask = torch.zeros(rows, rows).fill_diagonal_((float("inf")))

    for _ in trange(numSteps):
        optimizer.zero_grad()

        # normalize vectors
        normVal = (1 / vectors.norm(dim=1)).unsqueeze(-1)
        normVecs = vectors * normVal

        # Compute pairwise distances
        distances = torch.cdist(normVecs, normVecs)

        # Set diagonal to a large value to exclude self-distances
        distances = distances + mask

        # Find the minimum distance for each vector
        min_distances = distances.min(dim=1).values

        # Compute the total distance (negative sum of minimum distances)
        totalDist = -min_distances.sum()

        totalDist.backward()
        optimizer.step()
    
    print(f"{totalDist.item() / (2 * rows)}")

    vectors.data /= vectors.norm(dim=1).unsqueeze(-1)

    return vectors


@profile
def main():
    # Hyperparameters
    vocabSize = len(vocab)
    hiddenSize = 32
    embeddingSize = 128
    numEpochs = 1_000_000
    learningRate = 1e-4
    batchSize = 2

    # Settings
    modelSavePath = "models/embedArticle/0"
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
    model: RNNSimilarityEmbedder = torch.load(f"models/embedArticle/0/model.pt", map_location=device, weights_only=False)
    #model = RNNSimilarityEmbedder(vocabSize, hiddenSize, embeddingSize).to(device)
    print(f"Initializing embeddings")
    #model.titleModel.embedding = getEmbeddingInit(vocabSize, hiddenSize, 10000)
    #model.textModel.embedding = getEmbeddingInit(vocabSize, hiddenSize, 10000)
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    clearLines(6)
    print(f"Sub-Model Parameter Information:")
    print(f"Vocab Size: {vocabSize:,}")
    print(f"Hidden Dim: {hiddenSize:,}")
    print(f"# Embedding Params: {vocabSize * hiddenSize:,}")
    print(f"# Input->Hidden Params: {hiddenSize * hiddenSize:,}")
    print(f"# Hidden->Hidden Params: {hiddenSize * hiddenSize:,}")
    print(f"# Hidden Bias Params: {hiddenSize:,}")
    print(f"# Out Projection Params: {hiddenSize * embeddingSize:,}")
    print(
        f"Sub-Model Total # Params: {vocabSize * hiddenSize + 2 * hiddenSize * hiddenSize + hiddenSize + hiddenSize * embeddingSize:,}"
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

        titleStates = torch.zeros(batchSize, hiddenSize, device=device)
        otherTitleStates = torch.zeros(batchSize, hiddenSize, device=device)
        articleStates = torch.zeros(batchSize, hiddenSize, device=device)

        stepNum = 0

        for _ in range(numBatches):
            # Print progress, loss and tok/sec
            print(
                f"Epoch [{epoch+1}/{numEpochs}], Batch [{stepNum + 1}/{numBatches}] ({100.0 * (stepNum + 1) /numBatches:.4f}%), Last Loss: {lastLoss}, Last Tok/Sec: {lastTokSec}"
            )

            # Print model grad
            if stepNum > 0:
                print(
                    f"Title Model Embedding Grad: {model.titleModel.embedding.grad.norm()}"
                )
                print(f"Title Model I->H Grad: {model.titleModel.ih.grad.norm()}")
                print(f"Title Model H->H Grad: {model.titleModel.hh.grad.norm()}")
                print(f"Title Model Bias Grad: {model.titleModel.hh.grad.norm()}")
                print(f"Title Model Out Grad: {model.titleModel.out.grad.norm()}")

                print(
                    f"Text Model Embedding Grad: {model.textModel.embedding.grad.norm()}"
                )
                print(f"Text Model I->H Grad: {model.textModel.ih.grad.norm()}")
                print(f"Text Model H->H Grad: {model.textModel.hh.grad.norm()}")
                print(f"Text Model Bias Grad: {model.textModel.hh.grad.norm()}")
                print(f"Text Model Out Grad: {model.textModel.out.grad.norm()}")

            start = perf_counter()
            batch = []
            adjustedBatchSize = min(batchSize, numPagesPerEpoch - numPages)
            for i in range(adjustedBatchSize):
                fileIndex, titleTokens, textTokens = next(tokenLoader)
                batch.append((titleTokens, textTokens))
            numPages += adjustedBatchSize

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
                state = torch.zeros(model.hiddenSize, device=device)
                for token in batch[i][0]:
                    state = model.titleModel.forwardEmbeddedNoBatch(state, token)
                titleStates[i] = state

                # Get random "wrong" title that is not equal to correct title
                otherTitleTokens = random.choice(titles)
                while otherTitleTokens == batch[i][0]:
                    otherTitleTokens = random.choice(titles)

                # Get text embedding for "wrong" title
                otherState = torch.zeros(model.hiddenSize, device=device)
                for token in otherTitleTokens:
                    otherState = model.titleModel.forwardEmbeddedNoBatch(
                        otherState, token
                    )
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
            titleEmbeddings = model.titleModel.getOut(titleStates)
            otherTitleEmbeddings = model.titleModel.getOut(otherTitleStates)
            articleEmbeddings = model.textModel.getOut(articleStates)

            titleEmbeddings = titleEmbeddings * (
                1 / titleEmbeddings.norm(dim=-1).unsqueeze(-1)
            )  # correct title
            otherTitleEmbeddings = otherTitleEmbeddings * (
                1 / otherTitleEmbeddings.norm(dim=-1).unsqueeze(-1)
            )  # wrong title
            articleEmbeddings = articleEmbeddings * (
                1 / articleEmbeddings.norm(dim=-1).unsqueeze(-1)
            )  # article

            # Get dot products
            correctDot = torch.sum(titleEmbeddings * articleEmbeddings, dim=-1)
            wrongDot = torch.sum(otherTitleEmbeddings * articleEmbeddings, dim=-1)

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

            # Track loss and counters
            stop = perf_counter()
            lossValue = loss.item()
            totalLoss += lossValue
            lastLoss = lossValue
            windowLoss += lossValue
            windowSteps += 1
            numPages += adjustedBatchSize

            batchNumTokens = sum([len(item[0]) + len(item[1]) for item in batch])
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
            clearLines(4 + (10 if stepNum > 0 else 0))

            stepNum += 1
            

            # FOR TESTING ONLY
            # Stop after first few batches to see if we can overfit
            if stepNum == 1:
                break

        print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {totalLoss/stepNum:.4f}")

        # Save the trained model
        torch.save(model, f"{modelSavePath}/model_E{epoch + 1}.pt")


if __name__ == "__main__":
    main()
