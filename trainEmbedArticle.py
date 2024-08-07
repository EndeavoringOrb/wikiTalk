import os
import json
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


"""
Optimization Log (finish training on 6 pages)

Baseline
47.09, 42.57, 48.24, 42.05

use preCompute in training loop which precomputes hhScaled and embedded
17.73, 17.55, 19.20, 20.05, 20.38
"""


def getNumPages(folder):
    with open(f"{folder}/info.txt", "r", encoding="utf-8") as f:
        text = f.read()
        numPages = int(text.split(" ")[0].strip())
    return numPages


# Create vocab
# vocabChars = "abcdefghijklmnopqrstuvwxyz0123456789 ():.-'\",/?!&"
# vocabChars += "éöōáíüłçóèäńøæãðūëòà+ñ̇ğâāå♯żαđúćıʼęìň×ạấýσêš½ŵôčąőδḥ*șśşʻïăēþîọīřț—ž¡²ṛķņœễěβõếû…ß°ṯṟμ"
# vocabChars += "źπṅảʽẩầứồươệļģỏ′ė­ṃů@=ÿ″ǫ̨ħ−ǂǃŭŝĵĥĝĉƒùť$ụĩũŏ%ṣủẹəỳữ£ǐľʿǁġṇ­­­­­­ṭ高雄ḫ道⅓∞űởờ¹^ỉ₂ḍḷ\\ẻʾį³ɛ̃ỹậộꞌʹ"
# vocabChars += "ǀị;∴~κắċ̄±ṉųớợằ–·→ố⟨⟩京東ďỗửừḵẫ₀ĕŷự꞉•"
# vocabChars = sorted(list(set(vocabChars + vocabChars.lower() + vocabChars.upper())))
# vocabBIG = {character: idx for idx, character in enumerate(vocabChars)}


@profile
def main():
    # Hyperparameters
    vocabSize = len(vocab)
    hiddenDim = 128
    numEpochs = 100
    learningRate = 0.001

    # Settings
    modelSavePath = "models/embedArticle/0"
    saveInterval = 10
    tokenFolder = "tokenData"
    totalNumPages = 551_617

    device = torch.device(
        "cpu"
    )  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Initialize the model, loss function, and optimizer
    print("Initializing model...")
    model = RNNEmbedder(vocabSize, hiddenDim).to(device)
    # model = torch.compile(og_model)
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    print(f"Hidden Dim: {hiddenDim:,}")
    print(f"# Embedding Params: {vocabSize * hiddenDim:,}")
    print(f"# Input->Hidden Params: {hiddenDim * hiddenDim:,}")
    print(f"# Hidden->Hidden Params: {hiddenDim * hiddenDim:,}")
    print(f"# Hidden Bias Params: {hiddenDim * hiddenDim:,}")
    print(
        f"Total # Params: {vocabSize * hiddenDim + 2 * hiddenDim * hiddenDim + hiddenDim:,}"
    )

    # Get all titles
    print(f"Getting all titles...")
    titles = []
    for fileIndex, pageIndex, titleTokens in loadTitles(tokenFolder):
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

        for stepNum, (titleTokens, textTokens) in enumerate(loadTokens(tokenFolder)):
            print(
                f"Epoch [{epoch+1}/{numEpochs}], Step [{stepNum + 1}/{totalNumPages}], Last Loss: {lastLoss}, Last Tok/Sec: {lastTokSec}"
            )
            start = perf_counter()
            # zero model grad
            optimizer.zero_grad()

            model.preCompute()

            # Get text embedding for title
            state = torch.zeros(model.hiddenDim, device=device)
            for token in tqdm(titleTokens, desc="Getting Title Embedding"):
                state = model.titleModel.forwardEmbedded(state, token)

            # Get text embedding for text
            articleState = torch.zeros(model.hiddenDim, device=device)
            for token in tqdm(textTokens, desc="Getting Text Embedding"):
                articleState = model.textModel.forwardEmbedded(
                    articleState, token
                )  # get next state

            # Get text embedding for "wrong" title
            otherTitleTokens = random.choice(titles)
            while otherTitleTokens == titleTokens:
                otherTitleTokens = random.choice(titles)
            otherState = torch.zeros(model.hiddenDim, device=device)
            for token in tqdm(otherTitleTokens, desc="Getting Other Title Embedding"):
                otherState = model.titleModel.forwardEmbedded(otherState, token)

            # Normalize embeddings
            state = state * (1 / state.norm())
            articleState = articleState * (1 / articleState.norm())
            otherState = otherState * (1 / otherState.norm())

            # Get dot products
            correctDot = torch.sum(state * articleState)
            wrongDot = torch.sum(otherState * articleState)

            # Get loss (best value is -2, worst value is 2)
            loss = wrongDot - correctDot  # minimize wrong dot, maximize correctDot

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
            lastTokSec = int((2 * len(titleTokens) + len(textTokens)) / (stop - start))
            numPages += 1
            numTokens += len(titleTokens) + len(textTokens)

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

            # Exit after N pages for benchmarking purposes
            # if stepNum == 19:
            #   exit(0)

            clearLines(5)

        print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {totalLoss/numPages:.4f}")

        # Save the trained model
        torch.save(model, f"{modelSavePath}/model.pt")
        with open(f"{modelSavePath}/loss.txt", "a", encoding="utf-8") as f:
            f.write(
                f"{epoch * numPagesPerEpoch + stepNum}, {epoch * numTokensPerEpoch + numTokens}, {lastLoss}\n"
            )


if __name__ == "__main__":
    main()
