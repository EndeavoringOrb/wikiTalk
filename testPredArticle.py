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
from tokenizeWiki import loadTokens, loadTitles, countNumTokens, decode, tokenize
from time import perf_counter

from line_profiler import profile
import os

os.environ["LINE_PROFILE"] = "0"


def getNumPages(folder):
    with open(f"{folder}/info.txt", "r", encoding="utf-8") as f:
        text = f.read()
        numPages = int(text.split(" ")[0].strip())
    return numPages


@torch.no_grad
@profile
def main():
    # Settings
    modelSavePath = "models/tokenPredArticle2/current"

    device = torch.device(
        "cpu"
    )  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model, loss function, and optimizer
    print("Initializing model...")
    model: RecurrentTransformer = torch.load(
        f"{modelSavePath}/model.pt", weights_only=False, map_location=device
    )
    clearLines(1)
    print(f"Sub-Model Parameter Information:")
    print(f"Vocab Size: {model.vocabSize:,}")
    print(f"Hidden Dim: {model.hiddenSize:,}")
    print(f"Model Total # Params: {sum([p.numel() for p in model.parameters()]):,}")

    while True:
        state = model.initState
        numTokens = int(input("\n\nEnter # of tokens to generate: "))
        print(f"Generating...")

        start = perf_counter()

        tokens = []

        for i in range(numTokens):
            pred = model.getPreds(state)
            probs = F.softmax(pred, dim=0)
            token = torch.multinomial(probs, 1)
            tokens.append(token)
            state = model.nextState(state, token)

        elapsed = perf_counter() - start

        print(f"Generated {numTokens} tokens in {elapsed:.2f} s.")
        print(f"{numTokens / elapsed:.2f} tok/sec\n")

        text = decode([token.item() for token in tokens])
        print(text, flush=True)


if __name__ == "__main__":
    main()
