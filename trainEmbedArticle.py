import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from helperFuncs import *
from tqdm import tqdm, trange
from model import *
from vocab import *
from tokenizeWiki import read_compact_data


def getNumPages(folder):
    with open(f"{folder}/info.txt", "r", encoding="utf-8") as f:
        text = f.read()
        numPages = int(text.split(" ")[0].strip())
    return numPages


def loadTextTokens(folder):
    files = sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0]))
    for file in files:
        for titleTokens, textTokens in read_compact_data(f"{folder}/{file}"):
            yield textTokens


# Create vocab
# vocabChars = "abcdefghijklmnopqrstuvwxyz0123456789 ():.-'\",/?!&"
# vocabChars += "éöōáíüłçóèäńøæãðūëòà+ñ̇ğâāå♯żαđúćıʼęìň×ạấýσêš½ŵôčąőδḥ*șśşʻïăēþîọīřț—ž¡²ṛķņœễěβõếû…ß°ṯṟμ"
# vocabChars += "źπṅảʽẩầứồươệļģỏ′ė­ṃů@=ÿ″ǫ̨ħ−ǂǃŭŝĵĥĝĉƒùť$ụĩũŏ%ṣủẹəỳữ£ǐľʿǁġṇ­­­­­­ṭ高雄ḫ道⅓∞űởờ¹^ỉ₂ḍḷ\\ẻʾį³ɛ̃ỹậộꞌʹ"
# vocabChars += "ǀị;∴~κắċ̄±ṉųớợằ–·→ố⟨⟩京東ďỗửừḵẫ₀ĕŷự꞉•"
# vocabChars = sorted(list(set(vocabChars + vocabChars.lower() + vocabChars.upper())))
# vocabBIG = {character: idx for idx, character in enumerate(vocabChars)}

if __name__ == "__main__":
    # Hyperparameters
    vocab_size = len(vocab)
    hidden_dim = 128
    num_epochs = 100
    learning_rate = 0.001

    # Settings
    modelSavePath = "models/embedArticle/0.pt"
    tokenFolder = "tokenData"
    totalNumPages = 551_617

    device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Prepare your data
    numFiles = len(os.listdir(tokenFolder))

    # Initialize the model, loss function, and optimizer
    print("Initializing model...")
    model = RNNLanguage(vocab_size, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Hidden Dim: {hidden_dim:,}")
    print(f"# Embedding Params: {vocab_size * hidden_dim:,}")
    print(f"# Input->Hidden Params: {hidden_dim * hidden_dim:,}")
    print(f"# Hidden->Hidden Params: {hidden_dim * hidden_dim:,}")
    print(f"# Hidden Bias Params: {hidden_dim * hidden_dim:,}")
    print(f"Total # Params: {vocab_size * hidden_dim + 2 * hidden_dim * hidden_dim + hidden_dim:,}")


    # Training loop
    for epoch in range(num_epochs):
        totalLoss = 0
        lastLoss = "N/A"
        numPages = 0

        for stepNum, textTokens in enumerate(loadTextTokens(tokenFolder)):
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{stepNum + 1}/{totalNumPages}], Last Loss: {lastLoss}")
            # zero model grad
            optimizer.zero_grad()

            # tokenize text
            textTensor = torch.tensor(textTokens, device=device)

            # Get text embedding by passing text through model
            state = torch.zeros(model.hiddenDim, device=device)
            for token in tqdm(textTokens, desc="Getting Embedding"):
                state = model(state, token)

            # Initialize loss
            loss = 0

            # Get reconstruction loss for text
            for token in tqdm(textTensor, desc="Reconstructing"):
                logits = model.logits(state)  # get logits
                loss += criterion(logits, token)  # accumulate loss
                state = model(state, token)  # get next state

            # Backpropogation
            print("Doing backpropagation...")
            loss.backward()
            optimizer.step()

            # Track loss and numPages
            lossValue = loss.item()
            totalLoss += lossValue
            lastLoss = lossValue / len(textTokens)
            numPages += 1

            # Save the trained model
            print("Saving model...")
            torch.save(model, modelSavePath)

            clearLines(5)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {totalLoss/numPages:.4f}")

        # Save the trained model
        torch.save(model, modelSavePath)
