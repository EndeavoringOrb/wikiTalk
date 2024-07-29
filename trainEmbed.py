import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from helperFuncs import *
from tqdm import tqdm, trange
from model import *


# Define a custom dataset
class TextPairDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, title, same = self.data[idx]
        query_tensor = torch.tensor(
            [self.vocab[character] for character in query.lower()]
        )
        title_tensor = torch.tensor(
            [self.vocab[character] for character in title.lower()]
        )
        target = torch.tensor(1.0 if same else -1.0)
        return query_tensor, title_tensor, target


def loadEmbedData(folder):
    print("Loading data...")
    files: list[str] = os.listdir(folder)
    files = [file for file in files if file.endswith(".json")]

    all_data = []

    for file in files:
        with open(f"{folder}/{file}", "r", encoding="utf-8") as f:
            data = json.load(f)
        all_data.extend(data)

    print(f"{len(all_data):,} items loaded.")

    return all_data


# Create vocab
vocabChars = "abcdefghijklmnopqrstuvwxyz0123456789 ():.-'\",/?!&"
vocabChars += "éöōáíüłçóèäńøæãðūëòà+ñ̇ğâāå♯żαđúćıʼęìň×ạấýσêš½ŵôčąőδḥ*șśşʻïăēþîọīřț—ž¡²ṛķņœễěβõếû…ß°ṯṟμ"
vocabChars += "źπṅảʽẩầứồươệļģỏ′ė­ṃů@=ÿ″ǫ̨ħ−ǂǃŭŝĵĥĝĉƒùť$ụĩũŏ%ṣủẹəỳữ£ǐľʿǁġṇ­­­­­­ṭ高雄ḫ道⅓∞űởờ¹^ỉ₂ḍḷ\\ẻʾį³ɛ̃ỹậộꞌʹ"
vocabChars += "ǀị;∴~κắċ̄±ṉųớợằ–·→ố⟨⟩京東ďỗửừḵẫ₀ĕŷự꞉•"
vocabChars = sorted(list(set(vocabChars.lower())))
vocab = {character: idx for idx, character in enumerate(vocabChars)}

if __name__ == "__main__":
    # Hyperparameters
    vocab_size = len(vocabChars)
    embedding_dim = 32
    hidden_dim = 16
    num_epochs = 100
    learning_rate = 0.001

    # Settings
    modelSavePath = "models/embed/0.pt"

    # Prepare your data
    data = loadEmbedData("embeddingData")
    dataRows = len(data)

    # Create dataset and dataloader
    dataset = TextPairDataset(data, vocab)

    # Initialize the model, loss function, and optimizer
    print("Initializing model...")
    model = RNNEmbedder(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for itemIndex in trange(dataRows, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
            query, title, target = dataset[itemIndex]

            optimizer.zero_grad()

            # Get embeddings
            embedding1 = model(query)
            embedding2 = model(title)

            # Compute dot product
            dot_product = torch.sum(embedding1 * embedding2)

            # Compute loss
            loss = criterion(dot_product, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        clearLines(1)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/dataRows:.4f}")

        # Save the trained model
        torch.save(model, modelSavePath)
