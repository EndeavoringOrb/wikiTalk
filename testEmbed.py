import torch
from trainEmbed import vocab

modelSavePath = "models/embed/0.pt"

# Initialize the model, loss function, and optimizer
print("Loading model...")
model = torch.load(modelSavePath)

while True:
    query = input("Enter query: ")
    if query == "":
        break

    title = input("Enter title: ")
    if title == "":
        break

    query_tensor = torch.tensor([vocab[character] for character in query.lower()])
    title_tensor = torch.tensor([vocab[character] for character in title.lower()])

    similarity = model.getSimilarity(query_tensor, title_tensor)

    print(f"Similarity: {similarity.item()}")
