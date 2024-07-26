import torch.nn as nn
import torch

# Define the RNN model
class RNNLanguage(nn.Module):
    def __init__(self, vocabSize, embeddingDim, hiddenDim):
        super(RNNLanguage, self).__init__()
        self.embedding = nn.Embedding(vocabSize, embeddingDim)

        self.ih = nn.Linear(embeddingDim, hiddenDim)
        self.hh = nn.Linear(hiddenDim, hiddenDim)

        self.fc = nn.Linear(hiddenDim, vocabSize)

        self.activation = nn.Tanh()

        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        self.hiddenDim = hiddenDim

    def forward(self, state, x):
        embedded = self.embedding(x)
        newState = self.activation(self.ih(embedded) + self.hh(state))
        return newState
    
    def logits(self, state):
        return self.fc(state)


# Define the RNN model
class RNNEmbedder(nn.Module):
    def __init__(self, vocabSize, embeddingDim, hiddenDim):
        super(RNNEmbedder, self).__init__()
        self.embedding = nn.Embedding(vocabSize, embeddingDim)

        self.ih = nn.Linear(embeddingDim, hiddenDim)
        self.hh = nn.Linear(hiddenDim, hiddenDim)

        self.fc = nn.Linear(hiddenDim, embeddingDim)

        self.activation = nn.Tanh()

        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        self.hiddenDim = hiddenDim

    def forward(self, x):
        embedded = self.embedding(x)
        hiddenState = torch.zeros(self.hiddenDim)
        for embedding in embedded:
            hiddenState = self.activation(self.ih(embedding) + self.hh(hiddenState))
        return self.fc(hiddenState)
    
    @torch.no_grad()
    def getSimilarity(self, query, title):
        # Get embeddings
        embedding1 = self.forward(query)
        embedding2 = self.forward(title)

        # Compute dot product
        dot_product = torch.sum(embedding1 * embedding2)

        # Normalize
        length1 = torch.norm(embedding1)
        length2 = torch.norm(embedding2)

        dot_product /= length1 * length2

        return dot_product