import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tokenizeWiki import loadTokens
from helperFuncs import clearLines


# Define the Encoder
class EncoderRNN(nn.Module):
    def __init__(self, vocabSize, hiddenSize):
        super(EncoderRNN, self).__init__()
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize

        data = torch.normal(0, 0.02, (hiddenSize,))
        self.initState = nn.Parameter(data)
        data = torch.normal(0, 0.02, (vocabSize, hiddenSize))
        self.ih = nn.Parameter(data)
        data = torch.normal(0, 0.02, (hiddenSize, hiddenSize))
        self.hh = nn.Parameter(data)

    def forward(self, state, token):
        out = F.tanh(state @ self.hh + self.ih[token])
        return out

    def init_state(self, batchSize):
        # Initialize hidden state
        return self.initState.unsqueeze(0).repeat(batchSize, 1)


# Define the Decoder
class DecoderRNN(nn.Module):
    def __init__(self, vocabSize, hiddenSize):
        super(DecoderRNN, self).__init__()
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize

        data = torch.normal(0, 0.02, (vocabSize, hiddenSize))
        self.ih = nn.Parameter(data)
        data = torch.normal(0, 0.02, (hiddenSize, hiddenSize))
        self.hh = nn.Parameter(data)
        data = torch.normal(0, 0.02, (hiddenSize, vocabSize))
        self.ho = nn.Parameter(data)

    def forward(self, state, token):
        out = F.tanh(state @ self.hh + self.ih[token])
        return out

    def pred(self, state):
        return state @ self.ho


class Embedder(nn.Module):
    def __init__(self, vocabSize, hiddenSize):
        super(Embedder, self).__init__()
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize

        self.encoder = EncoderRNN(vocabSize, hiddenSize)
        self.decoder = DecoderRNN(vocabSize, hiddenSize)


def main():
    # Settings
    vocabSize = 95
    hiddenSize = 64

    epochs = 10000

    savePath = f"models/autoEncode/model.pt"
    saveInterval = 25

    # Init model
    embedder = Embedder(vocabSize, hiddenSize)
    # embedder: Embedder = torch.load(savePath, weights_only=False)
    print(
        f"Embedder has {sum(param.numel() for param in embedder.parameters()):,} parameters"
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(embedder.parameters())

    # Init trackers
    stepNum = 0
    totalTokensTrained = 0

    for epoch in range(epochs):
        for fileIndex, titleTokens, pageTokens in loadTokens("tokenData/articles"):
            numPageTokens = len(pageTokens)
            print(f"Num Page tokens: {numPageTokens:,}")

            # pageTokens = [0, 1, 2]
            # encode pageTokens
            print(f"Encoding")
            state = embedder.encoder.init_state(1)
            for token in pageTokens:
                state = embedder.encoder(state, token)

            # decode, get loss
            print(f"Decoding")
            loss = 0
            for token in pageTokens:
                pred = embedder.decoder.pred(state)
                loss += criterion(
                    pred, torch.tensor(token, dtype=torch.int64).unsqueeze(0)
                )
                state = embedder.decoder(state, token)

            # backprop
            print(f"Doing backprop")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update trackers
            stepNum += 1
            totalTokensTrained += numPageTokens

            # Clear lines
            clearLines(4 + (stepNum > 1))

            print(
                f"Epoch {epoch:,} - Step {stepNum:,} - # Tokens Trained: {totalTokensTrained:,} - Loss: {loss.item() / numPageTokens}"
            )

            # Save
            if stepNum % saveInterval == 0:
                torch.save(embedder, savePath)
                print()  # So that the info printed for this step will not be cleared


if __name__ == "__main__":
    main()
