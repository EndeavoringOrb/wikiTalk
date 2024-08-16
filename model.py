import torch.nn as nn
import torch
import torch.nn.functional as F

from line_profiler import profile
import os
os.environ["LINE_PROFILE"] = "0"


class CustomActivation(torch.autograd.Function):
    @profile
    def fastForward(x):
        xAbs = torch.abs(x)
        xAbs = 1 - (1 / (xAbs.square() + xAbs + 1))  # 1 - (1 / (x^2 + x + 1))
        return xAbs.copysign(x)

    @staticmethod
    @profile
    def forward(ctx, x):
        xAbs = torch.abs(x)

        term1 = xAbs + 1  # x + 1
        term2 = 1 / (xAbs.square() + term1)  # 1 / (x^2 + x + 1)

        xAbs = (xAbs + term1) * term2 * term2  # (2x + 1) / ((x^2 + x + 1)^2)
        ctx.save_for_backward(xAbs)

        xAbs = 1 - term2  # 1 - (1 / (x^2 + x + 1))

        return xAbs.copysign(x)

    @staticmethod
    @profile
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad * grad_output


class CustomActivationModule(nn.Module):
    def forward(self, x):
        return CustomActivation.apply(x)

    def fastForward(self, x):
        return CustomActivation.fastForward(x)


# Define the RNN model
class RNNLanguageOLD(nn.Module):
    def __init__(self, vocabSize, hiddenDim):
        super(RNNLanguage, self).__init__()
        data = torch.normal(0, 0.02, (vocabSize, hiddenDim))
        self.embedding = nn.Parameter(data)

        data = torch.normal(0, 0.02, (hiddenDim, hiddenDim))
        self.ih = nn.Parameter(data)
        data = torch.normal(0, 0.02, (hiddenDim, hiddenDim))
        self.hh = nn.Parameter(data)

        data = torch.normal(0, 0.02, (hiddenDim,))
        self.bias = nn.Parameter(data)

        self.activation = nn.Tanh()
        self.activation = CustomActivationModule()

        self.vocabSize = vocabSize
        self.hiddenDim = hiddenDim

    @profile
    def preCompute(self):
        # hh
        self.scaledHH = self.hh / (torch.norm(self.hh, dim=1) * self.hiddenDim)

        # embedding
        embedded = self.activation(self.embedding)
        self.embedded = embedded @ self.ih + self.bias

    @profile
    # Assumes model.preCompute() has been called after any previous parameter updates
    def forwardEmbedded(self, state, x):
        newState = self.activation(self.embedded[x] + state @ self.scaledHH)
        return newState

    @profile
    # Assumes model.preCompute() has been called after any previous parameter updates
    def forwardEmbeddedFast(self, state, x):
        newState = self.activation.fastForward(self.embedded[x] + state @ self.scaledHH)
        return newState

    @profile
    def logits(self, state):
        return (
            state @ self.embedding.T
        )  # we re-use the embedding matrix to save on param count

    def preprocess(self, x):
        state = torch.zeros(self.hiddenDim)
        for token in x:
            state = self.forward(state, token)
        return state

    def sample(self, state):
        logits = self.logits(state)
        probs = F.softmax(logits, dim=0)
        token = torch.multinomial(probs, 1)
        return token


# Define the RNN model
class RNNEmbedder(nn.Module):
    def __init__(self, vocabSize, hiddenDim):
        super(RNNEmbedder, self).__init__()
        self.titleModel = RNNLanguage(vocabSize, hiddenDim)
        self.textModel = RNNLanguage(vocabSize, hiddenDim)

        self.vocabSize = vocabSize
        self.hiddenDim = hiddenDim

    def preCompute(self):
        self.titleModel.preCompute()
        self.textModel.preCompute()

    @torch.no_grad
    @profile
    def fastEmbedTitle(self, tokens):
        state = torch.zeros(self.hiddenDim, device=self.titleModel.embedding.device)
        for token in tokens:
            self.titleModel.forwardEmbeddedFast(state, token)
        return state

    @torch.no_grad
    @profile
    def fastEmbedArticle(self, tokens):
        state = torch.zeros(self.hiddenDim, device=self.textModel.embedding.device)
        for token in tokens:
            self.textModel.forwardEmbeddedFast(state, token)
        return state


class RNNEmbedderNEW(nn.Module):
    def __init__(self, vocabSize, hiddenSize, outSize):
        super(RNNEmbedderNEW, self).__init__()
        data = torch.normal(0, 0.02, (vocabSize, hiddenSize))
        self.embedding = nn.Parameter(data)

        data = torch.normal(0, 0.02, (hiddenSize, hiddenSize))
        self.ih = nn.Parameter(data)
        data = torch.normal(0, 0.02, (hiddenSize, hiddenSize))
        self.hh = nn.Parameter(data)

        data = torch.normal(0, 0.02, (hiddenSize,))
        self.bias = nn.Parameter(data)

        data = torch.normal(0, 0.02, (hiddenSize, outSize))
        self.out = nn.Parameter(data)

        self.activation = nn.Tanh()
        self.activation = CustomActivationModule()

        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.outSize = outSize

    @profile
    def preCompute(self):
        # hh
        self.scaledHH = self.hh / (torch.norm(self.hh, dim=1) * self.hiddenSize)

        # embedding
        embedded = self.activation(self.embedding)
        self.embedded = embedded @ self.ih + self.bias
    
    @profile
    # Assumes model.preCompute() has been called after any previous parameter updates
    def forwardEmbeddedNoBatch(self, state, x):
        attention = torch.einsum('i,j->i', (state, self.embedded[x]))
        attention = F.softmax(attention, dim=-1)
        state = state + attention @ self.scaledHH
        state = F.tanh(state)
        return state.squeeze()

    @profile
    # Assumes model.preCompute() has been called after any previous parameter updates
    def forwardEmbedded(self, state, x):
        attention = torch.einsum('bi,bj->bi', (state, self.embedded[x]))
        attention = F.softmax(attention, dim=-1)
        state = state + attention @ self.scaledHH
        state = F.tanh(state)
        return state

    @profile
    def getOut(self, state):
        return state @ self.out

# Define the RNN model
class RNNSimilarityEmbedder(nn.Module):
    def __init__(self, vocabSize, hiddenSize, embeddingSize):
        super(RNNSimilarityEmbedder, self).__init__()
        self.titleModel = RNNEmbedderNEW(vocabSize, hiddenSize, embeddingSize)
        self.textModel = RNNEmbedderNEW(vocabSize, hiddenSize, embeddingSize)

        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.embeddingSize = embeddingSize

    def preCompute(self):
        self.titleModel.preCompute()
        self.textModel.preCompute()

class RNNLanguage(nn.Module):
    def __init__(self, vocabSize, hiddenSize, outSize):
        super(RNNLanguage, self).__init__()
        data = torch.normal(0, 0.02, (vocabSize, hiddenSize))
        self.embedding = nn.Parameter(data)

        data = torch.normal(0, 0.02, (hiddenSize, hiddenSize))
        self.ih = nn.Parameter(data)
        data = torch.normal(0, 0.02, (hiddenSize, hiddenSize))
        self.hh = nn.Parameter(data)

        data = torch.normal(0, 0.02, (hiddenSize,))
        self.bias = nn.Parameter(data)

        data = torch.normal(0, 0.02, (hiddenSize, outSize))
        self.out = nn.Parameter(data)

        data = torch.normal(0, 0.02, (outSize,))
        self.outBias = nn.Parameter(data)

        data = torch.normal(0, 0.02, (hiddenSize,))
        self.initState = nn.Parameter(data)

        self.activation = nn.Tanh()
        self.activation = CustomActivationModule()

        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.outSize = outSize

    @profile
    def preCompute(self):
        # hh
        self.scaledHH = self.hh / (torch.norm(self.hh, dim=1) * self.hiddenSize)

        # embedding
        embedded = self.activation(self.embedding)
        self.embedded = embedded @ self.ih + self.bias
    
    @profile
    # Assumes model.preCompute() has been called after any previous parameter updates
    def forwardEmbeddedNoBatch(self, state, x):
        attention = torch.einsum('i,j->i', (state, self.embedded[x]))
        attention = F.softmax(attention, dim=-1)
        state = state + attention @ self.scaledHH
        state = F.tanh(state)
        return state.squeeze()

    @profile
    # Assumes model.preCompute() has been called after any previous parameter updates
    def forwardEmbedded(self, state, x):
        attention = torch.einsum('bi,bj->bi', (state, self.embedded[x]))
        attention = F.softmax(attention, dim=-1)
        state = state + attention @ self.scaledHH
        state = F.tanh(state)
        return state

    @profile
    def getOut(self, state):
        return state @ self.out + self.outBias

    @profile
    def train(self, state, tokens, criterion):
        loss = 0
        numSteps = len(tokens[0])
        for i in range(numSteps):
            nextToken = tokens[:, i]
            pred = self.getOut(state)
            loss += criterion(pred, nextToken)
            state = self.forwardEmbedded(state, nextToken)
        return state, loss