import torch.nn as nn
import torch
import torch.nn.functional as F

from line_profiler import profile
import os

os.environ["LINE_PROFILE"] = "0"


class CustomActivation(torch.autograd.Function):
    @profile
    def fastForward(x):
        xSign = torch.sign(x)
        x = xSign * x  # x = abs(x)
        x += x * x  # x = x^2 + x
        output = x / (x + 1)  # (x^2 + x) / (x^2 + x + 1)
        return output * xSign  # -output if x < 0 else output

    @staticmethod
    @profile
    def forward(ctx, x):
        xAbs = torch.abs(x)

        term1 = xAbs + 1 # x + 1
        term2 = 1 / (xAbs.square() + term1) # 1 / (x^2 + x + 1)

        xAbs = (xAbs + term1) * term2 * term2 # (2x + 1) / ((x^2 + x + 1)^2)
        ctx.save_for_backward(xAbs)

        xAbs = 1 - term2 # 1 - (1 / (x^2 + x + 1))

        return xAbs.copysign(x)


        """xSign = torch.sign(x)
        x = xSign * x  # x = abs(x)
        term1 = x + 1
        term2 = x * x + term1  # x^2 + x + 1
        grad = (x + term1) / (term2 * term2)  # (2x + 1) / ((x^2 + x + 1)^2)
        ctx.save_for_backward(grad)
        output = (term2 - 1) / (term2)  # (x^2 + x) / (x^2 + x + 1)
        return output * xSign  # -output if x < 0 else output"""

    @staticmethod
    @profile
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        # term1 = x + 1
        # term2 = x * x + term1
        # grad = (x + term1) / (term2 * term2)  # (2x + 1) / ((x^2 + x + 1)^2)
        return grad * grad_output


class CustomActivationModule(nn.Module):
    def forward(self, x):
        return CustomActivation.apply(x)

    def fastForward(self, x):
        return CustomActivation.fastForward(x)


# Define the RNN model
class RNNLanguage(nn.Module):
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
    def fastForward(self, state, x):
        embedded = self.embedding[x]
        embedded = self.activation.fastForward(embedded)

        newState = embedded @ self.ih
        newState += state @ self.scaledHH
        newState += self.bias
        newState = self.activation.fastForward(newState)

        return newState

    @profile
    def forward(self, state, x):
        embedded = self.embedding[x]
        embedded = self.activation(embedded)

        newState = embedded @ self.ih
        newState += state @ self.scaledHH
        newState += self.bias
        newState = self.activation(newState)

        return newState

    @profile
    # Assumes model.preCompute() has been called after any previous parameter updates
    def forwardEmbedded(self, state, x):
        newState = self.activation(self.embedded[x] + state @ self.scaledHH)
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
