import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sympy as sp


class RNN(nn.Module):
    def __init__(self, vocabSize, hiddenSize):
        super(RNN, self).__init__()

        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize

        data = torch.normal(0, 0.02, (hiddenSize,))
        self.initHidden = nn.Parameter(data)

        data = torch.normal(0, 0.02, (vocabSize, hiddenSize))
        self.i2h = nn.Parameter(data)
        data = torch.normal(0, 0.02, (hiddenSize,))
        self.i2hb = nn.Parameter(data)

        self.h2h0 = nn.Linear(hiddenSize, hiddenSize)
        self.h2h1 = nn.Linear(hiddenSize, hiddenSize)

        self.h2o = nn.Linear(hiddenSize, vocabSize)

    def predToken(self, hidden):
        output = self.h2o(hidden)
        return output

    def nextState(self, hidden, token):
        out = F.tanh(self.i2h[token] + self.i2hb + self.h2h0(hidden))
        return F.tanh(self.h2h1(out))


def clearLines(numLines):
    for _ in range(numLines):
        print("\033[F\033[K", end="")

sequences = ["AC", "BC"]
sequences = ["ABABABABABABABABABABABABABABABABABABABABABABABAB"]
hiddenSize = 1
numIterations = 100000
learningRate = 0.01

"""
"CAR", "HAT"
vocabSize: 5
hiddenSize: 2
loss: ~1.38

"CAR", "HAT"
vocabSize: 5
hiddenSize: 1
loss: ~1.38

"CAR"
vocabSize: 3
hiddenSize: 2
loss: ~0

"ABAAB"
vocabSize: 2
hiddenSize: 2
loss: ~0

"ABAAB"
vocabSize: 2
hiddenSize: 1
loss: ~0

"ABACBAA"
vocabSize: 3
hiddenSize: 1
loss: ~0

"""

vocab = set()
for seq in sequences:
    vocab.update(seq)
chars = sorted(list(vocab))
charToIdx = {char: idx for idx, char in enumerate(chars)}
idxToChar = {idx: char for char, idx in charToIdx.items()}
encode = lambda x: [charToIdx[char] for char in x]
decode = lambda x: "".join([idxToChar[idx] for idx in x])

tokenSequences = [torch.tensor(encode(seq), dtype=torch.int64) for seq in sequences]

vocabSize = len(chars)

model = RNN(vocabSize, hiddenSize)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

print(f"Vocab Size: {vocabSize}")
print(f"Hidden Dim: {hiddenSize}")

print("\n" * 10)

for iteration in range(numIterations):
    optimizer.zero_grad()

    loss = 0
    for seq in tokenSequences:
        state = model.initHidden
        for idx, token in enumerate(seq):
            pred = model.predToken(state)
            loss += criterion(pred, token)
            state = model.nextState(state, token)

    loss.backward()
    optimizer.step()

    clearLines(10)

    print(f"{iteration}: {loss}")
    print(f"initHidden grad norm: {model.initHidden.grad.norm().item()}")

    print(f"i2h grad norm: {model.i2h.grad.norm().item()}")
    print(f"i2h bias grad norm: {model.i2hb.grad.norm().item()}")

    print(f"h2h0 grad norm: {model.h2h0.weight.grad.norm().item()}")
    print(f"h2h0 bias grad norm: {model.h2h0.bias.grad.norm().item()}")

    print(f"h2h1 grad norm: {model.h2h1.weight.grad.norm().item()}")
    print(f"h2h1 bias grad norm: {model.h2h1.bias.grad.norm().item()}")

    print(f"h2o grad norm: {model.h2o.weight.grad.norm().item()}")
    print(f"h2o bias grad norm: {model.h2o.bias.grad.norm().item()}")
