import torch.nn as nn
import torch
import torch.nn.functional as F

from line_profiler import profile
import os

os.environ["LINE_PROFILE"] = "0"


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
        attention = torch.einsum("i,j->i", (state, self.embedded[x]))
        attention = F.softmax(attention, dim=-1)
        state = state + attention @ self.scaledHH
        state = self.activation(state)
        return state.squeeze()

    @profile
    # Assumes model.preCompute() has been called after any previous parameter updates
    def forwardEmbedded(self, state, x):
        attention = torch.einsum("bi,bj->bi", (state, self.embedded[x]))
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


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, embdSize, headSize, device):
        super().__init__()
        self.key = nn.Linear(embdSize, headSize, bias=True)
        self.query = nn.Linear(1, headSize, bias=True)
        self.positions = torch.arange(0, embdSize).unsqueeze(dim=0).to(device)
        self.query_pos = nn.Embedding(embdSize, headSize)
        self.value = nn.Linear(embdSize, headSize, bias=False)

        self.proj = nn.Linear(headSize, 1)

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x, tok_emb):
        # x has size (batch, channels)
        # tok_emb has size (batch, channels)
        # output of size (batch, head size)
        k = self.key(tok_emb)  # (B,1,hs)

        q = self.query(x)  # (B,channels,hs)
        q2 = self.query_pos(self.positions)
        q = q + q2  # add the position embedding to the query

        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, C, hs) @ (B, hs, 1) -> (B, C, 1)
        wei = F.softmax(wei, dim=-2)  # (B, C, 1)
        # wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(tok_emb)  # (B,1,hs)
        out = wei @ v  # (B, C, 1) @ (B, 1, hs) -> (B, C, hs)
        out = self.proj(out).squeeze()  # (B, C)

        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, embdSize, nHead, headSize, device):
        super().__init__()
        self.heads = nn.ModuleList([Head(embdSize, headSize, device) for _ in range(nHead)])
        self.proj = nn.Linear(embdSize * nHead, embdSize)

    def forward(self, x, tok_emb):
        x = x.unsqueeze(dim=-1)
        out = torch.cat([h(x, tok_emb) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, embdSize):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embdSize, 4 * embdSize),
            nn.ReLU(),
            nn.Linear(4 * embdSize, embdSize),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, embdSize, nHead, headSize, device):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # head_size = n_embd // n_head
        self.sa = MultiHeadAttention(embdSize, nHead, headSize, device)
        self.ffwd = FeedFoward(embdSize)
        self.ln1 = nn.LayerNorm(embdSize)
        self.ln2 = nn.LayerNorm(embdSize)
        self.ln3 = nn.LayerNorm(embdSize)

    def forward(self, x):
        state, tok_emb = x
        state = state + self.sa(self.ln1(state), self.ln2(tok_emb))
        state = state + self.ffwd(self.ln3(state))
        return (state, tok_emb)


class RecurrentTransformer(nn.Module):
    def __init__(self, vocabSize, embdSize, nHead, headSize, nLayer, device):
        super().__init__()
        self.vocabSize = vocabSize
        self.hiddenSize = embdSize
        
        # each token directly reads off the logits for the next token from a lookup table, embeddings are not ternary
        self.token_embedding_table = nn.Embedding(vocabSize, embdSize)
        self.token_position_embedding_table = nn.Embedding(embdSize, embdSize)
        self.blocks = nn.Sequential(
            *[Block(embdSize, nHead, headSize, device) for _ in range(nLayer)]
        )
        self.ln_f = nn.LayerNorm(embdSize)  # final layer norm
        self.lm_head = nn.Linear(embdSize, vocabSize)  # lm_head is not ternary

        data = torch.normal(0, 0.02, (embdSize,))
        self.initState = nn.Parameter(data)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def set_heads_device(self, device):
        for i in range(len(self.blocks)):
            for j in range(len(self.blocks[i].sa.heads)):
                self.blocks[i].sa.heads[j].positions = (
                    self.blocks[i].sa.heads[j].positions.to(device)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, state, idx):
        tok_emb = self.token_embedding_table(idx)  # (B,C)
        state, tok_emb = self.blocks((state, tok_emb))  # (B,C)
        state = self.ln_f(state)  # (B,C), this is the new state
        logits = self.lm_head(state)  # (B,vocab_size)

        return state, logits

    def nextState(self, state, idx):
        tok_emb = self.token_embedding_table(idx)  # (B,C)
        state, tok_emb = self.blocks((state, tok_emb))  # (B,C)
        state = self.ln_f(state)  # (B,C), this is the new state
        return state

    def getPreds(self, state):
        return self.lm_head(state)

    def preCompute(self):
        pass

    @profile
    def train(self, state, tokens, criterion):
        loss = 0
        numSteps = len(tokens[0])
        for i in range(numSteps):
            nextToken = tokens[:, i]
            pred = self.getPreds(state)
            loss += criterion(pred, nextToken)
            state = self.nextState(state, nextToken)
        return state, loss
