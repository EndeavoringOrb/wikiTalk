import torch.nn as nn
import torch
import torch.nn.functional as F

from line_profiler import profile
import os

os.environ["LINE_PROFILE"] = "0"

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

        self.activation = nn.Tanh()

    def preCompute(self, tok_emb):
        self.kPreComputed = self.key(tok_emb)
        self.kPreComputed = self.activation(self.kPreComputed)
        self.kPreComputed = self.kPreComputed.transpose(-2, -1) * self.kPreComputed.shape[-1] ** -0.5

        self.vPreComputed = self.value(tok_emb)
        self.vPreComputed = self.activation(self.vPreComputed)

    #@profile
    def forwardPreComputed(self, x, tokens):
        # x has size (batch, channels)
        # output of size (batch, head size)
        k = self.kPreComputed[tokens]

        q = self.query(x)  # (B,channels,hs)
        q = self.activation(q)

        # compute attention scores ("affinities")
        wei = (
            q @ k
        )  # (B, C, hs) @ (B, hs, 1) -> (B, C, 1)
        wei = F.softmax(wei, dim=-2)  # (B, C, 1)
        # wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.vPreComputed[tokens]

        out = wei @ v  # (B, C, 1) @ (B, 1, hs) -> (B, C, hs)
        out = self.activation(out)

        out = self.proj(out).squeeze()  # (B, C)

        return out

    #@profile
    def forward(self, x, tok_emb):
        # x has size (batch, channels)
        # tok_emb has size (batch, channels)
        # output of size (batch, head size)
        k = self.key(tok_emb)  # (B,1,hs)
        k = self.activation(k)

        q = self.query(x)  # (B,channels,hs)
        q = self.activation(q)

        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, C, hs) @ (B, hs, 1) -> (B, C, 1)
        wei = F.softmax(wei, dim=-2)  # (B, C, 1)
        # wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(tok_emb)  # (B,1,hs)
        v = self.activation(v)

        out = wei @ v  # (B, C, 1) @ (B, 1, hs) -> (B, C, hs)
        out = self.activation(out)

        out = self.proj(out).squeeze()  # (B, C)

        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, embdSize, nHead, headSize, device):
        super().__init__()
        self.heads = nn.ModuleList([Head(embdSize, headSize, device) for _ in range(nHead)])
        self.proj = nn.Linear(embdSize * nHead, embdSize)

    #@profile
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
            nn.Tanh(),
            nn.Linear(4 * embdSize, embdSize),
        )

    #@profile
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
    
    def preCompute(self, embeddingTable):
        self.ln2_embed = self.ln2(embeddingTable)

    #@profile
    def forwardPreComputed(self, x):
        state, tokens = x
        state = state + self.sa(self.ln1(state), self.ln2_embed[tokens])
        state = state + self.ffwd(self.ln3(state))
        return (state, tokens)

    #@profile
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
        data = torch.normal(0, 0.02, (vocabSize, embdSize))
        self.token_embedding_table = nn.Parameter(data)

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
        tok_emb = self.token_embedding_table[idx]  # (B,C)
        state, tok_emb = self.blocks((state, tok_emb))  # (B,C)
        state = self.ln_f(state)  # (B,C), this is the new state
        logits = self.lm_head(state)  # (B,vocab_size)

        return state, logits

    #@profile
    def nextState(self, state, idx):
        tok_emb = self.token_embedding_table[idx]  # (B,C)
        state, tok_emb = self.blocks((state, tok_emb))  # (B,C)
        state = self.ln_f(state)  # (B,C), this is the new state
        return state

    def getPreds(self, state):
        return self.lm_head(state)

    def preCompute(self):
        pass
        #for block in self.blocks:
        #    block.preCompute()

    #@profile
    def train(self, state, tokens, criterion):
        loss = 0
        numSteps = len(tokens[0])
        for i in range(numSteps):
            nextToken = tokens[:, i]
            pred = self.getPreds(state)
            loss += criterion(pred, nextToken)
            state = self.nextState(state, nextToken)
        return state, loss
