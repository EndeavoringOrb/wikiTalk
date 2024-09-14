from AutoEncoderRNNtrain import EncoderRNN, DecoderRNN, Embedder
from tokenizeWiki import tokenize, decode, read_compact_titles, loadPage
import torch
import torch.nn.functional as F

print("Loading model")
loadPath = f"models/autoEncode/model.pt"
embedder: Embedder = torch.load(loadPath, weights_only=False)

# Load all titles
print("Loading all titles")
fileTitles = []
for fileIndex, pageIndex, pageTitleTokens in read_compact_titles(
    "tokenData/titles.bin"
):
    if fileIndex >= len(fileTitles):
        fileTitles.append([])
    fileTitles[fileIndex].append(pageTitleTokens)
print(f"Finished loading {sum([len(item) for item in fileTitles]):,} titles")

while True:
    text = input("\nEnter article name: ")
    queryTokens = tokenize(text)

    foundArticle = False
    titleTokens, pageTokens = None, None

    print("Searching titles")
    for fileIndex, titles in enumerate(fileTitles):
        for pageIndex, titleTokens in enumerate(titles):
            if queryTokens == titleTokens:
                _, tokens = loadPage("tokenData", fileIndex, pageIndex)
                foundArticle = True
                break
        if foundArticle:
            break

    if foundArticle:
        print(f'Found article: "{text}"')
        print(decode(tokens))
    else:
        print(f'No article found for "{text}"')
        continue

    print("Embedding")
    state = embedder.encoder.init_state(1)
    for token in tokens:
        state = embedder.encoder(state, token)

    decodedTokens = []

    print("Decoding")
    for i in range(len(tokens)):
        pred = embedder.decoder.pred(state)
        probs = F.softmax(pred, dim=-1)
        token = torch.multinomial(probs, 1).item()
        decodedTokens.append(token)
        state = embedder.decoder(state, token)

    print(f"\n\n\nDecoded:")
    print(decode(decodedTokens))
