import torch
from model import RecurrentTransformer
from tokenizeWiki import *
import bisect

from line_profiler import profile
import os

os.environ["LINE_PROFILE"] = "1"


@profile
def serializeVectors(data, filename):
    with open(filename, "wb") as f:
        # Write the number of pairs
        f.write(struct.pack("I", len(data)))

        # Write the length of each tensor (assuming all have the same length)
        if data:
            f.write(struct.pack("I", len(data[0])))

        # Write each tensor
        for tensor in data:
            # Write the tensor data
            f.write(tensor.cpu().numpy().tobytes())


@torch.no_grad()
@profile
def deserializeVectors(filename):
    with open(filename, "rb") as f:
        # Read the number of pairs
        num_pairs = struct.unpack("I", f.read(4))[0]

        # Read the length of each tensor
        tensor_length = struct.unpack("I", f.read(4))[0]

        read_size = tensor_length * 4

        # Read each tensor
        for _ in range(num_pairs):
            # Read the tensor data
            tensor_data = f.read(read_size)  # 4 bytes per float32
            tensor = torch.frombuffer(tensor_data, dtype=torch.float32)

            yield tensor


@profile
def loadVectors(folder):
    files = sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0]))
    for fileIndex, file in enumerate(files):
        for pageIndex, vector in enumerate(deserializeVectors(f"{folder}/{file}")):
            yield fileIndex, pageIndex, vector


@torch.no_grad()
@profile
def embedWiki(
    model: RecurrentTransformer,
    dataFolder: str,
    outputFolder: str,
    numPages,
):
    totalNumPages = sum(numPages)

    fileNum = 0
    f = open(f"{outputFolder}/{fileNum}.vectors", "wb")
    # Write the number of pairs
    f.write(struct.pack("I", numPages[fileNum]))
    # Write the length of each tensor
    f.write(struct.pack("I", model.hiddenSize))

    embedding: torch.Tensor = torch.zeros(model.hiddenSize)

    for fileIndex, titleTokens, textTokens in tqdm(
        loadTokens(dataFolder), desc="Embedding Wiki Pages", total=totalNumPages
    ):
        if fileIndex != fileNum:
            f.close()
            fileNum += 1
            f = open(f"{outputFolder}/{fileNum}.emb", "wb")
            # Write the number of pairs
            f.write(struct.pack("I", numPages[fileNum]))
            # Write the length of each tensor
            f.write(struct.pack("I", model.hiddenSize))

        embedding = model.initState.unsqueeze(0)
        for token in torch.tensor(textTokens, dtype=torch.int64):
            embedding = model.nextState(embedding, token.unsqueeze(0))

        # Normalize state
        invNorm = 1 / torch.norm(embedding)
        embedding *= invNorm

        # Write the tensor data
        f.write(embedding.cpu().numpy().tobytes())


@torch.no_grad()
@profile
def searchLoadedVectorWiki(
    query: str, model: RecurrentTransformer, dataFolder: str, wiki: dict
):
    print(f"Initializing search...")
    # Tokenize query
    queryTokens = tokenize(query)

    # Get text embedding by passing text through model
    queryEmbedding = model.fastEmbedTitle(queryTokens)

    # Normalize queryState
    invNorm = 1 / torch.norm(queryEmbedding)
    queryEmbedding *= invNorm

    # Init results
    results = []  # [(similarity, pageIndex)]

    print(f"Searching...")
    # Search
    for fileIndex, pages in wiki.items():
        for i in range(len(pages)):
            # Get embedding
            embedding = pages[i]

            # Compute dot product
            dot_product = torch.sum(queryEmbedding * embedding).item()

            # Insert result at correct position in results
            # TODO: iterate the other way, then you can stop if any are false
            inserted = False
            for i, result in enumerate(results):
                if dot_product > result[0]:
                    results.insert(i, (dot_product, fileIndex, i))
                    inserted = True
                    if len(results) > numResults:
                        results.pop()
                    break
            if not inserted and len(results) < numResults:
                results.append((dot_product, fileIndex, i))

    # Get result pages
    print(f"Loading top {numResults} pages...")
    resultPages = [(result[1], result[2]) for result in results]
    pages = []
    for i, (titleTokens, textTokens) in enumerate(
        loadTokensIndices(dataFolder, resultPages)
    ):
        title = decode(titleTokens)
        text = decode(textTokens)
        print(f"{i}: {title}")
        pages.append((title, text))

    print(f"Finished")
    return pages


def loadWiki(vectorDataFolder):
    wiki = {}
    currentFileIndex = -1

    for fileIndex, pageIndex, embedding in loadVectors(vectorDataFolder):
        if fileIndex != currentFileIndex:
            currentFileIndex = fileIndex
            wiki[fileIndex] = []

        wiki[fileIndex].append(embedding)

    return wiki


if __name__ == "__main__":
    # Initialize stuff
    print("Loading model...")
    model: RecurrentTransformer = torch.load(
        "models/tokenPredArticle/current/model.pt", weights_only=False
    )
    model.preCompute()

    dataFolder = "tokenData"
    vectorDataFolder = "embedWiki"

    numResults = 5

    numPages = countNumPages(dataFolder)
    totalNumPages = sum(numPages)

    affirmative = input("Embed wiki? [y/n]: ") == "y"
    if affirmative:
        embedWiki(model, dataFolder, vectorDataFolder, numPages)

    print("Loading wiki...")
    wiki = loadWiki(vectorDataFolder)

    while True:
        query = input("Enter query: ")
        if query == "":
            print("exiting...")
            break

        pages = searchLoadedVectorWiki(query, model, dataFolder, wiki)
