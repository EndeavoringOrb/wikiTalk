import torch
from trainEmbedArticle import RNNLanguage
from tokenizeWiki import *
import bisect

from line_profiler import profile
import os

os.environ["LINE_PROFILE"] = "1"


@profile
def serializeVectors(data, filename):
    with open(filename, "wb") as f:
        # Write the number of pairs
        f.write(struct.pack("!I", len(data)))

        # Write the length of each tensor (assuming all have the same length)
        if data:
            f.write(struct.pack("!I", len(data[0])))

        # Write each tensor
        for tensor in data:
            # Write the tensor data
            f.write(tensor.cpu().numpy().tobytes())


@torch.no_grad()
@profile
def deserializeVectors(filename):
    with open(filename, "rb") as f:
        # Read the number of pairs
        num_pairs = struct.unpack("!I", f.read(4))[0]

        # Read the length of each tensor
        tensor_length = struct.unpack("!I", f.read(4))[0]

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
    model: RNNLanguage,
    dataFolder: str,
    outputFolder: str,
    numPages,
):
    totalNumPages = sum(numPages)

    fileNum = 0
    f = open(f"{outputFolder}/{fileNum}.vectors", "wb")
    # Write the number of pairs
    f.write(struct.pack("!I", numPages[fileNum]))
    # Write the length of each tensor
    f.write(struct.pack("!I", model.hiddenDim))

    titleState: torch.Tensor = torch.zeros(model.hiddenDim)

    for fileIndex, pageIndex, titleTokens in tqdm(
        loadTitles(dataFolder), desc="Embedding Wiki Pages", total=totalNumPages
    ):
        if fileIndex != fileNum:
            f.close()
            fileNum += 1
            f = open(f"{outputFolder}/{fileNum}.vectors", "wb")
            # Write the number of pairs
            f.write(struct.pack("!I", numPages[fileNum]))
            # Write the length of each tensor
            f.write(struct.pack("!I", model.hiddenDim))

        titleState.fill_(0)
        for token in titleTokens:
            titleState = model.fastForward(titleState, token)
        titleState = model.fastForward(titleState, sepToken)

        # Normalize titleState
        invNorm = 1 / torch.norm(titleState)
        titleState *= invNorm

        # Write the tensor data
        f.write(titleState.cpu().numpy().tobytes())


@torch.no_grad()
@profile
def searchVectorWiki(
    query: str,
    model: RNNLanguage,
    dataFolder: str,
    vectorDataFolder: str,
    totalNumPages=None,
):
    print(f"Initializing search...")
    # Tokenize query
    queryTokens = tokenize(query) + [sepToken]

    # Get text embedding by passing text through model
    queryState = torch.zeros(model.hiddenDim)
    for token in queryTokens:
        queryState = model.fastForward(queryState, token)

    # Normalize queryState
    invNorm = 1 / torch.norm(queryState)
    queryState *= invNorm

    # Init results
    results = []  # [(similarity, pageIndex)]

    print(f"Searching...")
    # Search
    for fileIndex, pageIndex, titleState in loadVectors(vectorDataFolder):
        # Compute dot product
        dot_product = torch.sum(queryState * titleState).item()

        # Insert result at correct position in results
        # TODO: iterate the other way, then you can stop if any are false
        inserted = False
        for i, result in enumerate(results):
            if dot_product > result[0]:
                results.insert(i, (dot_product, fileIndex, pageIndex))
                inserted = True
                if len(results) > numResults:
                    results.pop()
                break
        if not inserted and len(results) < numResults:
            results.append((dot_product, fileIndex, pageIndex))

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


@profile
def searchWiki(query: str, model: RNNLanguage, dataFolder: str, totalNumPages=None):
    # Tokenize query
    queryTokens = tokenize(query) + [sepToken]

    with torch.no_grad():
        # Get text embedding by passing text through model
        queryState = torch.zeros(model.hiddenDim)
        for token in tqdm(queryTokens, desc="Getting Embedding"):
            queryState = model.fastForward(queryState, token)

        # Normalize queryState
        invNorm = 1 / torch.norm(queryState)
        queryState *= invNorm

    # Init results
    results = []  # [(similarity, pageIndex)]

    # Search
    with torch.no_grad():
        titleState: torch.Tensor = torch.zeros(model.hiddenDim)

    for fileIndex, pageIndex, titleTokens in tqdm(
        loadTitles(dataFolder), desc="Searching Wiki Pages", total=totalNumPages
    ):
        with torch.no_grad():
            titleState.fill_(0)
            for token in titleTokens:
                titleState = model.fastForward(titleState, token)
            titleState = model.fastForward(titleState, sepToken)

            # Normalize titleState
            invNorm = 1 / torch.norm(titleState)
            titleState *= invNorm

            # Compute dot product
            dot_product = torch.sum(queryState * titleState).item()

        # Insert result at correct position in results
        inserted = False
        for i, result in enumerate(results):
            if dot_product > result[0]:
                results.insert(
                    i, (dot_product, fileIndex, pageIndex, decode(titleTokens))
                )
                inserted = True
                if len(results) > numResults:
                    results.pop()
                break
        if not inserted and len(results) < numResults:
            results.append((dot_product, fileIndex, pageIndex, decode(titleTokens)))

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
        pages.append(title, text)

    print(f"Finished")
    return pages


if __name__ == "__main__":
    # Initialize stuff
    print("Loading model...")
    model: RNNLanguage = torch.load("models/embedArticle/1.pt", weights_only=False)
    model.preCompute()

    sepToken = len(charToToken)

    dataFolder = "tokenData"
    vectorDataFolder = "embedWiki"

    numResults = 5

    numPages = countNumPages(dataFolder)
    totalNumPages = sum(numPages)

    affirmative = input("Embed wiki? [y/n]: ") == "y"
    if affirmative:
        embedWiki(model, dataFolder, vectorDataFolder, numPages)

    while True:
        query = input("Enter query: ")
        if query == "":
            print("exiting...")
            break

        pages = searchVectorWiki(
            query, model, dataFolder, vectorDataFolder, totalNumPages
        )
        exit(0)
