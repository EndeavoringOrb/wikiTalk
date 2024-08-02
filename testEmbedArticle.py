import torch
from trainEmbedArticle import RNNLanguage
from tokenizeWiki import *

from line_profiler import profile
import os
os.environ["LINE_PROFILE"] = "1"


@profile
def searchWiki(query: str, model: RNNLanguage, dataFolder: str, totalNumPages = None):
    # Tokenize query
    queryTokens = tokenize(query) + [sepToken]

    with torch.no_grad():
        # Get text embedding by passing text through model
        queryState = torch.zeros(model.hiddenDim)
        for token in tqdm(queryTokens, desc="Getting Embedding"):
            queryState = model.fastForward(queryState, token)

        # Normalize queryState
        queryState /= torch.norm(queryState)

    # Init results
    results = []  # [(similarity, pageIndex)]

    # Search
    with torch.no_grad():
        titleState: torch.Tensor = torch.zeros(model.hiddenDim)

    for fileIndex, pageIndex, titleTokens in tqdm(
        loadTitles(dataFolder), desc="Searching Wiki Pages",
        total=totalNumPages
    ):
        with torch.no_grad():
            titleState.fill_(0)
            for token in titleTokens:
                titleState = model.fastForward(titleState, token)
            titleState = model.fastForward(titleState, sepToken)

            # Normalize titleState
            titleState /= torch.norm(titleState)

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
    for i, (titleTokens, textTokens) in enumerate(loadTokensIndices(dataFolder, resultPages)):
        title = decode(titleTokens)
        text = decode(textTokens)
        print(f"{i}: {title}")
        pages.append(title, text)

    print(f"Finished")
    return pages


if __name__ == "__main__":
    # Initialize stuff
    print("Loading model...")
    model: RNNLanguage = torch.load("models/embedArticle/2.pt")
    model.preCompute()

    sepToken = len(charToToken)

    dataFolder = "tokenData"

    numResults = 5

    totalNumPages = countNumPages(dataFolder)

    while True:
        query = input("Enter query: ")
        if query == "":
            print("exiting...")
            break

        pages = searchWiki(query, model, dataFolder, totalNumPages)
        exit(0)
