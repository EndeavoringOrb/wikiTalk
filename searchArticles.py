from tokenizeWiki import read_compact_titles, tokenize, decode, loadPage

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
    text = input("Enter article title: ")
    queryTokens = tokenize(text)

    foundArticle = False
    titleTokens, pageTokens = None, None

    for fileIndex, titles in enumerate(fileTitles):
        for pageIndex, titleTokens in enumerate(titles):
            if queryTokens == titleTokens:
                titleTokens, pageTokens = loadPage("tokenData", fileIndex, pageIndex)
                foundArticle = True
                break
        if foundArticle:
            break

    if foundArticle:
        print(f"Found article: {decode(titleTokens)}")
        print(decode(pageTokens))
    else:
        print(f'No article found for "{text}"')
