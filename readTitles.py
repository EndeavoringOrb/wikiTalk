from tokenizeWiki import read_compact_titles, tokenize, decode, loadPage

# Load all titles
print("Loading all titles")
numToPrint = int(input("How many titles do you want to print?: "))
for fileIndex, pageIndex, pageTitleTokens in read_compact_titles(
    "tokenData/titles.bin"
):
    if numToPrint <= 0:
        numToPrint = int(input("How many titles do you want to print?: "))
    numToPrint -= 1
    print(decode(pageTitleTokens))
