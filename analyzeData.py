from helperFuncs import *
from vocab import *


def main():
    folder = "wikiData"

    numPages, numValidPages, newVocab = getInfo(folder, vocab, replace)

    print(f"{folder} has {numPages:,} pages.")
    print(f"{folder} has {numValidPages:,} valid pages.")

    print(f"{folder} has extra vocab with size of {len(newVocab):,}")
    newVocab = sorted(list(newVocab.items()), key=lambda x: x[1], reverse=True)
    print(newVocab)

    with open(f"{folder}/vocab.txt", "w", encoding="utf-8") as f:
        f.write("".join([item[0] for item in newVocab]))


if __name__ == "__main__":
    main()
