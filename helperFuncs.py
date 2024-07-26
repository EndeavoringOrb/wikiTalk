import mwxml
import os


def wikiLoader(path, vocab):
    # Load the Wikipedia dump
    dump: mwxml.Dump = mwxml.Dump.from_file(path)

    # Iterate through pages
    for i, page in enumerate(dump):
        if page.namespace == 0 and page.redirect == None and hasattr(page, "title"):
            title = page.title
            text: str = ""
            for revision in page:
                text = revision.text

            title = title.lower().replace("–", "-")
            validTitle = True
            for char in title:
                if char not in vocab:
                    validTitle = False
                    break

            if validTitle:
                yield title, text


def getNextDataFile(folder):
    files: list[str] = os.listdir(folder)
    files = [file for file in files if file.endswith(".json")]
    if len(files) == 0:
        return 0  # return 0 if there are no files

    # otherwise, return 1 higher than the largest number
    maxNum = max(files, key=lambda x: int(x.split(".")[0]))
    maxNum = int(maxNum.split(".")[0])
    return maxNum + 1


def clearLines(numLines):
    for _ in range(numLines):
        print("\033[F\033[K", end="")
