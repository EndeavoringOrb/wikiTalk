import mwxml
import os
from tqdm import tqdm

def replaceChars(text: str, replace: list):
    for invalid, valid in replace:
        text = text.replace(invalid, valid)
    return text


"""counts the number of valid pages given the vocab"""
def getInfo(folder, vocab, replace):
    numPages = 0
    numValidPages = 0

    newVocab = {}

    with tqdm(desc="valid pages: 0/0", mininterval=1) as pbar:
        for subFolder in sorted(
            os.listdir(folder), key=lambda x: int(x.strip("wiki")) if "wiki" in x else 0
        ):
            if subFolder.endswith(".txt"):
                continue
            for file in os.listdir(f"{folder}/{subFolder}"):
                if file.endswith(".txt"):
                    continue

                dump: mwxml.Dump = mwxml.Dump.from_file(f"{folder}/{subFolder}/{file}")

                for page in dump:
                    if page.namespace == 0 and page.redirect == None:
                        numPages += 1
                        pbar.update(1)

                        title = replaceChars(page.title, replace)
                        text: str = ""
                        for revision in page:
                            text = revision.text
                        text = replaceChars(text, replace)

                        pageSet = set(title + text)
                        if not pageSet.issubset(vocab):
                            # update newVocab freqencies
                            for char in title + text:
                                if char not in vocab:
                                    if char not in newVocab:
                                        newVocab[char] = 0
                                    newVocab[char] += 1

                            continue

                        numValidPages += 1

                    if numPages % 100 == 0 and numPages > 0:
                        pbar.set_description(
                            f"valid pages: {numValidPages:,}/{numPages:,} ({100 * numValidPages / numPages:.2f}%)"
                        )

    # return numPages, the characters in newVocab that are not in vocab
    return numPages, numValidPages, newVocab


def wikiLoader(folder):
    for subFolder in os.listdir(folder):
        if subFolder.endswith(".txt"):
            continue
        for file in os.listdir(f"{folder}/{subFolder}"):
            if file.endswith(".txt"):
                continue

            # Load the Wikipedia dump
            dump: mwxml.Dump = mwxml.Dump.from_file(f"{folder}/{subFolder}/{file}")

            # Iterate through pages
            for i, page in enumerate(dump):
                if page.namespace == 0 and page.redirect == None:
                    title = page.title
                    text: str = ""
                    for revision in page:
                        text = revision.text

                    yield title, text

def wikiLoader(folder, vocab, replace):
    for subFolder in os.listdir(folder):
        if subFolder.endswith(".txt"):
            continue
        for file in os.listdir(f"{folder}/{subFolder}"):
            if file.endswith(".txt"):
                continue

            # Load the Wikipedia dump
            dump: mwxml.Dump = mwxml.Dump.from_file(f"{folder}/{subFolder}/{file}")

            # Iterate through pages
            for i, page in enumerate(dump):
                if page.namespace == 0 and page.redirect == None:
                    # Clean title
                    title = replaceChars(page.title, replace)

                    # Get most recent revision text, the clean text
                    text: str = ""
                    for revision in page:
                        text = revision.text
                    text = replaceChars(text, replace)

                    # If the title and text are only using characters that are in vocab, then yield the title and text
                    pageSet = set(title + text)
                    if pageSet.issubset(vocab):
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