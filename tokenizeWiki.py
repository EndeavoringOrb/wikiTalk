import struct
import vocab
from helperFuncs import *
from tqdm import trange

def write_compact_data(data, filename):
    with open(filename, "wb") as f:
        # Write the number of tuples
        f.write(struct.pack("I", len(data)))

        for inner_list1, inner_list2 in data:
            # Write the length of each inner list (using 4 bytes for each length)
            f.write(struct.pack("II", len(inner_list1), len(inner_list2)))

            # Write the data for each inner list
            f.write(bytes(inner_list1))
            f.write(bytes(inner_list2))


def read_compact_data(filename):
    with open(filename, "rb") as f:
        # Read the number of tuples
        num_tuples = struct.unpack("I", f.read(4))[0]

        for _ in range(num_tuples):
            # Read the length of each inner list
            len1, len2 = struct.unpack("II", f.read(8))

            # Read the data for each inner list
            inner_list1 = list(f.read(len1))
            inner_list2 = list(f.read(len2))

            yield inner_list1, inner_list2


def read_compact_data_indices(filename, indices=None):
    with open(filename, "rb") as f:
        # Read the number of tuples
        num_tuples = struct.unpack("I", f.read(4))[0]

        if indices is None:
            indices = range(num_tuples)
        else:
            indices = sorted(set(indices))  # Remove duplicates and sort

        current_index = 0
        for index in indices:
            if index >= num_tuples:
                break

            # Skip tuples until we reach the desired index
            while current_index < index:
                # Read and skip the lengths
                len1, len2 = struct.unpack("II", f.read(8))
                # Skip the data
                f.seek(len1 + len2, 1)  # 1 means relative to current position
                current_index += 1

            # Read the length of each inner list
            len1, len2 = struct.unpack("II", f.read(8))

            # Read the data for each inner list
            inner_list1 = list(f.read(len1))
            inner_list2 = list(f.read(len2))

            yield inner_list1, inner_list2
            current_index += 1


def read_compact_data_titles(filename):
    with open(filename, "rb") as f:
        # Read the number of tuples
        num_tuples = struct.unpack("I", f.read(4))[0]

        for pageIndex in trange(num_tuples, desc=f"Reading {filename}"):
            # Read the length of each inner list
            len1, len2 = struct.unpack("II", f.read(8))

            # Read the data for inner_list1
            inner_list1 = list(f.read(len1))

            # Skip inner_list2
            f.seek(len2, 1)  # 1 means relative to current position

            yield pageIndex, inner_list1
        clearLines(1)


def read_compact_data_texts(filename):
    with open(filename, "rb") as f:
        # Read the number of tuples
        num_tuples = struct.unpack("I", f.read(4))[0]

        for pageIndex in range(num_tuples):
            # Read the length of each inner list
            len1, len2 = struct.unpack("II", f.read(8))

            # Skip inner_list1
            f.seek(len1, 1)  # 1 means relative to current position

            # Read the data for inner_list2
            inner_list2 = list(f.read(len2))

            yield pageIndex, inner_list2


def loadTokens(folder):
    files = sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0]))
    for fileIndex, file in enumerate(files):
        for titleTokens, textTokens in read_compact_data(f"{folder}/{file}"):
            yield fileIndex, titleTokens, textTokens


def loadTokensIndices(folder, indices):
    files = sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0]))
    indices = sorted(indices)  # [(fileIndex, pageIndex), ...]

    # Group indices by file
    file_indices = {}
    for file_index, page_index in indices:
        if file_index not in file_indices:
            file_indices[file_index] = []
        file_indices[file_index].append(page_index)

    for file_index, file in enumerate(files):
        if file_index in file_indices:
            filename = f"{folder}/{file}"
            for titleTokens, textTokens in read_compact_data_indices(
                filename, file_indices[file_index]
            ):
                yield titleTokens, textTokens


def loadTitles(folder):
    files = sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0]))
    for fileIndex, file in enumerate(files):
        for pageIndex, titleTokens in read_compact_data_titles(f"{folder}/{file}"):
            yield fileIndex, pageIndex, titleTokens


def getPage(titleTokens, folder):
    # First, find the file and page index
    fileIndex, pageIndex = findFileAndPageIndex(titleTokens, folder)

    if fileIndex is None or pageIndex is None:
        return None  # Page not found

    # Construct the filename
    filename = f"{folder}/{fileIndex}.bin"

    # Use read_compact_data_indices to get the page
    for inner_list1, inner_list2 in read_compact_data_indices(filename, [pageIndex]):
        return inner_list1, inner_list2

    return None  # This should not happen if the index was correct


def findFileAndPageIndex(titleTokens, folder):
    files = sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0]))

    for fileIndex, file in enumerate(files):
        filename = f"{folder}/{file}"
        for pageIndex, pageTitleTokens in read_compact_data_titles(filename):
            if pageTitleTokens == titleTokens:
                return fileIndex, pageIndex

    return None, None  # Page not found


def countNumPages(folder):
    numPages = []
    files = sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0]))

    for file in files:
        filename = os.path.join(folder, file)
        with open(filename, "rb") as f:
            # Read the number of tuples in this file
            numTuples = struct.unpack("I", f.read(4))[0]
            numPages.append(numTuples)

    return numPages


def countNumTokens(folder):
    numPages = 0
    numTokens = 0

    files = sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0]))

    for file in files:
        filename = f"{folder}/{file}"
        fileNumPages, fileNumTokens = countTokensInFile(filename)
        numPages += fileNumPages
        numTokens += fileNumTokens

    return numPages, numTokens


def countTokensInFile(filename):
    file_tokens = 0
    with open(filename, "rb") as f:
        # Read the number of tuples
        num_tuples = struct.unpack("I", f.read(4))[0]

        for _ in trange(num_tuples, desc=f"Reading {filename}"):
            # Read the length of each inner list
            len1, len2 = struct.unpack("II", f.read(8))

            # Add the lengths to the token count
            file_tokens += len1 + len2

            # Skip the actual data
            f.seek(len1 + len2, 1)  # 1 means relative to current position
        clearLines(1)

    return num_tuples, file_tokens


def tokenize(text):
    return [charToToken[character] for character in text]


def decode(tokens):
    return "".join([tokenToChar[token] for token in tokens])


charToToken = {
    character: idx for idx, character in enumerate(sorted(list(vocab.vocab)))
}

tokenToChar = {idx: character for character, idx in charToToken.items()}

def main():
    wikiFolder = "wikiData"
    saveFolder = "tokenData"
    maxNumTokensPerFile = 50_000_000

    tokens = []
    numTokens = 0
    currentNumTokens = 0
    fileNum = 0

    for title, text in tqdm(
        wikiLoader(wikiFolder, vocab.vocab, vocab.replace), desc="Tokenizing pages"
    ):
        titleTokens = tokenize(title)
        textTokens = tokenize(text)

        numTokens += len(titleTokens) + len(textTokens)
        currentNumTokens += len(titleTokens) + len(textTokens)

        if currentNumTokens > maxNumTokensPerFile:
            write_compact_data(tokens, f"{saveFolder}/{fileNum}.bin")
            tokens = []
            currentNumTokens = len(titleTokens) + len(textTokens)
            fileNum += 1

        tokens.append((titleTokens, textTokens))

    print(f"Total number of tokens: {numTokens:,}")


if __name__ == "__main__":
    main()
