import struct
import vocab
from helperFuncs import *

def write_compact_data(data, filename):
    with open(filename, 'wb') as f:
        # Write the number of tuples
        f.write(struct.pack('I', len(data)))
        
        for inner_list1, inner_list2 in data:
            # Write the length of each inner list (using 4 bytes for each length)
            f.write(struct.pack('II', len(inner_list1), len(inner_list2)))
            
            # Write the data for each inner list
            f.write(bytes(inner_list1))
            f.write(bytes(inner_list2))

def read_compact_data(filename):
    with open(filename, 'rb') as f:
        # Read the number of tuples
        num_tuples = struct.unpack('I', f.read(4))[0]

        for _ in range(num_tuples):
            # Read the length of each inner list
            len1, len2 = struct.unpack('II', f.read(8))
            
            # Read the data for each inner list
            inner_list1 = list(f.read(len1))
            inner_list2 = list(f.read(len2))

            yield inner_list1, inner_list2

def main():
    wikiFolder = "wikiData"
    saveFolder = "tokenData"
    maxNumTokens = 50_000_000

    tokens = []
    numTokens = 0
    currentNumTokens = 0
    fileNum = 0

    charToToken = {character: idx for idx, character in enumerate(sorted(list(vocab.vocab)))}

    for title, text in tqdm(wikiLoader(wikiFolder, vocab.vocab, vocab.replace), desc="Tokenizing pages"):
        titleTokens = [charToToken[character] for character in title]
        textTokens = [charToToken[character] for character in text]

        numTokens += len(titleTokens) + len(textTokens)
        currentNumTokens += len(titleTokens) + len(textTokens)

        if currentNumTokens > maxNumTokens:
            write_compact_data(tokens, f"{saveFolder}/{fileNum}.bin")
            tokens = []
            currentNumTokens = len(titleTokens) + len(textTokens)
            fileNum += 1

        tokens.append((titleTokens, textTokens))

    print(f"Total number of tokens: {numTokens:,}")

if __name__ == "__main__":
    main()