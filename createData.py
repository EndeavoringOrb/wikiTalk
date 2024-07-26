from helperFuncs import *
import json
import os
from trainEmbed import vocab
import torch
from tqdm import tqdm

@torch.no_grad()
def searchWiki(query, path, model, numResults=5):
    results = []

    query_tensor = torch.tensor(
        [vocab[character] for character in query.lower()]
    )
    queryEmbedding = model(query_tensor)
    length1 = torch.norm(queryEmbedding)

    files = os.listdir(path)
    if files[0] == "info.txt":
        wikiPath = files[1]
    else:
        wikiPath = files[0]
    with open(f"{path}/info.txt", "r", encoding="utf-8") as f:
        text = f.read()
    numPages = int(text.split(" ")[0].strip())

    for title, text in tqdm(wikiLoader(f"{path}/{wikiPath}", vocab), desc="Searching Wiki", total=numPages):

        title_tensor = torch.tensor(
            [vocab[character] for character in title.lower()]
        )
        titleEmbedding = model(title_tensor)
        similarity = torch.sum(queryEmbedding * titleEmbedding).item()

        # Normalize
        length2 = torch.norm(titleEmbedding)
        similarity /= length1 * length2

        if len(results) == 0:
            results.append((title, text, similarity))
            continue

        inserted = False

        for i, result in enumerate(results):
            if similarity > result[2]:
                results.insert(i, (title, text, similarity))
                if len(results) == numResults + 1:
                    results.pop()
                inserted = True
                break

        if not inserted and len(results) < numResults:
            results.append((title, text, similarity))

    return results


def getConversationText(conversation: list[str]):
    return "\n".join(conversation)


def getConversationSpeech(conversation: list[str]):
    valid_lines = []
    for line in conversation:
        if line.startswith("User: ") or line.startswith("Assistant: "):
            valid_lines.append(line)
    return "\n".join(valid_lines)


def getAssistantInput(conversation: list[str], prompt: str):
    os.system("cls")
    conversation.append(prompt)
    text = input(getConversationText(conversation))  # get search query
    conversation[-1] += text
    return text


def addOptions(conversation: list[str], *options):
    for option in options:
        conversation.append(f"-{option}")


def processSearch(conversation: list[str], wikiPath, model, numSearchResults):
    assistantText = getAssistantInput(conversation, "Enter query: ")  # get search query
    searchResults = searchWiki(assistantText, wikiPath, model, numSearchResults)  # search wiki

    # print titles
    for i in range(len(searchResults)):
        conversation.append(f"{i + 1}: {searchResults[i][0]}")

    # add options
    addOptions(conversation, "get article", "talk")

    # get assistant response
    assistantText = getAssistantInput(conversation, "+")

    while assistantText == "get article":
        assistantText = getAssistantInput(conversation, "Enter Article #: ")
        articleNum = int(assistantText)
        conversation.append(searchResults[articleNum - 1][1])

        # add options
        addOptions(conversation, "get article", "search", "talk")

        # get assistant response
        assistantText = getAssistantInput(conversation, "+")

    if assistantText == "search":
        processSearch(conversation)


def getUserInput(conversation: list[str]):
    os.system("cls")
    conversationSpeech = getConversationSpeech(conversation)

    # get user text
    if len(conversation) > 0:
        userText = input(conversationSpeech + "\nUser: ")
    else:
        userText = input("User: ")
    
    conversation.append(f"User: {userText}")


def main():
    saveFolder = "conversationData"
    wikiPath = "wikiData\wiki0"
    modelLoadPath = "models/embed/0.pt"
    numSearchResults = 5

    model = torch.load(modelLoadPath)

    while True:
        input("Start new conversation?")
        conversation = []
        fileNum = getNextDataFile(saveFolder)

        while True:
            # get user text
            getUserInput(conversation)
            if conversation[-1] == "User: ":
                conversation.pop()
                if conversation[-1] == "\n":
                    conversation.pop()
                break

            # add options
            addOptions(conversation, "search", "talk")

            # get assistant response
            assistantText = getAssistantInput(conversation, "+")

            # process assistant response
            if assistantText == "search":
                processSearch(conversation, wikiPath, model, numSearchResults)

            # assistant has selected talk

            # get assistant response
            getAssistantInput(conversation, "Assistant: ")

            print("saving...")
            if len(conversation) > 0:
                with open(f"{saveFolder}/{fileNum}.json", "w", encoding="utf-8") as f:
                    json.dump(conversation, f)


if __name__ == "__main__":
    main()