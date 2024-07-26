from bs4 import BeautifulSoup
import requests
import json
import os
from helperFuncs import *


def google_search(query, num_results=10):
    url = f"https://www.google.com/search?q={query}&num={num_results}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    search_results = []
    for result in soup.find_all("div", class_="yuRUbf"):
        title = result.find("h3", class_="LC20lb MBeuO DKV0Md").text
        # title = result.find('h3', class_='r').text
        link = result.find("a")["href"]
        search_results.append({"title": title, "link": link})

        if len(search_results) >= num_results:
            break

    return search_results


def getTitles():
    titles = []

    while True:
        title = input("Enter title: ")
        if title == "":
            break

    return titles


def getTitle(link: str):
    title = link.split("https://en.wikipedia.org/wiki/")[1].replace("_", " ")
    return title


def countDataTypes(folder):
    files: list[str] = os.listdir(folder)
    files = [file for file in files if file.endswith(".json")]

    numTrue = 0
    numFalse = 0

    for file in files:
        with open(f"{folder}/{file}", "r", encoding="utf-8") as f:
            data = json.load(f)
        for query, title, same in data:
            if same:
                numTrue += 1
            else:
                numFalse += 1

    return numTrue, numFalse


def main():
    # Set save folder
    saveFolder = "embeddingData"
    manual = True

    # Initialize data
    data = []

    # Get counters
    numTrue, numFalse = countDataTypes(saveFolder)
    fileNum = getNextDataFile(saveFolder)

    while True:
        print(f"# Same: {numTrue:,}")
        print(f"# Different: {numFalse:,}")
        # Get query
        query = input("Enter Query: ")
        if query == "":
            print("exiting data collection loop...")
            break
        googleQuery = f"{query} site:en.wikipedia.org"

        # Get results
        if manual:
            title = input("Enter Title: ")
            same = True if input("Same?: ").lower() == "y" else False
            if same:
                numTrue += 1
            else:
                numFalse += 1
            data.append((query, title, same))
            clearLines(5)
        else:
            print("searching...")
            results = google_search(googleQuery, num_results=5)
            clearLines(1)

            # Display results
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"     {result['link']}\n")

            choice = input(f"Enter choice [{1}-{len(results)}]: ")
            choice = int(choice)

            if choice == 0:
                result = input("Enter False Result: ")
                data.append((query, result, False))
                numFalse += 1

            if choice > 0 and choice <= len(results):
                data.append((query, getTitle(results[choice - 1]["link"]), True))
                numTrue += 1

            clearLines(4 + len(results) * 3 + (choice == 0))

        print("saving...")
        if len(data) > 0:
            with open(f"{saveFolder}/{fileNum}.json", "w", encoding="utf-8") as f:
                json.dump(data, f)

        clearLines(1)


if __name__ == "__main__":
    main()
