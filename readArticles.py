from tokenizeWiki import loadTokens, decode
import os

tokenFolder = "tokenData"
tokenLoader = loadTokens(tokenFolder)

for fileNum, title, text in tokenLoader:
    if input("Print new Article? [Y/N]: ").lower() == "y":
        os.system("cls")
        print(decode(title))
        print(decode(text))
