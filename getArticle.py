from tokenizeWiki import *

title = input("Enter exact title: ")

print("tokenizing query title...")
titleTokens = tokenize(title)

print("searching...")
result = getPage(titleTokens, "tokenData")
if result == None:
    print("no pages found")
    exit(0)
pageTitleTokens, pageTextTokens = result

print("decoding results...")
pageTitle = decode(pageTitleTokens)
pageText = decode(pageTextTokens)

print(pageTitle)
print()
print(pageText)
print()
print(f'Text in page "{pageTitle}" has {len(pageText):,} characters.')
