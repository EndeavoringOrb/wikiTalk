from tokenizeWiki import loadTitles, write_compact_titles

currentFileIndex = -1
allTitles = []
for fileIndex, pageIndex, titleTokens in loadTitles("tokenData/articles"):
    if fileIndex != currentFileIndex:
        allTitles.append([])
    allTitles[fileIndex].append(titleTokens)

write_compact_titles(allTitles, f"tokenData/titles.bin")