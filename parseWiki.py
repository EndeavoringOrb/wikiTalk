import mwxml
from trainEmbed import vocab

# Load the Wikipedia dump
path = "wikiData\wiki0\enwiki-20240720-pages-articles-multistream1.xml-p1p41242"
dump: mwxml.Dump = mwxml.Dump.from_file(path)

invalidChars = {}

# Iterate through pages
for i, page in enumerate(dump):
    if page.namespace == 0 and page.redirect == None and hasattr(page, "title"):
        title = page.title
        text: str = ""
        for revision in page:
            text = revision.text
        
        for char in title.lower().replace("–", "-"):
            if char not in vocab:
                if char not in invalidChars:
                    invalidChars[char] = 1
                else:
                    invalidChars[char] += 1
                print(f"{i}: {title}, {len(text)}")

invalidChars = sorted(list(invalidChars.items()), key=lambda x: x[1])
for char, freq in invalidChars:
    print(f"{char}: {freq}")