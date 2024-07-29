import mwxml
from tqdm import tqdm
from trainEmbed import vocab
import os


def countPages(folder):
    numPages = 0
    with tqdm(desc="counting pages") as pbar:
        for subFolder in sorted(
            os.listdir(folder), key=lambda x: int(x.strip("wiki")) if "wiki" in x else 0
        ):
            if subFolder == "info.txt":
                continue
            for file in os.listdir(f"{folder}/{subFolder}"):
                if file == "info.txt":
                    continue

                pbar.set_description(f"counting pages [{folder}/{subFolder}]")

                dump: mwxml.Dump = mwxml.Dump.from_file(f"{folder}/{subFolder}/{file}")

                for page in dump:
                    if page.namespace == 0 and page.redirect == None:
                        numPages += 1
                        pbar.update(1)

    return numPages


# Load the Wikipedia dump
path = "wikiData\wiki0\enwiki-20240720-pages-articles-multistream1.xml-p1p41242"  # 27372
path = "wikiData\wiki1\enwiki-20240720-pages-articles-multistream2.xml-p41243p151573"  # 83498
path = "wikiData\wiki2\enwiki-20240720-pages-articles-multistream3.xml-p151574p311329"  # 89644
path = "wikiData\wiki3\enwiki-20240720-pages-articles-multistream4.xml-p311330p558391"  # 140261
path = "wikiData\wiki4\enwiki-20240720-pages-articles-multistream5.xml-p558392p958045"  # 228686
path = "wikiData\wiki5\enwiki-20240720-pages-articles-multistream6.xml-p958046p1483661"  # 262117
path = "wikiData\wiki6\enwiki-20240720-pages-articles-multistream7.xml-p1483662p2134111"  # 303592

dump: mwxml.Dump = mwxml.Dump.from_file(path)

invalidChars = {}
numPages = 0

# Iterate through pages
for i, page in enumerate(tqdm(dump)):
    if page.namespace == 0 and page.redirect == None and hasattr(page, "title"):
        title = page.title
        text: str = ""
        for revision in page:
            text = revision.text

        for char in title.lower():
            if char not in vocab:
                if char not in invalidChars:
                    invalidChars[char] = 1
                else:
                    invalidChars[char] += 1
                print(f"{i}: {title}, {len(text)}")
    numPages += 1

print(f"Num Pages: {numPages:,}")

invalidChars = sorted(list(invalidChars.items()), key=lambda x: x[1])
for char, freq in invalidChars:
    print(f"{char}: {freq}")
