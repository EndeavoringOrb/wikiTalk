import bz2
import mwxml
from collections import Counter
import pickle

dump = mwxml.Dump.from_file("wikiData/enwiki-20240720-pages-articles-multistream.xml")

total_char_frequency = Counter()
processed_pages = 0
total_pages = 0

# Iterate through the pages
for page in dump:
    total_pages += 1
    # Process each page as needed
    #print(f"Processing page: {page.title}")

    # Make sure the page is a main article
    if page.namespace != 0 or page.redirect is not None:
        continue

    # Get the most recent revision
    mostRecentRevision = None
    for revision in page:
        mostRecentRevision = revision.text

    if mostRecentRevision is None:
        continue
    
    total_char_frequency.update(mostRecentRevision)
    processed_pages += 1

    # Optional: Print progress every 1000 pages
    if processed_pages % 1000 == 0:
        print(f"Processed {processed_pages}/{total_pages} pages")
        with open("freqs.pickle", "wb") as f:
            pickle.dump(total_char_frequency, f)

# Print the total character frequency
print("\nTotal character frequency:")
for char, count in total_char_frequency.most_common():
    print(f"'{char}': {count}")

print(f"\nTotal pages processed: {processed_pages}")