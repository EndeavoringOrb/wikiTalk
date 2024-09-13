import bz2
import mwxml

# Open the .bz2 file
with bz2.open("wikiData/enwiki-20240720-pages-articles-multistream.xml.bz2", 'rt') as file:
    # Use mwxml to parse the file
    dump = mwxml.Dump.from_file(file)

    # Iterate through the pages
    for page in dump:
        # Process each page as needed
        print(page.title)
        
        # If you want to process revisions
        for revision in page:
            print(revision.text)