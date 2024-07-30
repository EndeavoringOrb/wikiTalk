vocab = "abcdefghijklmnopqrstuvwxyz"  # lowercase letters
vocab += vocab.upper()  # uppercase letters
vocab += "\n "  # whitespace
vocab += "0123456789"  # numbers
vocab += "-,.!?—"  # punctuation
vocab += "'\""  # quotation
vocab += "$%^&*(){|}+=[]:;/\\_#<>~"  # symbols
vocab = set(vocab)

replace = [
    ("–", "-"),
    ("−", "-"),
    ("а", "a"),
    ("о", "o"),
    ("е", "e"),
    ("’", "'"),
    ("”", "\""),
    ("“", "\""),
]

# all pages: 1,418,133
# valid pages: 551,617