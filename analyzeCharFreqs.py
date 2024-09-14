import pickle
from collections import Counter

with open("freqs.pickle", "rb") as f:
    total_char_frequency: Counter = pickle.load(f)

for i, item in enumerate(total_char_frequency.most_common(512)):
    print(f"{i}: {item}")

print(f"Total # characters: {len(total_char_frequency):,}")