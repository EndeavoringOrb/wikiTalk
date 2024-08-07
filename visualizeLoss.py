import matplotlib.pyplot as plt
import csv

# Lists to store the data
pages = []
loss = []

# Read the data from the file
with open('models/embedArticle/0/loss.txt', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        pages.append(int(row[0]))
        loss.append(float(row[2]))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(pages, loss, marker='o')

# Customize the plot
plt.title('Loss vs. Pages Trained On')
plt.xlabel('Number of Pages Trained On')
plt.ylabel('Loss')

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()