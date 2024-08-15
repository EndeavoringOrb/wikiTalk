import matplotlib.pyplot as plt
import csv
import numpy as np

# Lists to store the data
pages = []
tokens = []
loss = []

# Read the data from the file
with open("models/embedArticle/0/loss.txt", "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        pages.append(int(row[0]))
        tokens.append(int(row[1]))
        loss.append(float(row[2]))

# Convert lists to numpy arrays
pages_array = np.array(pages)
tokens_array = np.array(tokens)
loss_array = np.array(loss)

# set x
x_array = tokens_array
x = tokens 

# Calculate the line of best fit
coefficients = np.polyfit(x_array, loss_array, 1)
line_of_best_fit = np.poly1d(coefficients)

m, b = line_of_best_fit.coeffs

print(f"Line of Best Fit: {m}x + {b}")

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(x, loss, color="blue", label="Data points")
plt.plot(
    x_array, line_of_best_fit(x_array), color="red", label="Line of best fit"
)

# Customize the plot
plt.title("Embedding Loss")
plt.xlabel("Number of Tokens Trained On")
plt.ylabel("Loss [0, 16] (lower is better)")

# Add grid lines
plt.grid(True, linestyle="--", alpha=0.7)

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
