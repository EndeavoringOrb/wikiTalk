import matplotlib.pyplot as plt
import csv
import numpy as np

# Lists to store the data
pages = []
loss = []

# Read the data from the file
with open("models/embedArticle/0/loss.txt", "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        pages.append(int(row[0]))
        loss.append(float(row[2]))

# Convert lists to numpy arrays
pages_array = np.array(pages)
loss_array = np.array(loss)

# Calculate the line of best fit
coefficients = np.polyfit(pages_array, loss_array, 1)
line_of_best_fit = np.poly1d(coefficients)

m, b = line_of_best_fit.coeffs

print(f"Line of Best Fit: {m}x + {b}")

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(pages, loss, color="blue", label="Data points")
plt.plot(
    pages_array, line_of_best_fit(pages_array), color="red", label="Line of best fit"
)

# Customize the plot
plt.title("Loss vs. Pages Trained On")
plt.xlabel("Number of Pages Trained On")
plt.ylabel("Loss")

# Add grid lines
plt.grid(True, linestyle="--", alpha=0.7)

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
