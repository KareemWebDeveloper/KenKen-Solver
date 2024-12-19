import matplotlib.pyplot as plt
import numpy as np

# Example Data
generations = np.arange(1, 21)  # Generations 1 to 20
best_fitness = [10, 12, 15, 18, 20, 25, 28, 30, 32, 35, 37, 38, 39, 40, 40, 40, 40, 40, 40, 40]
avg_fitness = [7.5, 9.2, 10.8, 13, 15, 18, 21, 24, 26, 28, 30, 31, 32, 33, 34, 34.5, 35, 35.2, 35.4, 35.6]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(generations, best_fitness, label="Best Fitness", marker='o', color='b')
plt.plot(generations, avg_fitness, label="Average Fitness", marker='s', color='g', linestyle='--')

# Labels and Title
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Genetic Algorithm: Fitness Over Generations")
plt.legend()
plt.grid(True)

# Show Plot
plt.show()
