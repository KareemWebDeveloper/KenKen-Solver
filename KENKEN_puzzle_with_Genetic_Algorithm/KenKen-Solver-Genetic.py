import numpy as np
import random
import math

def generate_individual(grid_size):
    """Generate a valid individual (grid) with no row or column duplicates."""
    grid = np.zeros((grid_size, grid_size), dtype=int)
    
    for row in range(grid_size):
        available_numbers = set(range(1, grid_size + 1))
        
        for col in range(grid_size):
            used_in_col = set(grid[:row, col])  # Numbers already used in the column
            valid_numbers = available_numbers - used_in_col

            if not valid_numbers:
                return generate_individual(grid_size)

            selected_number = random.choice(list(valid_numbers))
            grid[row, col] = selected_number
            available_numbers.remove(selected_number)
    
    return grid


def calculate_cage_result(cage_values, operation):
    if operation == "+":
        return sum(cage_values)
    elif operation == "-":
        if len(cage_values) != 2:
            return float('inf')
        return abs(cage_values[0] - cage_values[1])
    elif operation == "*":
        result = 1
        for value in cage_values:
            result *= value
        return result
    elif operation == "/":
        if len(cage_values) != 2:
            return float('inf')
        return max(cage_values) // min(cage_values)
    else:
        raise ValueError("Unsupported operation: {}".format(operation))


def check_uniqueness(solution, grid_size):
    fitness_penalty = 0
    MAX_PENALTY = grid_size * 10  # Increased penalty to ensure high impact
    
    for i in range(grid_size):
        row_set = set()
        col_set = set()
        for j in range(grid_size):
            row_value = solution[i][j]
            col_value = solution[j][i]

            # Apply high penalty for duplicates in rows or columns
            if row_value in row_set:
                fitness_penalty += MAX_PENALTY
            else:
                row_set.add(row_value)

            if col_value in col_set:
                fitness_penalty += MAX_PENALTY
            else:
                col_set.add(col_value)

    return fitness_penalty

def calculate_fitness(individual, cages, grid_size):
    """Calculates the fitness of an individual.

    Args:
        individual: A 2D numpy array representing a Kenken solution.
        cages: A list of cages, each represented as a tuple (operation, target, cells).
        grid_size: The size of the Kenken grid.

    Returns:
        The fitness score of the individual.
    """

    fitness = 0

    # Check for uniqueness in rows and columns
    fitness += check_uniqueness(individual, grid_size)

    # Check cage constraints with a hard penalty for invalid results
    HARD_PENALTY = 100 # Large fixed penalty for constraint violation

    for operation, target, cells in cages:
        cage_values = [individual[i][j] for i, j in cells]
        result = calculate_cage_result(cage_values, operation)
        
        if result != target:
            fitness += HARD_PENALTY  # Apply hard penalty for invalid cage constraint

    print(f"Generation Fitness: {fitness}")
    return fitness


def tournament_selection(population, fitness_scores, tournament_size):
    tournament_indices = np.random.randint(0, len(population), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    best_index = tournament_indices[np.argmin(tournament_fitness)]
    return population[best_index]

def one_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, grid_size - 1)
    child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(grid_size):
        for j in range(grid_size):
            if np.random.rand() < mutation_rate:
                row1, col1 = np.random.randint(0, grid_size, 2)
                individual[i][j], individual[row1][col1] = individual[row1][col1], individual[i][j]
    return individual

def pmx_crossover(parent1, parent2):
    crossover_points = np.random.choice(grid_size - 1, 2, replace=False)
    crossover_points.sort()
    child1 = parent1.copy()
    child2 = parent2.copy()

    for i in range(crossover_points[0], crossover_points[1] + 1):
        child1[i] = parent2[i]
        child2[i] = parent1[i]

    # Ensure no duplicates are introduced in rows or columns
    def fix_duplicates(child, parent1, parent2):
        for i in range(grid_size):
            for j in range(grid_size):
                if child[i][j] == 0:
                    value = parent1[i][j]
                    while value in child[i] or value in [child[x][j] for x in range(grid_size)]:
                        value = parent2[i][j]
                    child[i][j] = value
        return child

    child1 = fix_duplicates(child1, parent1, parent2)
    child2 = fix_duplicates(child2, parent1, parent2)

    return child1, child2


def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [fitness / total_fitness for fitness in fitness_scores]
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]

import numpy as np
import time

def genetic_algorithm(cages, grid_size, population_size, max_generations, mutation_rate, elitism_rate, tolerance=1e-6):
    population = [generate_individual(grid_size) for _ in range(population_size)]
    best_fitness = float('inf')
    start_time = time.time()  # Start timing
    best_solution = None
    best_generation = 0

    for generation in range(max_generations):
        # Calculate fitness for all individuals
        fitness_scores = [calculate_fitness(individual, cages, grid_size) for individual in population]
        
        # Find the best solution in the current generation
        best_index = np.argmin(fitness_scores)
        current_best_fitness = fitness_scores[best_index]
        
        # Print generation stats
        print(f"Generation {generation} - Best Fitness: {current_best_fitness}")
        print(f"Solution:\n{population[best_index]}")

        # Update global best solution
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[best_index]
            best_generation = generation
        
        # Check for optimal solution
        if current_best_fitness == 0:
            print(f"Optimal solution found in generation {generation}")
            print(f"Time taken: {time.time() - start_time:.2f} seconds")
            print(f"Best Solution Fitness: {best_fitness}")
            print("Optimal Solution:")
            print(best_solution)
            return best_solution

        # Early termination check
        # if abs(best_fitness - current_best_fitness) < tolerance:
        #     print("Early termination: No significant improvement in the fitness value")
        #     print(f"The Fitness Value : {current_best_fitness}")
        #     print(f"Time taken: {time.time() - start_time:.2f} seconds")
        #     print(f"Best Solution Found in Generation {best_generation}")
        #     print(best_solution)
        #     return best_solution

        # Elitism: Select the best individuals
        elite_size = int(population_size * elitism_rate)
        elite_indices = np.argsort(fitness_scores)[:elite_size]
        elite_individuals = [population[i] for i in elite_indices]

        # Create a new population
        new_population = elite_individuals.copy()
        while len(new_population) < population_size:
            parent1 = roulette_wheel_selection(population, fitness_scores)
            parent2 = roulette_wheel_selection(population, fitness_scores)
            child1, child2 = pmx_crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.append(child1)
            new_population.append(child2)

        population = new_population

    # Final output after max generations
    print("Max generations reached.")
    print(f"Best Fitness Achieved: {best_fitness}")
    print(f"Found in Generation: {best_generation}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print("Best Solution:")
    print(best_solution)

    return best_solution

cages = [
    ('+', 8, [(0, 0), (0, 1)]),
    ('-', 1, [(0, 2), (0, 3)]),
    ('*', 4, [(1, 0), (1, 1)]),
    # ('+', 3, [(1, 2), (1, 3)]),
    # ('-', 1, [(2, 0), (2, 1)]),
    # ('*', 6, [(2, 2), (2, 3)]),
    # ('+', 5, [(3, 0), (3, 1)]),
    # ('-', 1, [(3, 2), (3, 3)])
]

# cages = [
#     ('+', 6, [(0, 0), (1, 0)]),
#     ('-', 1, [(0, 1), (0, 2)]),
#     ('*', 6, [(0, 3), (1, 3)]),
#     ('/', 3, [(0, 4), (0, 5)]),
#     ('+', 11, [(1, 1), (1, 2), (2, 1)]),
#     ('-', 2, [(1, 4), (1, 5)]),
#     ('*', 12, [(2, 0), (2, 2)]),
#     ('+', 5, [(2, 3), (2, 4)]),
#     ('/', 2, [(2, 5), (3, 5)]),
#     ('-', 1, [(3, 0), (3, 1)]),
#     ('*', 6, [(3, 2), (3, 3)]),
#     ('+', 9, [(3, 4), (4, 3), (4, 4)]),
#     ('+', 8, [(4, 0), (4, 1)]),
#     ('-', 1, [(4, 2), (5, 2)]),
#     ('*', 6, [(5, 0), (5, 1)]),
#     ('+', 7, [(5, 3), (5, 4)])
# ]

grid_size = 6
population_size = 1000
max_generations = 50
mutation_rate = 0.7
elitism_rate = 0.2

# Solve the Kenken puzzle
best_solution = genetic_algorithm(cages, grid_size, population_size, max_generations, mutation_rate , elitism_rate)

# import random
# import math
# import numpy as np


# cages = [
#     ('*', 3, [(0, 0), (1, 0)]),
#     ('+', 5, [(0, 1), (0, 2)]),
#     ('/', 2, [(0, 3), (1, 3)]), 
#     ('*', 8, [(2, 0), (3, 0), (3, 1)]),
#     ('-', 1, [(1, 1), (2, 1)]),
#     ('+', 3, [(1, 2), (2, 2)]),
#     ('+', 7, [(2, 3), (3, 2), (3, 3)])
# ]

# # cages = [
# #    ('+', 7, [(0, 0), (0, 1)]),
# #    ('*', 15, [(0, 2), (0, 3), (1, 2)]),
# #    ('*', 10, [(0, 4), (1, 4), (2,4)]), 
# #    ('*', 4, [(1, 0), (2, 0)]),
# #    ('-', 2, [(1, 1), (2, 1)]),
# #    ('+', 9, [(1, 3), (2, 3), (3,3)]),
# #    ('-', 3, [(2, 2), (3, 2)]),
# #    ('/', 2, [(3, 0), (3, 1)]),
# #    ('*', 12, [(3, 4), (4, 4)]),
# #    ('+', 7, [(4, 0), (4, 1)]),
# #    ('-', 3, [(4, 2), (4, 3)])
# # ]

# def generate_individual(grid_size, seed=None):
#     random.seed(seed)
#     while True:
#         numbers = list(range(1, grid_size + 1))
#         grid = np.zeros((grid_size, grid_size), dtype=int)

#         for i in range(grid_size):
#             random.shuffle(numbers)
#             grid[i] = numbers[:]

#         row_duplicates = any(len(np.unique(row)) < grid_size for row in grid)
#         col_duplicates = any(len(np.unique(col)) < grid_size for col in grid.T)

#         if not (row_duplicates or col_duplicates):
#             break

#     return grid

# def create_population(population_size, grid_size):
#     population = [generate_individual(grid_size, seed=i) for i in range(population_size)]
#     return population

# def calculate_cage_result(cage_values, operation):
#     if operation == "+":
#         return sum(cage_values)
#     elif operation == "-":
#         if len(cage_values) < 2:
#             return 0  # Return 0 if there are not enough values for subtraction
#         return abs(cage_values[0] - sum(cage_values[1:]))
#     elif operation == "*":
#         if len(cage_values) == 0:
#             return 0  # Return 0 if there are no values for multiplication
#         result = 1
#         for value in cage_values:
#             result *= value
#         return result
#     elif operation == "/":
#         if len(cage_values) < 2:
#             return 0  # Return 0 if there are not enough values for division
#         cage_values = sorted(cage_values)
#         result = cage_values[-1]
#         for value in cage_values[:-1]:
#             result /= value
#         return math.floor(result)
#     else:
#         raise ValueError("Unsupported operation: {}".format(operation))

# def check_uniqueness(solution, grid_size):
#     fitness_penalty = 0

#     try:
#         grid = solution  # Assuming 'grid' is a key in your solution dictionary
#         if grid is None or not isinstance(grid, np.ndarray) or grid.shape != (grid_size, grid_size):
#             return grid_size  # Penalize with the maximum possible penalty

#         for i in range(grid_size):
#             row_values = set()
#             col_values = set()
#             for j in range(grid_size):
#                 value_row = grid[i, j]
#                 if value_row is None:
#                     return grid_size  # Penalize with the maximum possible penalty

#                 if value_row in row_values:
#                     fitness_penalty += 1
#                 row_values.add(value_row)

#                 value_col = grid[j, i]
#                 if value_col is None:
#                     return grid_size  # Penalize with the maximum possible penalty

#                 if value_col in col_values:
#                     fitness_penalty += 1
#                 col_values.add(value_col)

#     except IndexError as e:
#         print(f"IndexError: {e}")
#         # Handle the error, possibly by returning a default penalty
#         return grid_size

#     return fitness_penalty

# def evaluate_cage(solution, cages):
#     fitness_penalty = 0
    
#     grid = solution

#     for operation, target_value, cells in cages:
#         # Ensure the cells are valid indices for the grid
#         valid_cells = [(i, j) for i, j in cells if 0 <= i < len(grid) and 0 <= j < len(grid[0])]

#         # Extract the values in the cells corresponding to the current cage
#         cage_values = [grid[i][j] for i, j in valid_cells]

#         # Calculate the result of the cage operation using the extracted values
#         cage_result = calculate_cage_result(cage_values, operation)

#         # Add the absolute difference between the target value and the cage result to the penalty
#         fitness_penalty += abs(target_value - cage_result)

#     return int(fitness_penalty)

# def evaluate_fitness(population, grid_size, cages):
#     fitness_values = []
#     for i in range(len(population)):
#         uniqueness_penalty = check_uniqueness(population[i], grid_size)
#         cage_penalty = evaluate_cage(population[i], cages)
        
#         # if not ()
#         total_penalty = uniqueness_penalty + cage_penalty
        
#         # Use the total penalty as the fitness score (negative to minimize)
#         fitness_values.append(-total_penalty)
    
#     return fitness_values

# def selection(population, fitness_values):
#     # Create a list to hold the selected population
#     selected_population = []

#     # Calculate total fitness
#     total_fitness = sum(fitness_values)

#     # Calculate selection probabilities
#     selection_probabilities = [fitness / total_fitness for fitness in fitness_values]

#     # Select individuals based on their probabilities
#     for _ in range(len(population)):
#         selected = False
#         while not selected:
#             # Randomly choose an individual index based on the probabilities
#             selected_index = random.choices(range(len(population)), weights=selection_probabilities)[0]

#             # Add the selected individual to the population
#             selected_population.append(population[selected_index])
#             selected = True

#     return selected_population

# def crossover(parent1, parent2):
#     length = len(parent1)
    
#     # No crossover is possible for short parents or if either parent is empty
#     if length <= 2 or len(parent2) == 0:
#         return parent1.copy(), parent2.copy()

#     position = random.randint(1, length - 1)

#     child1_list = list(parent1[:position]) + list(parent2[position:])
#     child2_list = list(parent2[:position]) + list(parent1[position:])
    
#     return child1_list, child2_list


# def uniform_crossover(parent1, parent2, probability=0.5):
#     # Ensure parents have the same length
#     assert len(parent1) == len(parent2), "Parents must have the same length"

#     # Create two child individuals by selecting genes with the specified probability
#     child1 = [gene1 if random.random() < probability else gene2 for gene1, gene2 in zip(parent1, parent2)]
#     child2 = [gene1 if random.random() < probability else gene2 for gene1, gene2 in zip(parent1, parent2)]

#     return child1, child2

# def mutate(solution, pm):
#     # Mutate a solution with a probability pm
#     mutated_solution = solution.copy()

#     grid_size = len(solution)
    
#     for i in range(grid_size):
#         for j in range(grid_size):
#             if random.random() < pm:
#                 mutated_solution[i][j] = np.random.randint(1, grid_size + 1)
    
#     return mutated_solution

# def swap_mutate(solution, pm):
#     mutated_solution = solution.copy()
#     grid_size = len(solution)
    
#     for _ in range(grid_size):  # Perform multiple swap mutations
#         if random.random() < pm:
#             # Randomly choose two distinct indices (i, j) and (k, l)
#             i, j = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
#             k, l = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            
#             # Swap values at (i, j) and (k, l)
#             mutated_solution[i][j], mutated_solution[k][l] = mutated_solution[k][l], mutated_solution[i][j]
    
#     return mutated_solution

# def kenken(population_size, max_generations, grid_size, cages, pm):
#     population = create_population(population_size, grid_size)
#     best_fitness_overall = None
#     best_solution = None
    
#     for i_gen in range(max_generations):
#         fitness_values = evaluate_fitness(population, grid_size, cages)
#         best_i = fitness_values.index(max(fitness_values))
#         best_fitness = fitness_values[best_i]
#         best_solution_gen = population[best_i]
#         best_solution_gen = np.array(best_solution_gen)
        
#         if best_fitness_overall is None or best_fitness > best_fitness_overall:
#             best_fitness_overall = best_fitness
#             best_solution = population[best_i]
        
#         # Calculate and print the best fitness for the current population
#         population_best_fitness = max(fitness_values)
#         print(f'i_gen = {i_gen:06}   Best fitness in population: {-population_best_fitness:03}')
#         print(best_solution_gen)
#         if best_fitness == 0:
#             print('Found optimal solution')
#             break
        
#         selected_pop = selection(population, fitness_values)
#         children = []
        
#         for i in range(0, len(selected_pop), 2):
#             if i + 1 < len(selected_pop):
#                 child1, child2 = crossover(selected_pop[i], selected_pop[i + 1])
#                 # Apply mutation to children
#                 child1 = swap_mutate(child1, pm)
#                 child2 = swap_mutate(child2, pm)
#                 children.append(child1)
#                 children.append(child2)

#         population = children  # Update the population with the crossover children
    
#     print()
#     print('Best solution:')
#     print(best_solution)
#     print('\r' + f' Best fitness={-best_fitness_overall:03}', end='')

#     return best_solution  # Return the best_solution array


# kenken(100, 1000, 6, cages, 0.7)