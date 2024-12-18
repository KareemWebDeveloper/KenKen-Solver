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
        cages: A list of cages, each represented as an Object that contains these attributes (operation, target, cells).
        grid_size: The size of the Kenken grid.

    Returns:
        The fitness score of the individual.
    """

    fitness = 0

    # Check for uniqueness in rows and columns
    fitness += check_uniqueness(individual, grid_size)

    # Check cage constraints with a hard penalty for invalid results
    HARD_PENALTY = 100 # Large fixed penalty for constraint violation

    for cage in cages:  # Iterate over Cage objects
        operation = cage.operation
        target = cage.target_value
        cells = cage.cells

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
            performance_percentage = (1 - (best_fitness / float('inf'))) * 100 if best_fitness != 0 else 100
            return best_fitness, generation, performance_percentage, best_solution
        
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
    performance_percentage = (1 - (best_fitness / float('inf'))) * 100 if best_fitness != 0 else 100
    return best_fitness, best_generation, performance_percentage, best_solution


# def genetic_algorithm(cages, grid_size, population_size, max_generations, mutation_rate, elitism_rate, tolerance=1e-6):
#     population = [generate_individual(grid_size) for _ in range(population_size)]
#     best_fitness = float('inf')
#     start_time = time.time()  # Start timing
#     best_solution = None
#     best_generation = 0

#     for generation in range(max_generations):
#         # Calculate fitness for all individuals
#         fitness_scores = [calculate_fitness(individual, cages, grid_size) for individual in population]
        
#         # Find the best solution in the current generation
#         best_index = np.argmin(fitness_scores)
#         current_best_fitness = fitness_scores[best_index]
        
#         # Print generation stats
#         print(f"Generation {generation} - Best Fitness: {current_best_fitness}")
#         print(f"Solution:\n{population[best_index]}")

#         # Update global best solution
#         if current_best_fitness < best_fitness:
#             best_fitness = current_best_fitness
#             best_solution = population[best_index]
#             best_generation = generation
        
#         # Check for optimal solution
#         if current_best_fitness == 0:
#             print(f"Optimal solution found in generation {generation}")
#             print(f"Time taken: {time.time() - start_time:.2f} seconds")
#             print(f"Best Solution Fitness: {best_fitness}")
#             print("Optimal Solution:")
#             print(best_solution)
#             return best_solution

#         # Early termination check
#         # if abs(best_fitness - current_best_fitness) < tolerance:
#         #     print("Early termination: No significant improvement in the fitness value")
#         #     print(f"The Fitness Value : {current_best_fitness}")
#         #     print(f"Time taken: {time.time() - start_time:.2f} seconds")
#         #     print(f"Best Solution Found in Generation {best_generation}")
#         #     print(best_solution)
#         #     return best_solution

#         # Elitism: Select the best individuals
#         elite_size = int(population_size * elitism_rate)
#         elite_indices = np.argsort(fitness_scores)[:elite_size]
#         elite_individuals = [population[i] for i in elite_indices]

#         # Create a new population
#         new_population = elite_individuals.copy()
#         while len(new_population) < population_size:
#             parent1 = roulette_wheel_selection(population, fitness_scores)
#             parent2 = roulette_wheel_selection(population, fitness_scores)
#             child1, child2 = pmx_crossover(parent1, parent2)
#             child1 = mutate(child1, mutation_rate)
#             child2 = mutate(child2, mutation_rate)
#             new_population.append(child1)
#             new_population.append(child2)

#         population = new_population

#     # Final output after max generations
#     print("Max generations reached.")
#     print(f"Best Fitness Achieved: {best_fitness}")
#     print(f"Found in Generation: {best_generation}")
#     print(f"Time taken: {time.time() - start_time:.2f} seconds")
#     print("Best Solution:")
#     print(best_solution)

#     return best_solution

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
# best_solution = genetic_algorithm(cages, grid_size, population_size, max_generations, mutation_rate , elitism_rate)
