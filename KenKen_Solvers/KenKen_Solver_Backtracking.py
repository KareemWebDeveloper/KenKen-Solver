import numpy as np

def calculate_cage_result(cage_values, operation):
    """Calculates the result of a cage operation."""
    if operation == "+":
        return sum(cage_values)
    elif operation == "-":
        if len(cage_values) < 2:
            return 0  # Invalid operation
        return abs(cage_values[0] - sum(cage_values[1:]))
    elif operation == "*":
        result = 1
        for value in cage_values:
            result *= value
        return result
    elif operation == "/":
        if len(cage_values) < 2:
            return 0  # Invalid operation
        cage_values = sorted(cage_values)
        result = cage_values[-1]
        for value in cage_values[:-1]:
            result /= value
        return int(result)
    else:
        raise ValueError("Unsupported operation: {}".format(operation))

def is_valid(grid, row, col, num):
    """Checks if placing num at grid[row][col] is valid."""
    # Check row and column for duplicates
    if num in grid[row]:
        return False
    if num in grid[:, col]:
        return False
    return True

def is_cage_valid(grid, cages):
    """Checks if all cages in the grid are valid."""
    for cage in cages:  # Iterate over Cage objects
        operation = cage.operation
        target_value = cage.target_value
        cells = cage.cells
        cage_values = [grid[i][j] for i, j in cells]
        
        # Only check cages that are fully filled
        if all(value != 0 for value in cage_values):
            cage_result = calculate_cage_result(cage_values, operation)
            if cage_result != target_value:
                return False
    return True

def solve_kenken(grid_size, cages):
    """Solves the KenKen puzzle using backtracking."""
    
    # Initialize an empty grid with zeros
    grid = np.zeros((grid_size, grid_size), dtype=int)

    def backtrack():
        # Find the first empty cell (denoted by 0)
        for row in range(grid_size):
            for col in range(grid_size):
                if grid[row][col] == 0:
                    # Try placing numbers from 1 to grid_size
                    for num in range(1, grid_size + 1):
                        if is_valid(grid, row, col, num):
                            grid[row][col] = num  # Place the number

                            # Check if the current state is valid
                            if is_cage_valid(grid, cages):
                                if backtrack():  # Recur to continue solving
                                    return True

                            grid[row][col] = 0  # Reset on backtrack

                    return False  # No valid number found; trigger backtrack

        return True  # All cells filled successfully

    if backtrack():
        return grid
    else:
        return None  # No solution found

# Example usage with cages defined previously
cages = [
    ('+', 7, [(0, 0), (0, 1)]),
    ('*', 15, [(0, 2), (0, 3), (1, 2)]),
    ('*', 10, [(0, 4), (1, 4), (2,4)]), 
    ('*', 4, [(1, 0), (2, 0)]),
    # ('-', 2, [(1, 1), (2, 1)]),
    # ('+', 9, [(1, 3), (2, 3), (3,3)]),
    # ('-', 3 ,[(2 ,2), (3 ,2)]),
    # ('/', 2 ,[(3 ,0), (3 ,1)]),
    # ('*',12 ,[(3 ,4), (4 ,4)]),
    # ('+',7 ,[(4 ,0), (4 ,1)]),
    # ('-',3 ,[(4 ,2), (4 ,3)])
]

# Solve the KenKen puzzle with a specified grid size and cages
# grid_size = 5
# solution = solve_kenken(grid_size=grid_size, cages=cages)

# if solution is not None:
#     print("Solution found:")
#     print(solution)
# else:
#     print("No solution exists.")
