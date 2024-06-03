#Explanation of Changes
#Mutation Method: Instead of swapping two values in a row, we now change a random empty cell in the puzzle to a new random number between 1 and 9. This change helps explore different potential solutions more effectively.
#Diversity Preservation: Introduced new random solutions periodically to maintain genetic diversity.
#Population Management: Ensured that the population size remains constant by adding new random solutions and then maintaining the population size.
#This method of mutation aims to introduce new genetic material into the population, thereby improving the chances of finding a valid Sudoku solution.


import random
import copy

# Fitness Function
def fitness_function(solution):
    fitness = 0
    for row in solution:
        fitness += len(set(row)) - row.count(0)
    for col in range(9):
        column = [solution[row][col] for row in range(9)]
        fitness += len(set(column)) - column.count(0)
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = [solution[row][col] for row in range(i, i+3) for col in range(j, j+3)]
            fitness += len(set(box)) - box.count(0)
    return fitness

# Generate a random solution
def generate_random_solution(puzzle):
    solution = copy.deepcopy(puzzle)
    for i in range(9):
        available_numbers = [n for n in range(1, 10) if n not in solution[i]]
        for j in range(9):
            if solution[i][j] == 0:
                solution[i][j] = random.choice(available_numbers)
                available_numbers.remove(solution[i][j])
    return solution

# Tournament Selection
def tournament_selection(population, tournament_size):
    selected_parents = []
    while len(selected_parents) < len(population):
        actual_tournament_size = min(tournament_size, len(population))
        tournament = random.sample(population, actual_tournament_size)
        winner = max(tournament, key=fitness_function)
        selected_parents.append(winner)
    return selected_parents

# Crossover
def crossover(parent1, parent2):
    row_index = random.randint(0, 8)
    col_index = random.randint(0, 8)
    offspring = []
    for i in range(9):
        if i < row_index:
            offspring.append(parent1[i][:])
        elif i == row_index:
            offspring.append(parent1[i][:col_index] + parent2[i][col_index:])
        else:
            offspring.append(parent2[i][:])
    return offspring

# New Mutation Method
def mutate(solution, puzzle, mutation_rate):
    mutated_solution = copy.deepcopy(solution)
    for _ in range(mutation_rate):
        row = random.randint(0, 8)
        col = random.randint(0, 8)
        if puzzle[row][col] == 0:
            new_value = random.randint(1, 9)
            mutated_solution[row][col] = new_value
    return mutated_solution

# Print Sudoku
def print_sudoku(puzzle):
    for row in puzzle:
        print(" ".join(map(str, row)))

# Generate a fully solved Sudoku board
def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

def solve_sudoku(board):
    empty = find_empty(board)
    if not empty:
        return True
    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0
    return False

def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

def generate_full_board():
    board = [[0 for _ in range(9)] for _ in range(9)]
    solve_sudoku(board)
    return board

def remove_numbers(board, num_to_remove=40):
    puzzle = copy.deepcopy(board)
    for _ in range(num_to_remove):
        row, col = random.randint(0, 8), random.randint(0, 8)
        while puzzle[row][col] == 0:
            row, col = random.randint(0, 8), random.randint(0, 8)
        puzzle[row][col] = 0
    return puzzle

def generate_sudoku_puzzle():
    full_board = generate_full_board()
    puzzle = remove_numbers(full_board)
    return puzzle

# Genetic Algorithm
def genetic_algorithm(puzzle, population_size=1900, tournament_size=60, num_generations=10, mutation_rate=11):
    best_fitness_value = float('-inf')
    best_solution = None
    population = [generate_random_solution(puzzle) for _ in range(population_size)]

    for generation in range(num_generations):
        fitness_values = [fitness_function(solution) for solution in population]
        current_best_fitness_value = max(fitness_values)
        if current_best_fitness_value > best_fitness_value:
            best_fitness_value = current_best_fitness_value
            best_solution = population[fitness_values.index(best_fitness_value)]
        if best_fitness_value == 243:  # Optimal fitness for a complete valid Sudoku solution
            break

        print(f"Generation {generation + 1} - Best Solution (Fitness Value: {best_fitness_value}):")
        print_sudoku(best_solution)
        print()

        selected_parents = tournament_selection(population, tournament_size)
        offspring = []
        for i in range(0, len(selected_parents), 2):
            if i + 1 < len(selected_parents):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1]
                child = crossover(parent1, parent2)
                offspring.append(mutate(child, puzzle, mutation_rate))
        
        # Preserve genetic diversity by introducing new random solutions periodically
        if generation % 100 == 0:
            population.extend([generate_random_solution(puzzle) for _ in range(tournament_size)])

        # Ensure the population size remains constant
        population = offspring + random.sample(population, population_size - len(offspring))

    print(f"Final Best Solution (Fitness Value: {best_fitness_value}):")
    print_sudoku(best_solution)
    return best_solution, best_fitness_value

# Generate and solve a Sudoku puzzle
puzzle = generate_sudoku_puzzle()
print("Generated Sudoku Puzzle:")
print_sudoku(puzzle)

solution, fitness_value = genetic_algorithm(puzzle)
print("Final Solution (Fitness Value: {}):".format(fitness_value))
print_sudoku(solution)
