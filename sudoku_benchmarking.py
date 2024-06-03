#Run this file to compare the performances of the different GA configurations
#It can take between 5 and 10 minutes
#It's possible to adjust the number of runs


import random
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

# fitness Function
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

# generate random solution
def generate_random_solution(puzzle):
    solution = copy.deepcopy(puzzle)
    for i in range(9):
        available_numbers = [n for n in range(1, 10) if n not in solution[i]]
        for j in range(9):
            if solution[i][j] == 0:
                solution[i][j] = random.choice(available_numbers)
                available_numbers.remove(solution[i][j])
    return solution

# tournament Selection
def tournament_selection(population, fitness_values, tournament_size):
    selected_parents = []
    while len(selected_parents) < len(population):
        tournament = random.sample(range(len(population)), tournament_size)
        best = max(tournament, key=lambda i: fitness_values[i])
        selected_parents.append(population[best])
    return selected_parents

# roulette Wheel Selection
def roulette_wheel_selection(population, fitness_values, num_parents):
    max_fitness = sum(fitness_values)
    selection_probs = [fitness / max_fitness for fitness in fitness_values]
    selected_parents = random.choices(population, weights=selection_probs, k=num_parents)
    return selected_parents

# crossover
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

# Uniform Crossover
def uniform_crossover(parent1, parent2):
    offspring = [[parent1[i][j] if random.random() < 0.5 else parent2[i][j] for j in range(9)] for i in range(9)]
    return offspring

# Mutation- Swap
def mutate(solution, puzzle, mutation_rate):
    mutated_solution = copy.deepcopy(solution)
    for _ in range(mutation_rate):
        row = random.randint(0, 8)
        col1, col2 = random.sample(range(9), 2)
        if puzzle[row][col1] == 0 and puzzle[row][col2] == 0:
            mutated_solution[row][col1], mutated_solution[row][col2] = mutated_solution[row][col2], mutated_solution[row][col1]
    return mutated_solution

# Alternative Mutation Method- Random Resetting
def alternative_mutate(solution, puzzle, mutation_rate):
    mutated_solution = copy.deepcopy(solution)
    for _ in range(mutation_rate):
        row = random.randint(0, 8)
        col = random.randint(0, 8)
        if puzzle[row][col] == 0:
            new_value = random.randint(1, 9)
            mutated_solution[row][col] = new_value
    return mutated_solution

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

# Base Genetic Algorithm
def genetic_algorithm(puzzle, selection_method, crossover_method, mutation_method, population_size=1900, tournament_size=60, num_generations=350, mutation_rate=11, elitism=True):
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

        selected_parents = selection_method(population, fitness_values, tournament_size)
        offspring = []
        for i in range(0, len(selected_parents), 2):
            if i + 1 < len(selected_parents):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1]
                child = crossover_method(parent1, parent2)
                offspring.append(mutation_method(child, puzzle, mutation_rate))
        
        if elitism:
            best_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)[:tournament_size]
            elites = [population[i] for i in best_indices]
            population = offspring + elites
        else:
            population = offspring

        if generation % 100 == 0:
            population.extend([generate_random_solution(puzzle) for _ in range(tournament_size)])

    return best_solution, best_fitness_value, generation


# Run and collect results for each GA variant

def run_ga(ga_function, puzzle, num_runs=20):
    results = {
        "computational_cost": [],
        "fitness_maxed_count": 0,
        "generations_to_solution": [],
        "average_fitness_value": []
    }

    for _ in range(num_runs):
        start_time = time.time()
        solution, fitness_value, generations = ga_function(puzzle)
        end_time = time.time()
        
        computational_cost = end_time - start_time
        fitness_maxed = 1 if fitness_value == 243 else 0
        
        results["computational_cost"].append(computational_cost)
        results["fitness_maxed_count"] += fitness_maxed
        results["generations_to_solution"].append(generations)
        results["average_fitness_value"].append(fitness_value)
    
    return results

# run and collect results for all GAs

def collect_all_results(puzzle):
    all_results = {}

    # Base GA
    def base_ga(puzzle):
        return genetic_algorithm(puzzle, tournament_selection, crossover, mutate)

    # GA with roulette wheel selection
    def roulette_wheel_ga(puzzle):
        return genetic_algorithm(puzzle, roulette_wheel_selection, crossover, mutate)
    
    # GA with roulette wheel selection without elitism
    def roulette_wheel_no_elitism_ga(puzzle):
        return genetic_algorithm(puzzle, roulette_wheel_selection, crossover, mutate, elitism=False)

    # GA with alternative mutation
    def alternative_mutation_ga(puzzle):
        return genetic_algorithm(puzzle, tournament_selection, crossover, alternative_mutate)
    
    # GA with uniform crossover
    def uniform_crossover_ga(puzzle):
        return genetic_algorithm(puzzle, tournament_selection, uniform_crossover, mutate)
    
    all_results["base"] = run_ga(base_ga, puzzle)
    all_results["roulette_wheel"] = run_ga(roulette_wheel_ga, puzzle)
    all_results["roulette_wheel_no_elitism"] = run_ga(roulette_wheel_no_elitism_ga, puzzle)
    all_results["alternative_mutation"] = run_ga(alternative_mutation_ga, puzzle)
    all_results["uniform_crossover"] = run_ga(uniform_crossover_ga, puzzle)
    
    return all_results

# generate the summary table

def generate_summary_table(results):
    categories = ["base", "roulette_wheel", "roulette_wheel_no_elitism", "alternative_mutation", "uniform_crossover"]
    metrics = ["computational_cost", "fitness_maxed_count", "generations_to_solution", "average_fitness_value"]
    summary_table = {}

    for category in categories:
        summary_table[category] = {}
        for metric in metrics:
            if metric == "fitness_maxed_count":
                summary_table[category][metric] = results[category][metric]
            elif metric == "generations_to_solution":
                summary_table[category][metric] = {
                    "mean": np.mean(results[category][metric])
                }
            else:
                summary_table[category][metric] = {
                    "mean": np.mean(results[category][metric])
                }
    
    return summary_table

# plot average convergence

def plot_average_convergence(results):
    plt.figure(figsize=(10, 6))
    
    for category, data in results.items():
        average_generations = np.mean(data["generations_to_solution"])
        plt.plot(range(len(data["generations_to_solution"])), data["generations_to_solution"], label=f'{category} (avg: {average_generations:.2f})')
    
    plt.xlabel('Run')
    plt.ylabel('Generations to Solution')
    plt.title('Average Convergence Rate')
    plt.legend()
    plt.show()

# Example of running the comparison and generating the summary table

# Generate a Sudoku puzzle
puzzle = generate_sudoku_puzzle()

# Print evaluation message
print("Evaluation in progress...")

# Collect results from all GAs
results = collect_all_results(puzzle)

# Generate summary table
summary_table = generate_summary_table(results)

# Print the summary table
print("| Category                    | Computational Cost (avg per run- seconds) | Max Fitness Count | Generations to Solution (avg per run) | Average Fitness Value (avg per run) |")
print("|-----------------------------|--------------------------------------|----------------------------|--------------------------------|-----------------------------|")
for category, metrics in summary_table.items():
    print(f"| {category:<27} | {metrics['computational_cost']['mean']:.2f}                        | {metrics['fitness_maxed_count']}                   | {metrics['generations_to_solution']['mean']:.2f}                   | {metrics['average_fitness_value']['mean']:.2f}                |")

# Plot average convergence
#plot_average_convergence(results)