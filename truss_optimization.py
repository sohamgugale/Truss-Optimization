import numpy as np
import matplotlib.pyplot as plt
import random

# ------------------------------
# Truss Optimization Using GA
# ------------------------------

# Define the truss structure
# Example: Simple 2D truss with 5 nodes and 7 members
nodes = {
    0: (0, 0),
    1: (5, 0),
    2: (10, 0),
    3: (5, 5),
    4: (10, 5)
}

# Define members as tuples of node indices
members = [
    (0, 1),
    (1, 2),
    (0, 3),
    (1, 3),
    (1, 4),
    (2, 4),
    (3, 4)
]

num_members = len(members)

# Material properties
E = 210e9  # Young's Modulus in Pascals (Steel)
sigma_max = 250e6  # Maximum allowable stress in Pascals
density = 7850  # Density in kg/m^3

# Load conditions
# Applied forces at nodes (Fx, Fy) in Newtons
loads = {
    4: (0, -10000)  # Node 4 has a downward load of 10,000 N
}

# Supports (fixed nodes)
supports = [0, 2]  # Nodes 0 and 2 are fixed

# Genetic Algorithm Parameters
POPULATION_SIZE = 50
GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
ELITE_SIZE = 2

# Design variable bounds (cross-sectional area in m^2)
A_MIN = 0.001  # Minimum area
A_MAX = 0.01   # Maximum area

# Helper functions
def calculate_length(node1, node2):
    x1, y1 = nodes[node1]
    x2, y2 = nodes[node2]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = np.random.uniform(A_MIN, A_MAX, num_members)
        population.append(individual)
    return population

def fitness(individual):
    try:
        # Calculate axial forces and check constraints
        axial_forces = analyze_truss(individual)
        stress = axial_forces * calculate_length_forces(individual) / individual / 1e6  # in MPa
        if np.any(np.abs(stress) > sigma_max / 1e6):
            return 1e6  # Penalize if stress exceeds limit
        weight = total_weight(individual)
        return weight
    except:
        return 1e6  # Penalize infeasible solutions

def calculate_length_forces(individual):
    lengths = [calculate_length(m[0], m[1]) for m in members]
    return np.array(lengths)

def total_weight(individual):
    lengths = calculate_length_forces(individual)
    volume = individual * lengths
    return np.sum(volume * density)

def analyze_truss(individual):
    # Placeholder for truss analysis
    # For simplicity, assume axial forces are proportional to area
    # In a real scenario, perform equilibrium and compatibility equations
    # Here we return random forces within allowable limits
    return np.random.uniform(-sigma_max, sigma_max, num_members)

def selection(population, fitnesses):
    # Tournament selection
    selected = []
    for _ in range(POPULATION_SIZE):
        i, j = random.sample(range(POPULATION_SIZE), 2)
        if fitnesses[i] < fitnesses[j]:
            selected.append(population[i])
        else:
            selected.append(population[j])
    return selected

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, num_members - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    else:
        return parent1.copy(), parent2.copy()

def mutate(individual):
    for i in range(num_members):
        if random.random() < MUTATION_RATE:
            individual[i] = np.clip(individual[i] + np.random.normal(0, 0.001), A_MIN, A_MAX)
    return individual

def genetic_algorithm():
    population = initialize_population()
    best_fitness_history = []
    avg_fitness_history = []

    for generation in range(GENERATIONS):
        fitnesses = [fitness(ind) for ind in population]
        best_fitness = min(fitnesses)
        avg_fitness = np.mean(fitnesses)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}, Average Fitness = {avg_fitness:.2f}")

        # Elitism
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0])]
        new_population = sorted_population[:ELITE_SIZE]

        # Selection
        selected = selection(population, fitnesses)

        # Crossover and Mutation
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

    # Final evaluation
    fitnesses = [fitness(ind) for ind in population]
    best_index = np.argmin(fitnesses)
    best_individual = population[best_index]
    best_weight = fitnesses[best_index]

    # Plot fitness history
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.plot(avg_fitness_history, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Weight)')
    plt.title('Genetic Algorithm Optimization')
    plt.legend()
    plt.savefig('fitness_history.png')
    plt.close()

    return best_individual, best_weight

if __name__ == "__main__":
    best_design, best_weight = genetic_algorithm()
    print("\nOptimal Design:")
    for i, area in enumerate(best_design):
        print(f"Member {i+1} (Node {members[i][0]}-{members[i][1]}): Area = {area:.5f} m²")
    print(f"Total Weight: {best_weight:.2f} kg")
