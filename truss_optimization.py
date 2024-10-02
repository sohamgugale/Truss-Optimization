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

num_nodes = len(nodes)
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

def calculate_angle(node1, node2):
    x1, y1 = nodes[node1]
    x2, y2 = nodes[node2]
    length = calculate_length(node1, node2)
    return ( (x2 - x1) / length, (y2 - y1) / length )

def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = np.random.uniform(A_MIN, A_MAX, num_members)
        population.append(individual)
    return population

def total_weight(individual):
    lengths = [calculate_length(m[0], m[1]) for m in members]
    volume = individual * np.array(lengths)
    return np.sum(volume * density)

def build_stiffness_matrix(individual):
    size = 2 * num_nodes
    K = np.zeros((size, size))
    
    for i, (n1, n2) in enumerate(members):
        L = calculate_length(n1, n2)
        cos, sin = calculate_angle(n1, n2)
        A = individual[i]
        k = (A * E) / L
        # Stiffness matrix for member i
        k_matrix = k * np.array([
            [cos*cos, cos*sin, -cos*cos, -cos*sin],
            [cos*sin, sin*sin, -cos*sin, -sin*sin],
            [-cos*cos, -cos*sin, cos*cos, cos*sin],
            [-cos*sin, -sin*sin, cos*sin, sin*sin]
        ])
        # Indices in global stiffness matrix
        index_map = [
            2*n1, 2*n1+1,
            2*n2, 2*n2+1
        ]
        for row in range(4):
            for col in range(4):
                K[index_map[row], index_map[col]] += k_matrix[row, col]
    return K

def apply_boundary_conditions(K, F, fixed_dofs):
    # Modify K and F to apply boundary conditions
    for dof in fixed_dofs:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1
        F[dof] = 0
    return K, F

def solve_displacements(K, F):
    try:
        displacements = np.linalg.solve(K, F)
        return displacements
    except np.linalg.LinAlgError:
        return None

def calculate_axial_forces(individual, displacements):
    axial_forces = []
    for i, (n1, n2) in enumerate(members):
        L = calculate_length(n1, n2)
        cos, sin = calculate_angle(n1, n2)
        A = individual[i]
        k = (A * E) / L
        index = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        u = displacements[index]
        # Calculate axial force
        force = k * ( (u[2] - u[0]) * cos + (u[3] - u[1]) * sin )
        axial_forces.append(force)
    return np.array(axial_forces)

def fitness(individual):
    try:
        # Calculate total weight
        weight = total_weight(individual)
        
        # Build stiffness matrix
        K = build_stiffness_matrix(individual)
        
        # Build load vector
        F = np.zeros(2 * num_nodes)
        for node, load in loads.items():
            F[2*node] = load[0]
            F[2*node+1] = load[1]
        
        # Define fixed degrees of freedom
        fixed_dofs = []
        for node in supports:
            fixed_dofs.extend([2*node, 2*node+1])
        
        # Apply boundary conditions
        K_mod, F_mod = apply_boundary_conditions(K, F, fixed_dofs)
        
        # Solve for displacements
        displacements = solve_displacements(K_mod, F_mod)
        if displacements is None:
            return 1e6  # Penalize infeasible solutions
        
        # Calculate axial forces
        axial_forces = calculate_axial_forces(individual, displacements)
        
        # Calculate stresses
        stresses = axial_forces / individual  # σ = F / A
        
        # Check stress constraints
        if np.any(np.abs(stresses) > sigma_max):
            penalty = np.sum(np.abs(stresses[np.abs(stresses) > sigma_max]) - sigma_max)
            return weight + penalty * 1e3  # Penalize overweight solutions
        
        return weight
    except:
        return 1e6  # Penalize infeasible solutions

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
            individual[i] = np.clip(individual[i] + np.random.normal(0, 0.0005), A_MIN, A_MAX)
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
        print(f"Generation {generation+1}: Best Fitness = {best_fitness:.2f} kg, Average Fitness = {avg_fitness:.2f} kg")

        # Elitism
        sorted_indices = np.argsort(fitnesses)
        new_population = [population[i] for i in sorted_indices[:ELITE_SIZE]]

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
    plt.ylabel('Fitness (Weight in kg)')
    plt.title('Genetic Algorithm Optimization')
    plt.legend()
    plt.savefig('fitness_history.png')
    plt.close()

    return best_individual, best_weight, fitnesses[best_index]

def analyze_optimal_design(individual):
    # Build stiffness matrix
    K = build_stiffness_matrix(individual)
    
    # Build load vector
    F = np.zeros(2 * num_nodes)
    for node, load in loads.items():
        F[2*node] = load[0]
        F[2*node+1] = load[1]
    
    # Define fixed degrees of freedom
    fixed_dofs = []
    for node in supports:
        fixed_dofs.extend([2*node, 2*node+1])
    
    # Apply boundary conditions
    K_mod, F_mod = apply_boundary_conditions(K, F, fixed_dofs)
    
    # Solve for displacements
    displacements = solve_displacements(K_mod, F_mod)
    if displacements is None:
        print("Failed to solve displacements for the optimal design.")
        return
    
    # Calculate axial forces
    axial_forces = calculate_axial_forces(individual, displacements)
    
    # Calculate stresses
    stresses = axial_forces / individual  # σ = F / A
    
    # Display results
    print("\nOptimal Design Analysis:")
    for i, (force, stress) in enumerate(zip(axial_forces, stresses)):
        print(f"Member {i+1} (Node {members[i][0]}-{members[i][1]}):")
        print(f"  Cross-sectional Area = {individual[i]:.5f} m²")
        print(f"  Axial Force = {force:.2f} N")
        print(f"  Stress = {stress/1e6:.2f} MPa")
    print(f"\nTotal Weight: {total_weight(individual):.2f} kg")

if __name__ == "__main__":
    best_design, best_weight, _ = genetic_algorithm()
    print("\nOptimal Design:")
    for i, area in enumerate(best_design):
        print(f"Member {i+1} (Node {members[i][0]}-{members[i][1]}): Area = {area:.5f} m²")
    print(f"Total Weight: {best_weight:.2f} kg")
    
    # Analyze and display detailed information about the optimal design
    analyze_optimal_design(best_design)

