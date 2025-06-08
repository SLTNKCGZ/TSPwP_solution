import math
import random
import time
from typing import List, Tuple, Dict
from collections import defaultdict

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> int:
    """Calculate Euclidean distance between two points"""
    return round(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

def create_distance_matrix(cities: List[Tuple[int, float, float]], penalty: int) -> Dict[int, Dict[int, int]]:
    """Create a distance matrix for faster lookups"""
    n = len(cities)
    matrix = defaultdict(dict)
    
    for i in range(n):
        city1_id, x1, y1 = cities[i]
        for j in range(i+1, n):
            city2_id, x2, y2 = cities[j]
            dist = euclidean_distance((x1, y1), (x2, y2))
            if dist < penalty:  # Only store distances less than penalty
                matrix[city1_id][city2_id] = dist
                matrix[city2_id][city1_id] = dist
    
    return matrix

def calculate_tour_cost(tour: List[int], distance_matrix: Dict[int, Dict[int, int]], penalty: int, unvisited: List[int]) -> int:
    """Calculate total cost of a tour including penalties for unvisited cities"""
    total_cost = 0
    
    # Calculate cost of visited cities (including return to start)
    for i in range(len(tour)-1):
        current_city = tour[i]
        next_city = tour[i+1]
        
        # If cities are connected in distance matrix
        if current_city in distance_matrix and next_city in distance_matrix[current_city]:
            total_cost += distance_matrix[current_city][next_city]
        else:
            # If cities are not connected, add penalty
            total_cost += penalty
    
    # Add penalty for unvisited cities
    total_cost += len(unvisited) * penalty
    
    return total_cost

def get_neighbor_solution(tour: List[int], unvisited: List[int], distance_matrix: Dict[int, Dict[int, int]], penalty: int) -> Tuple[List[int], List[int]]:
    """Generate a neighbor solution using multiple strategies with focus on visiting more cities"""
    new_tour = tour.copy()
    new_unvisited = unvisited.copy()
    
    # Choose a strategy based on current tour length and unvisited cities
    strategy = random.random()
    
    if strategy < 0.3 and len(new_tour) > 3:
        # 2-opt move (don't touch the start/end city)
        i = random.randint(1, len(new_tour)-2)
        j = random.randint(i+1, len(new_tour)-1)
        new_tour[i:j] = new_tour[i:j][::-1]
    elif strategy < 0.5 and len(new_tour) > 4:
        # 3-opt move
        i = random.randint(1, len(new_tour)-3)
        j = random.randint(i+1, len(new_tour)-2)
        k = random.randint(j+1, len(new_tour)-1)
        
        # Try different 3-opt combinations
        opt_type = random.randint(0, 3)
        if opt_type == 0:
            # Reverse segment i to j
            new_tour[i:j] = new_tour[i:j][::-1]
        elif opt_type == 1:
            # Reverse segment j to k
            new_tour[j:k] = new_tour[j:k][::-1]
        elif opt_type == 2:
            # Reverse both segments
            new_tour[i:j] = new_tour[i:j][::-1]
            new_tour[j:k] = new_tour[j:k][::-1]
        else:
            # Swap segments
            new_tour[i:k] = new_tour[j:k] + new_tour[i:j]
    elif strategy < 0.8 and new_unvisited:
        # Smart insertion of unvisited cities
        if len(new_tour) > 2:
            # Find the best position to insert each unvisited city
            best_city = None
            best_pos = None
            best_cost_increase = float('inf')
            
            for city in new_unvisited:
                for pos in range(1, len(new_tour)):
                    # Calculate cost increase of inserting city at this position
                    if pos < len(new_tour)-1:
                        prev_city = new_tour[pos-1]
                        next_city = new_tour[pos]
                        
                        # Check if all cities are connected
                        if (prev_city in distance_matrix and city in distance_matrix[prev_city] and
                            city in distance_matrix and next_city in distance_matrix[city]):
                            
                            old_cost = distance_matrix[prev_city][next_city]
                            new_cost = (distance_matrix[prev_city][city] + 
                                      distance_matrix[city][next_city])
                            cost_increase = new_cost - old_cost
                            
                            if cost_increase < best_cost_increase:
                                best_cost_increase = cost_increase
                                best_city = city
                                best_pos = pos
            
            if best_city is not None:
                new_tour.insert(best_pos, best_city)
                new_unvisited.remove(best_city)
    else:
        # Remove a city and try to insert an unvisited city
        if len(new_tour) > 3 and new_unvisited:
            remove_pos = random.randint(1, len(new_tour)-2)
            removed_city = new_tour.pop(remove_pos)
            
            # Try to insert an unvisited city
            if new_unvisited:
                city = random.choice(new_unvisited)
                insert_pos = random.randint(1, len(new_tour)-1)
                
                # Check if the insertion is valid
                if insert_pos < len(new_tour):
                    prev_city = new_tour[insert_pos-1]
                    next_city = new_tour[insert_pos]
                    
                    if (prev_city in distance_matrix and city in distance_matrix[prev_city] and
                        city in distance_matrix and next_city in distance_matrix[city]):
                        new_tour.insert(insert_pos, city)
                        new_unvisited.remove(city)
                        new_unvisited.append(removed_city)
    
    return new_tour, new_unvisited

def simulated_annealing(cities: List[Tuple[int, float, float]], penalty: int, distance_matrix: Dict[int, Dict[int, int]], 
                       initial_tour: List[int], initial_unvisited: List[int], 
                       initial_temp: float = 200.0, cooling_rate: float = 0.98, 
                       iterations_per_temp: int = 200) -> Tuple[List[int], int, List[int]]:
    """Optimized Simulated Annealing algorithm for TSPWP with focus on visiting more cities"""
    current_tour = initial_tour.copy()
    current_unvisited = initial_unvisited.copy()
    current_cost = calculate_tour_cost(current_tour, distance_matrix, penalty, current_unvisited)
    
    best_tour = current_tour.copy()
    best_unvisited = current_unvisited.copy()
    best_cost = current_cost
    
    temperature = initial_temp
    no_improvement_count = 0
    max_no_improvement = 50
    
    while temperature > 0.1 and no_improvement_count < max_no_improvement:
        improvements = 0
        total_attempts = 0
        
        for _ in range(iterations_per_temp):
            # Generate neighbor solution
            new_tour, new_unvisited = get_neighbor_solution(current_tour, current_unvisited, distance_matrix, penalty)
            new_cost = calculate_tour_cost(new_tour, distance_matrix, penalty, new_unvisited)
            
            # Calculate cost difference
            delta_cost = new_cost - current_cost
            
            # Accept new solution if it's better or with probability based on temperature
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
                current_tour = new_tour
                current_unvisited = new_unvisited
                current_cost = new_cost
                
                if delta_cost < 0:
                    improvements += 1
                
                # Update best solution if needed
                if current_cost < best_cost:
                    best_tour = current_tour.copy()
                    best_unvisited = current_unvisited.copy()
                    best_cost = current_cost
                    no_improvement_count = 0
            
            total_attempts += 1
        
        # Adaptive cooling based on improvement rate
        improvement_rate = improvements / total_attempts if total_attempts > 0 else 0
        if improvement_rate < 0.1:
            temperature *= cooling_rate
            no_improvement_count += 1
        else:
            # If we're finding good solutions, cool down more slowly
            temperature *= (cooling_rate + 0.05)
    
    return best_tour, best_cost, best_unvisited

def read_input(filename: str) -> Tuple[int, List[Tuple[int, float, float]]]:
    """Read input file"""
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    penalty = int(lines[0])
    cities = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 3:
            id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            cities.append((id, x, y))
    
    return penalty, cities

def write_solution(filename: str, tour: List[int], total_cost: int, unvisited: List[int]):
    """Write solution to file"""
    with open(filename, 'w') as f:
        f.write(f"{total_cost}\n")
        f.write(f"{len(tour)}\n")
        f.write(" ".join(map(str, tour)) + "\n")
        f.write(f"{len(unvisited)}\n")
        if unvisited:
            f.write(" ".join(map(str, unvisited)) + "\n")

def create_initial_population(cities: List[Tuple[int, float, float]], penalty: int, distance_matrix: Dict[int, Dict[int, int]], population_size: int = 10) -> List[Tuple[List[int], List[int]]]:
    """Create initial population using different starting cities"""
    population = []
    
    # Try different starting cities
    for _ in range(population_size):
        # Randomly select a starting city
        start_city = random.choice([city[0] for city in cities])
        tour = [start_city]
        unvisited = [city[0] for city in cities if city[0] != start_city]
        
        while unvisited:
            current_city = tour[-1]
            best_city = None
            best_cost = float('inf')
            
            # Find nearest unvisited city with distance less than penalty
            for next_city in unvisited:
                if current_city in distance_matrix and next_city in distance_matrix[current_city]:
                    dist = distance_matrix[current_city][next_city]
                    if dist < penalty:
                        # Calculate potential cost including penalty for remaining cities
                        remaining_penalty = (len(unvisited) - 1) * penalty
                        total_cost = dist + remaining_penalty
                        if total_cost < best_cost:
                            best_city = next_city
                            best_cost = total_cost
            
            if best_city is None:
                # Try to find any city that can be visited
                for next_city in unvisited:
                    if current_city in distance_matrix and next_city in distance_matrix[current_city]:
                        dist = distance_matrix[current_city][next_city]
                        if dist < best_cost:
                            best_city = next_city
                            best_cost = dist
            
            if best_city is None:
                break
            
            tour.append(best_city)
            unvisited.remove(best_city)
        
        # Add start city at the end if not already there
        if tour[-1] != tour[0]:
            tour.append(tour[0])
        
        population.append((tour, unvisited))
    
    return population

def genetic_algorithm(cities: List[Tuple[int, float, float]], penalty: int, distance_matrix: Dict[int, Dict[int, int]], 
                     population: List[Tuple[List[int], List[int]]], generations: int = 50) -> Tuple[List[int], int, List[int]]:
    """Genetic algorithm for TSPWP"""
    best_tour = None
    best_unvisited = None
    best_cost = float('inf')
    
    for generation in range(generations):
        # Evaluate fitness of each solution
        fitness_scores = []
        for tour, unvisited in population:
            cost = calculate_tour_cost(tour, distance_matrix, penalty, unvisited)
            fitness_scores.append((cost, tour, unvisited))
        
        # Sort by fitness (lower cost is better)
        fitness_scores.sort()
        
        # Update best solution
        if fitness_scores[0][0] < best_cost:
            best_cost = fitness_scores[0][0]
            best_tour = fitness_scores[0][1]
            best_unvisited = fitness_scores[0][2]
        
        # Create new population
        new_population = []
        
        # Elitism: Keep the best solutions
        elite_count = max(2, len(population) // 5)
        new_population.extend([(tour, unvisited) for _, tour, unvisited in fitness_scores[:elite_count]])
        
        # Create offspring through crossover and mutation
        while len(new_population) < len(population):
            # Select parents using tournament selection
            parent1 = random.choice(fitness_scores[:len(fitness_scores)//2])[1]
            parent2 = random.choice(fitness_scores[:len(fitness_scores)//2])[1]
            
            # Crossover
            if len(parent1) > 3 and len(parent2) > 3:
                # Order crossover
                start = random.randint(1, len(parent1)-2)
                end = random.randint(start+1, len(parent1)-1)
                
                # Create child by copying segment from parent1
                child = parent1[start:end]
                
                # Fill remaining positions from parent2
                remaining = [city for city in parent2 if city not in child]
                child = remaining[:start] + child + remaining[start:]
                
                # Mutation
                if random.random() < 0.1:  # 10% mutation rate
                    i = random.randint(1, len(child)-2)
                    j = random.randint(1, len(child)-2)
                    child[i], child[j] = child[j], child[i]
                
                # Calculate unvisited cities
                unvisited = [city[0] for city in cities if city[0] not in child]
                
                new_population.append((child, unvisited))
        
        population = new_population
    
    return best_tour, best_cost, best_unvisited

def main():
    start_time = time.time() * 1000
    
    # Read input
    penalty, cities = read_input("input_1.txt")
    
    # Create distance matrix
    distance_matrix = create_distance_matrix(cities, penalty)
    
    # Create initial population
    population = create_initial_population(cities, penalty, distance_matrix, population_size=10)
    
    # Run genetic algorithm
    genetic_tour, genetic_cost, genetic_unvisited = genetic_algorithm(
        cities, penalty, distance_matrix, population, generations=50
    )
    
    # Run simulated annealing on the best solution from genetic algorithm
    final_tour, final_cost, final_unvisited = simulated_annealing(
        cities, penalty, distance_matrix,
        genetic_tour, genetic_unvisited,
        initial_temp=200.0,
        cooling_rate=0.98,
        iterations_per_temp=200
    )
    
    # Write solution
    write_solution("output_1.txt", final_tour, final_cost, final_unvisited)
    
    # Print statistics
    end_time = time.time() * 1000
    print(f"\nFinal Statistics:")
    print(f"Total cost: {final_cost}")
    print(f"Number of cities visited: {len(final_tour)}")
    print(f"Number of cities skipped: {len(final_unvisited)}")
    print(f"Tour: {final_tour}")
    print(f"Execution time: {end_time - start_time:.2f} milliseconds")

if __name__ == "__main__":
    main()