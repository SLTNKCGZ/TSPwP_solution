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
    """Generate a neighbor solution using 2-opt or random insertion"""
    new_tour = tour.copy()
    new_unvisited = unvisited.copy()
    
    # Randomly choose between 2-opt and random insertion
    if random.random() < 0.5 and len(new_tour) > 3:
        # 2-opt move (don't touch the start/end city)
        i = random.randint(1, len(new_tour)-2)
        j = random.randint(i+1, len(new_tour)-1)
        new_tour[i:j] = new_tour[i:j][::-1]
    elif new_unvisited:
        # Random insertion (don't insert after the last city)
        if len(new_tour) > 2:
            insert_pos = random.randint(1, len(new_tour)-1)
            city = random.choice(new_unvisited)
            new_tour.insert(insert_pos, city)
            new_unvisited.remove(city)
    
    return new_tour, new_unvisited

def simulated_annealing(cities: List[Tuple[int, float, float]], penalty: int, distance_matrix: Dict[int, Dict[int, int]], 
                       initial_tour: List[int], initial_unvisited: List[int], 
                       initial_temp: float = 100.0, cooling_rate: float = 0.95, 
                       iterations_per_temp: int = 100) -> Tuple[List[int], int, List[int]]:
    """Simulated Annealing algorithm for TSPWP"""
    current_tour = initial_tour.copy()
    current_unvisited = initial_unvisited.copy()
    current_cost = calculate_tour_cost(current_tour, distance_matrix, penalty, current_unvisited)
    
    best_tour = current_tour.copy()
    best_unvisited = current_unvisited.copy()
    best_cost = current_cost
    
    temperature = initial_temp
    
    while temperature > 0.1:
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
                
                # Update best solution if needed
                if current_cost < best_cost:
                    best_tour = current_tour.copy()
                    best_unvisited = current_unvisited.copy()
                    best_cost = current_cost
        
        # Cool down
        temperature *= cooling_rate
    
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

def create_initial_solution(cities: List[Tuple[int, float, float]], penalty: int, distance_matrix: Dict[int, Dict[int, int]]) -> Tuple[List[int], List[int]]:
    """Create initial solution using simple greedy approach"""
    n = len(cities)
    if n <= 1:
        return [cities[0][0]], []
    
    # Start with the first city
    tour = [cities[0][0]]
    unvisited = [city[0] for city in cities[1:]]
    
    while unvisited:
        current_city = tour[-1]
        best_city = None
        best_cost = float('inf')
        
        # Find nearest unvisited city
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
    
    return tour, unvisited

def main():
    start_time = time.time() * 1000
    
    # Read input
    penalty, cities = read_input("input_1.txt")
    
    # Create distance matrix
    distance_matrix = create_distance_matrix(cities, penalty)
    
    # Create initial solution
    initial_tour, initial_unvisited = create_initial_solution(cities, penalty, distance_matrix)
    
    # Run simulated annealing
    final_tour, total_cost, final_unvisited = simulated_annealing(
        cities, penalty, distance_matrix,
        initial_tour, initial_unvisited,
        initial_temp=100.0,
        cooling_rate=0.95,
        iterations_per_temp=100
    )
    
    # Write solution
    write_solution("output_1.txt", final_tour, total_cost, final_unvisited)
    
    # Print statistics
    end_time = time.time() * 1000
    print(f"\nFinal Statistics:")
    print(f"Total cost: {total_cost}")
    print(f"Number of cities visited: {len(final_tour)}")
    print(f"Number of cities skipped: {len(final_unvisited)}")
    print(f"Tour: {final_tour}")
    print(f"Execution time: {end_time - start_time:.2f} milliseconds")

if __name__ == "__main__":
    main()