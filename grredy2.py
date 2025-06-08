import math
from typing import List, Tuple, Set

class City:
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y

def calculate_distance(city1: City, city2: City) -> int:
    """Calculate rounded Euclidean distance between two cities"""
    return round(math.sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2))

def calculate_tour_distance(tour: List[int], cities: List[City]) -> int:
    """Calculate total distance of a tour"""
    total_distance = 0
    for i in range(len(tour)):
        current = cities[tour[i]]
        next_city = cities[tour[(i + 1) % len(tour)]]
        total_distance += calculate_distance(current, next_city)
    return total_distance

def find_best_starting_city(cities: List[City], penalty_value: float) -> int:
    """Find the best starting city based on cost minimization"""
    n = len(cities)
    if n <= 1:
        return 0
    
    best_city = 0
    min_total_cost = float('inf')
    
    # Try each city as a starting point
    for i in range(n):
        total_cost = 0
        visited = {i}
        
        # Calculate cost of visiting nearby cities
        for j in range(n):
            if i != j:
                dist = calculate_distance(cities[i], cities[j])
                if dist < penalty_value:
                    total_cost += dist
                    visited.add(j)
        
        # Add penalties for unvisited cities
        total_cost += penalty_value * (n - len(visited))
        
        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_city = i
    
    return best_city

def create_greedy_tour(cities: List[City], penalty_value: float) -> List[int]:
    """Create a tour using greedy approach focusing on cost minimization"""
    n = len(cities)
    if n <= 1:
        return list(range(n))
    
    # Find best starting city
    start_city = find_best_starting_city(cities, penalty_value)
    
    # Initialize tour and remaining cities
    tour = [start_city]
    remaining = set(range(n)) - {start_city}
    
    # Create distance matrix for faster lookups
    distance_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dist = calculate_distance(cities[i], cities[j])
            distance_matrix[i][j] = distance_matrix[j][i] = dist
    
    # Create sorted neighbor lists for each city
    neighbor_lists = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                neighbor_lists[i].append((j, distance_matrix[i][j]))
        neighbor_lists[i].sort(key=lambda x: x[1])
    
    # Build tour
    while remaining:
        current = tour[-1]
        best_next = None
        min_cost = float('inf')
        
        # Try each remaining city
        for next_city in remaining:
            # Calculate cost of visiting this city
            visit_cost = distance_matrix[current][next_city]
            
            # If visiting is cheaper than skipping, consider it
            if visit_cost < penalty_value:
                # Calculate potential total cost
                potential_cost = visit_cost
                
                # Add a small penalty for cities that are far from their neighbors
                neighbor_avg_dist = sum(d for _, d in neighbor_lists[next_city][:5]) / 5
                potential_cost += neighbor_avg_dist * 0.1
                
                if potential_cost < min_cost:
                    min_cost = potential_cost
                    best_next = next_city
        
        # If no beneficial city found, stop
        if best_next is None:
            break
        
        # Add best city to tour
        tour.append(best_next)
        remaining.remove(best_next)
    
    return tour

def read_cities_from_file(filename: str) -> Tuple[float, List[City]]:
    """Read penalty value and cities from a file"""
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
            cities.append(City(id, x, y))
    return penalty, cities

def write_solution_to_file(filename: str, tour: List[int], cities: List[City], penalty_value: float):
    """Write solution to file in the required format"""
    # Calculate total distance
    total_distance = calculate_tour_distance(tour, cities)
    
    # Add penalties for unvisited cities
    total_cost = total_distance + penalty_value * (len(cities) - len(tour))
    
    with open(filename, 'w') as f:
        # Write total cost and number of visited cities
        f.write(f"{total_cost} {len(tour)}\n")
        
        # Write city IDs in tour order
        for city_id in tour:
            f.write(f"{cities[city_id].id}\n")
        
        # Add blank line at the end
        f.write("\n")

def main():
    # Read input
    input_filename = "input_1.txt"
    output_filename = "output_1.txt"
    
    penalty_value, cities = read_cities_from_file(input_filename)
    
    # Create tour using greedy approach
    tour = create_greedy_tour(cities, penalty_value)
    
    # Write solution
    write_solution_to_file(output_filename, tour, cities, penalty_value)
    
    # Print summary
    total_distance = calculate_tour_distance(tour, cities)
    total_cost = total_distance + penalty_value * (len(cities) - len(tour))
    print(f"Total cost: {total_cost}")
    print(f"Number of cities visited: {len(tour)}")
    print(f"Number of cities skipped: {len(cities) - len(tour)}")

if __name__ == "__main__":
    main() 
