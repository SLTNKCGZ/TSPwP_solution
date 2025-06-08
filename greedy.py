import math
from typing import List, Tuple, Set, Dict
from collections import defaultdict

class City:
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y

def calculate_distance(city1: City, city2: City) -> int:
    """Calculate rounded Euclidean distance between two cities"""
    return round(math.sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2))

def find_best_starting_city(cities: List[City], penalty_value: float) -> int:
    """Find the best starting city that minimizes total cost"""
    n = len(cities)
    if n <= 1:
        return 0
    
    best_city = 0
    min_total_cost = float('inf')
    
    # Try each city as starting point
    for start in range(n):
        # Calculate total cost if we start from this city
        total_distance = 0
        visited = {start}
        current = start
        
        # Find nearest unvisited city until no more beneficial cities
        while True:
            best_next = None
            best_dist = float('inf')
            
            for next_city in range(n):
                if next_city not in visited:
                    dist = calculate_distance(cities[current], cities[next_city])
                    # Only consider cities that would reduce total cost
                    if dist < penalty_value and dist < best_dist:
                        best_dist = dist
                        best_next = next_city
            
            if best_next is None or best_dist >= penalty_value:
                break
                
            total_distance += best_dist
            visited.add(best_next)
            current = best_next
        
        # Calculate total cost (distance + penalties for skipped cities)
        total_cost = total_distance + penalty_value * (n - len(visited))
        
        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_city = start
    
    return best_city

def find_closest_pairs(cities: List[City], penalty_value: float) -> List[Tuple[int, int, int]]:
    """Find closest pairs between cities using divide and conquer"""
    n = len(cities)
    if n <= 1:
        return []
    
    # Sort cities by x-coordinate
    sorted_indices = sorted(range(n), key=lambda i: cities[i].x)
    
    def find_pairs_between(left: int, right: int) -> List[Tuple[int, int, int]]:
        if right - left <= 1:
            return []
        
        mid = (left + right) // 2
        mid_x = cities[sorted_indices[mid]].x
        
        # Find pairs in left and right halves
        left_pairs = find_pairs_between(left, mid)
        right_pairs = find_pairs_between(mid, right)
        
        # Find pairs that cross the middle line
        strip = []
        for i in range(left, right):
            if abs(cities[sorted_indices[i]].x - mid_x) < penalty_value:
                strip.append(sorted_indices[i])
        
        # Sort strip by y-coordinate
        strip.sort(key=lambda i: cities[i].y)
        
        # Find closest pairs in strip
        strip_pairs = []
        for i in range(len(strip)):
            # Only check next 7 points (geometric progression)
            for j in range(i+1, min(i+8, len(strip))):
                dist = calculate_distance(cities[strip[i]], cities[strip[j]])
                if dist < penalty_value:
                    strip_pairs.append((strip[i], strip[j], dist))
        
        return left_pairs + right_pairs + strip_pairs
    
    # Find all closest pairs
    pairs = find_pairs_between(0, n)
    
    # Sort pairs by distance
    return sorted(pairs, key=lambda x: x[2])

def create_tour_divide_conquer(cities: List[City], penalty_value: float) -> List[int]:
    """Create tour that minimizes total cost (distance + penalties)"""
    n = len(cities)
    if n <= 1:
        return list(range(n))
    
    # Find best starting city
    start_city = find_best_starting_city(cities, penalty_value)
    
    # Initialize tour
    tour = [start_city]
    remaining = set(range(n)) - {start_city}
    total_distance = 0
    
    while remaining:
        current = tour[-1]
        best_next = None
        best_cost = float('inf')
        
        # Find next city that minimizes total cost
        for next_city in remaining:
            dist = calculate_distance(cities[current], cities[next_city])
            # Calculate cost of visiting this city
            visit_cost = dist
            # Calculate cost of skipping this city
            skip_cost = penalty_value
            
            # If visiting is cheaper than skipping, consider it
            if visit_cost < skip_cost:
                # Calculate potential total cost
                potential_cost = total_distance + visit_cost + penalty_value * (len(remaining) - 1)
                if potential_cost < best_cost:
                    best_cost = potential_cost
                    best_next = next_city
        
        # If no beneficial city found, stop
        if best_next is None:
            break
        
        # Add best city to tour
        tour.append(best_next)
        remaining.remove(best_next)
        total_distance += calculate_distance(cities[current], cities[best_next])
    
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
    total_distance = 0
    for i in range(len(tour)):
        current = cities[tour[i]]
        next_city = cities[tour[(i + 1) % len(tour)]]
        total_distance += calculate_distance(current, next_city)
    
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
    
    # Create tour
    tour = create_tour_divide_conquer(cities, penalty_value)
    
    # Write solution
    write_solution_to_file(output_filename, tour, cities, penalty_value)
    
    # Print summary
    total_distance = 0
    for i in range(len(tour)):
        current = cities[tour[i]]
        next_city = cities[tour[(i + 1) % len(tour)]]
        total_distance += calculate_distance(current, next_city)
    
    total_cost = total_distance + penalty_value * (len(cities) - len(tour))
    print(f"Total cost: {total_cost}")
    print(f"Number of cities visited: {len(tour)}")
    print(f"Number of cities skipped: {len(cities) - len(tour)}")

if __name__ == "__main__":
    main() 
