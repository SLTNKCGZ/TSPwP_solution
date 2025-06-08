import math
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import time

def euclidean_distance(p1, p2):
    return round(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

def create_distance_matrix(cities: List[Tuple[int, float, float]], penalty: int) -> Tuple[Dict[int, Dict[int, int]], Dict[int, List[Tuple[int, int]]], int]:
    """Create a distance matrix for faster lookups, only storing distances less than penalty
    Also stores nearest neighbors for each city and finds the best starting city"""
    n = len(cities)
    matrix = defaultdict(dict)
    nearest_neighbors = defaultdict(list)  # Store nearest neighbors for each city
    best_start_city = cities[0][0]  # First city's ID
    min_two_nearest_sum = float('inf')  # Track the sum of two nearest distances
    
    # Track two smallest distances for each city
    city_two_smallest = defaultdict(lambda: [float('inf'), float('inf')])
    
    # Single pass to collect distances and find best starting city
    for i in range(n):
        city1_id, x1, y1 = cities[i]
        for j in range(i+1, n):
            city2_id, x2, y2 = cities[j]
            
            # Skip if x or y difference is greater than penalty
            if abs(x1 - x2) >= penalty or abs(y1 - y2) >= penalty:
                continue
                
            dist = euclidean_distance((x1, y1), (x2, y2))
            if dist < penalty:  # Skip distances >= penalty
                # Store distance in matrix
                matrix[city1_id][city2_id] = dist
                matrix[city2_id][city1_id] = dist
                
                # Add to neighbor lists
                nearest_neighbors[city1_id].append((city2_id, dist))
                nearest_neighbors[city2_id].append((city1_id, dist))
                
                # Update two smallest distances and check for best starting city
                for city_id in [city1_id, city2_id]:
                    if dist < city_two_smallest[city_id][0]:
                        city_two_smallest[city_id][1] = city_two_smallest[city_id][0]
                        city_two_smallest[city_id][0] = dist
                    elif dist < city_two_smallest[city_id][1]:
                        city_two_smallest[city_id][1] = dist
                    
                    # Check if this city could be the best starting city
                    if city_two_smallest[city_id][1] != float('inf'):
                        two_nearest_sum = city_two_smallest[city_id][0] + city_two_smallest[city_id][1]
                        if two_nearest_sum < min_two_nearest_sum:
                            min_two_nearest_sum = two_nearest_sum
                            best_start_city = city_id
    
    return matrix, nearest_neighbors, best_start_city

def create_optimized_tour(cities: List[Tuple[int, float, float]], penalty: int, distance_matrix: Dict[int, Dict[int, int]], nearest_neighbors: Dict[int, List[Tuple[int, int]]], start_city: int) -> Tuple[List[int], int, List[int]]:
    """Create an optimized tour using a greedy approach"""
    n = len(cities)
    if n <= 1:
        return [cities[0][0]], 0, []
    
    # Initialize tour and remaining cities
    tour = [start_city]
    unvisited = set(city[0] for city in cities) - {start_city}
    total_cost = 0
    
    while unvisited:
        current_city = tour[-1]
        best_city = None
        best_cost = float('inf')
        
        # Sort neighbors only when needed
        if not nearest_neighbors[current_city]:
            nearest_neighbors[current_city].sort(key=lambda x: x[1])
        
        # Try nearest neighbors in order
        for next_city, dist in nearest_neighbors[current_city]:
            if next_city in unvisited:
                # Check if this city can reach the start city
                can_reach_start = False
                if start_city in distance_matrix[next_city]:
                    can_reach_start = distance_matrix[next_city][start_city] < penalty
                
                # If this is the last city to visit, it must be able to reach start
                if len(unvisited) == 1 and not can_reach_start:
                    continue
                
                best_city = next_city
                best_cost = dist
                break  # Take the first unvisited neighbor
        
        # If no neighbor found, backtrack
        if best_city is None:
            if len(tour) > 1:
                tour.pop()  # Remove current city
                continue
            else:
                break  # If we're back at start and no neighbors, we're done
        
        # Add best city to tour
        tour.append(best_city)
        unvisited.remove(best_city)
        total_cost += best_cost
    
    # If tour doesn't end at start, try to return
    if tour[-1] != start_city:
        if start_city in distance_matrix[tour[-1]]:
            return_dist = distance_matrix[tour[-1]][start_city]
            if return_dist < penalty:
                tour.append(start_city)
                total_cost += return_dist
    
    # Add penalties for unvisited cities
    penalty_cost = len(unvisited) * penalty
    total_cost += penalty_cost
    
    return tour, total_cost, list(unvisited)

def read_input(filename: str) -> Tuple[int, List[Tuple[int, float, float]]]:
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
            cities.append((id, x, y))
    return penalty, cities

def write_solution_to_file(filename: str, tour: List[int], total_cost: int):
    """Write solution to file in the required format"""
    with open(filename, 'w') as f:
        f.write(f"{total_cost} {len(tour)}\n")
        for city_id in tour:
            f.write(f"{city_id}\n")
        f.write("\n")

def main():
    start_time = time.time()*1000
    # Read input
    penalty, cities = read_input("input_3.txt")
    
    # Create distance matrix and find best starting city
    distance_matrix, nearest_neighbors, best_start = create_distance_matrix(cities, penalty)
    
    # Create optimized tour
    tour, total_cost, unvisited = create_optimized_tour(cities, penalty, distance_matrix, nearest_neighbors, best_start)
    
    # Write solution
    with open("output_3.txt", "w") as f:
        f.write(f"{total_cost}\n")
        f.write(f"{len(tour)}\n")
        f.write(" ".join(map(str, tour)) + "\n")
        f.write(f"{len(unvisited)}\n")
        if unvisited:
            f.write(" ".join(map(str, unvisited)) + "\n")
    
    # Print statistics
    print(f"Total cost: {total_cost}")
    print(f"Number of cities visited: {len(cities) - len(unvisited)}")
    print(f"Number of cities skipped: {len(unvisited)}")
    print(f"tour başlangıç ve bitiş noktaları:{tour[0],tour[-1]}")
    end_time = time.time()*1000
    print(f"İşlem süresi: {end_time - start_time:.2f} milliseconds") 

if __name__ == "__main__":
    main()