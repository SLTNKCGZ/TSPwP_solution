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
    two_cities_1=0
    two_cities_2=0
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
                            two_cities_1 =city_two_smallest[city_id][0]
                            two_cities_2=city_two_smallest[city_id][1]
                            min_two_nearest_sum = two_nearest_sum
                            best_start_city = city_id
    print(f"two_cities_1:{two_cities_1},two_cities_2:{two_cities_2}")                        
    for city_id in nearest_neighbors:
        nearest_neighbors[city_id].sort(key=lambda x: x[1])
    return matrix, nearest_neighbors, best_start_city

def create_optimized_tour(cities: List[Tuple[int, float, float]], penalty: int, distance_matrix: Dict[int, Dict[int, int]], nearest_neighbors: Dict[int, List[Tuple[int, int]]], start_city: int) -> Tuple[List[int], int, List[int]]:
    """Create an optimized tour using a greedy approach"""
    n = len(cities)
    if n <= 1:
        print("1 sayılı")
        return [cities[0][0]], 0, []
    
    # Initialize tour and remaining cities
    tour = [start_city]
    unvisited = set(city[0] for city in cities) - {start_city}
    total_cost = 0
    last_cost = 0  # Track the cost of the last move
    
    while True:  # Continue until we return to start city
        current_city = tour[-1]
        best_city = None
        best_cost = float('inf')
        
        # Try nearest neighbors first (they are already sorted by distance)
        for next_city, dist in nearest_neighbors[current_city]:
            if(next_city==start_city):
                best_city=next_city
                best_cost = dist
            elif next_city in unvisited and dist > 0:
                best_city = next_city
                best_cost = dist
                break  # Take the first valid neighbor since they are sorted
        print(tour)    
        print(nearest_neighbors[current_city])
        
        # If no valid city found
        if best_city is None:
            # If we're at start city and no neighbors, we're done
            if current_city == start_city:
                break
            # Otherwise, backtrack to previous city
            if len(tour) > 1:
                # Remove current city and subtract its cost
                last_city = tour.pop()
                total_cost -= last_cost
                unvisited.add(last_city)
                # Remove the last city from the neighbors of the previous city
                if tour[-1] in nearest_neighbors:
                    nearest_neighbors[tour[-1]] = [n for n in nearest_neighbors[tour[-1]] if n[0] != last_city]
                continue
            else:
                break  # If we're back at start and no neighbors, we're done
        
        # Add best city to tour
        tour.append(best_city)
        if best_city != start_city:
            unvisited.remove(best_city)
        last_cost = best_cost  # Save the cost of this move
        total_cost += best_cost
        
        # If we reached the start city, we're done
        if best_city == start_city:
            break
        
        # Print the current tour and cost
        #print(f"Current tour: {tour}, Cost so far: {total_cost}")
    
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
    penalty, cities = read_input("input_1.txt")
    
    # Create distance matrix and find best starting city
    distance_matrix, nearest_neighbors, best_start = create_distance_matrix(cities, penalty)
    
    # Create optimized tour
    tour, total_cost, unvisited = create_optimized_tour(cities, penalty, distance_matrix, nearest_neighbors, best_start)
    
    # Write solution
    with open("output_1.txt", "w") as f:
        f.write(f"{total_cost}\n")
        f.write(f"{len(tour)}\n")
        f.write(" ".join(map(str, tour)) + "\n")
        f.write(f"{len(unvisited)}\n")
    
    # Print statistics
    print(f"\nFinal Statistics:")
    print(f"Total cost: {total_cost}")
    print(f"Number of cities visited: {len(tour)-1}")
    print(f"Number of cities skipped: {len(unvisited)}")
    print(f"Tour: {tour}")
    end_time = time.time()*1000
    print(f"Execution time: {end_time - start_time:.2f} milliseconds")

if __name__ == "__main__":
    main()