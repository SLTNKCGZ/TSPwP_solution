from typing import List, Tuple, Dict

def create_optimized_tour(cities: List[Tuple[int, float, float]], penalty: int, distance_matrix: Dict[int, Dict[int, int]], start_city: int) -> Tuple[List[int], int, List[int]]:
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
        
        # Try all unvisited cities
        for next_city in unvisited:
            if next_city in distance_matrix[current_city]:
                dist = distance_matrix[current_city][next_city]
                if dist < best_cost:
                    # Check if this city can reach the start city
                    can_reach_start = False
                    if start_city in distance_matrix[next_city]:
                        can_reach_start = distance_matrix[next_city][start_city] < penalty
                    
                    # If this is the last city to visit, it must be able to reach start
                    if len(unvisited) == 1 and not can_reach_start:
                        continue
                    
                    best_city = next_city
                    best_cost = dist
        
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
        
        # Print the current tour
        print(f"Current tour: {tour}")
    
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