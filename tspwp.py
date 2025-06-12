import math
import random
import time
from typing import List, Tuple, Dict, Set
from collections import defaultdict, deque
from functools import lru_cache

# --- Yeni eklenen kod: On-demand mesafe hesaplama ve cache ---
city_dict = {}

@lru_cache(maxsize=1000000)
def get_distance(city1_id: int, city2_id: int) -> int:
    c1 = city_dict[city1_id]
    c2 = city_dict[city2_id]
    return euclidean_distance((c1[1], c1[2]), (c2[1], c2[2]))


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> int:
    return round(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))


def read_input(filename: str) -> Tuple[int, List[Tuple[int, float, float]]]:
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


def create_distance_matrix(cities: List[Tuple[int, float, float]], k_neighbors: int = 20) -> Tuple[Dict[int, List[Tuple[int, int]]], int]:
    n = len(cities)
    global city_dict
    city_dict = {city[0]: city for city in cities}
    nearest_neighbors = {}

    for i in range(n):
        city1 = cities[i]
        distances = []
        for j in range(n):
            if i != j:
                city2 = cities[j]
                dist = get_distance(city1[0], city2[0])
                distances.append((city2[0], dist))
        # Sadece en yakın k komşuyu sakla
        distances.sort(key=lambda x: x[1])
        nearest_neighbors[city1[0]] = distances[:k_neighbors]

    best_start = min(cities, key=lambda city: sum(dist for _, dist in nearest_neighbors[city[0]]))[0]
    return nearest_neighbors, best_start


def greedy_tour_with_restarts(cities: List[Tuple[int, float, float]], 
                              penalty: int, 
                              nearest_neighbors: Dict[int, List[Tuple[int, int]]],
                              num_restarts: int = 10) -> Tuple[List[int], int, List[int]]:
    best_tour = []
    best_cost = float('inf')
    best_skipped = []
    city_ids = [city[0] for city in cities]
    
    for _ in range(num_restarts):
        start_city = random.choice(city_ids)
        tour = [start_city]
        unvisited = set(city_ids) - {start_city}
        total_cost = 0
        
        while unvisited:
            current = tour[-1]
            found = False
            
            for neighbor, dist in nearest_neighbors[current]:
                if neighbor in unvisited:
                    tour.append(neighbor)
                    unvisited.remove(neighbor)
                    total_cost += dist
                    found = True
                    break
            
            if not found:
                break
        
        if tour[-1] != start_city:
            try:
                return_cost = get_distance(tour[-1], start_city)
                tour.append(start_city)
                total_cost += return_cost
            except KeyError:
                for neighbor, dist in nearest_neighbors[tour[-1]]:
                    if neighbor == start_city:
                        tour.append(start_city)
                        total_cost += dist
                        break
        
        skipped_cost = len(unvisited) * penalty
        total_cost += skipped_cost
        
        if total_cost < best_cost:
            best_tour = tour
            best_cost = total_cost
            best_skipped = list(unvisited)
    
    return best_tour, best_cost, best_skipped


def try_insert_skipped(tour: List[int], skipped: List[int], penalty: int) -> List[int]:
    for city in skipped:
        best_increase = float('inf')
        best_pos = None
        for i in range(1, len(tour)):
            increase = get_distance(tour[i-1], city) + get_distance(city, tour[i]) - get_distance(tour[i-1], tour[i])
            if increase < best_increase:
                best_increase = increase
                best_pos = i
        if best_increase < penalty:
            tour.insert(best_pos, city)
    return tour


def calculate_cost(tour: List[int], 
                   penalty: int, 
                   all_cities: Set[int]) -> int:
    cost = 0
    visited = set(tour)
    
    for i in range(len(tour)-1):
        cost += get_distance(tour[i], tour[i+1])
    
    skipped = all_cities - visited
    cost += len(skipped) * penalty
    
    return cost


def two_opt_swap(tour: List[int], i: int, k: int) -> List[int]:
    return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]


def tabu_search_optimized(initial_tour: List[int], 
                         penalty: int, 
                         all_cities: Set[int],
                         max_iter: int = 1000,
                         tabu_size: int = 30) -> Tuple[List[int], int]:
    current_tour = initial_tour[:]
    best_tour = initial_tour[:]
    best_cost = calculate_cost(best_tour, penalty, all_cities)
    tabu_list = deque(maxlen=tabu_size)
    
    for _ in range(max_iter):
        best_move = None
        best_delta = 0
        
        for i in range(1, len(current_tour)-2):
            for k in range(i+1, len(current_tour)-1):
                a, b = current_tour[i-1], current_tour[i]
                c, d = current_tour[k], current_tour[k+1]
                delta = (get_distance(a, c) + get_distance(b, d)) - (get_distance(a, b) + get_distance(c, d))
                move = (current_tour[i], current_tour[k])
                
                if (move not in tabu_list) or (best_cost + delta < best_cost):
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, k, move)
        
        if best_move:
            i, k, move = best_move
            current_tour = two_opt_swap(current_tour, i, k)
            current_cost = best_cost + best_delta
            
            if current_cost < best_cost:
                best_tour = current_tour[:]
                best_cost = current_cost
            
            tabu_list.append(move)
        else:
            break
    
    return best_tour, best_cost


def simulated_annealing(initial_tour: List[int],
                       penalty: int,
                       all_cities: Set[int],
                       initial_temp: float = 10000,
                       cooling_rate: float = 0.995,
                       min_temp: float = 1) -> Tuple[List[int], int]:
    current_tour = initial_tour[:]
    current_cost = calculate_cost(current_tour, penalty, all_cities)
    best_tour = current_tour[:]
    best_cost = current_cost
    temp = initial_temp
    
    while temp > min_temp:
        i = random.randint(1, len(current_tour)-3)
        k = random.randint(i+1, len(current_tour)-2)
        
        a, b = current_tour[i-1], current_tour[i]
        c, d = current_tour[k], current_tour[k+1]
        delta = (get_distance(a, c) + get_distance(b, d)) - (get_distance(a, b) + get_distance(c, d))
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_tour = two_opt_swap(current_tour, i, k)
            current_cost += delta
            
            if current_cost < best_cost:
                best_tour = current_tour[:]
                best_cost = current_cost
        
        temp *= cooling_rate
    
    return best_tour, best_cost


def hybrid_optimization(initial_tour: List[int],
                       penalty: int,
                       all_cities: Set[int]) -> Tuple[List[int], int]:
    tabu_tour, tabu_cost = tabu_search_optimized(
        initial_tour, penalty, all_cities,
        max_iter=500, tabu_size=30
    )
    
    final_tour, final_cost = simulated_annealing(
        tabu_tour, penalty, all_cities,
        initial_temp=5000, cooling_rate=0.993
    )
    
    return final_tour, final_cost


def write_output(filename: str, tour: List[int], cost: int, all_cities: Set[int]):
    visited = set(tour)
    skipped = list(all_cities - visited)
    
    with open(filename, 'w') as f:
        f.write(f"{cost} {len(tour)} \n")
        for city in tour:
            f.write(f"{city}\n")
        f.write(" ")


# --- Dinamik k_neighbors belirleme fonksiyonu ---
def get_k_neighbors(n_cities):
    return max(5, min(30, n_cities // 20))


def main():
    start_time = time.time()
    
    penalty, cities = read_input("test-input-4.txt")
    city_ids = [city[0] for city in cities]
    all_cities = set(city_ids)
    
    k_neighbors = get_k_neighbors(len(cities))
    nearest_neighbors, best_start = create_distance_matrix(cities, k_neighbors=k_neighbors)
    
    greedy_tour, greedy_cost, skipped = greedy_tour_with_restarts(
        cities, penalty, nearest_neighbors, num_restarts=20
    )
    
    # Atlanan şehirleri tur içine ekle
    improved_tour = try_insert_skipped(greedy_tour, skipped, penalty)
    
    optimized_tour, optimized_cost = hybrid_optimization(
        improved_tour, penalty, all_cities
    )
    optimized_tour.pop()
    write_output("test-output-4.txt", optimized_tour, optimized_cost, all_cities)
    
   


if __name__ == "__main__":
    main() 