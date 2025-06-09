import math
import random
import time
from typing import List, Tuple, Dict, Set
from collections import defaultdict, deque


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> int:
    """İki nokta arasındaki Öklid mesafesini hesaplar ve yuvarlar."""
    return round(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))


def read_input(filename: str) -> Tuple[int, List[Tuple[int, float, float]]]:
    """Girdi dosyasını okur ve ceza değeri ile şehir listesini döndürür."""
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


def create_distance_matrix(cities: List[Tuple[int, float, float]]) -> Tuple[Dict[int, Dict[int, int]], Dict[int, List[Tuple[int, int]]], int]:
    """Şehirler arası mesafe matrisini ve en yakın komşuları oluşturur."""
    n = len(cities)
    matrix = defaultdict(dict)
    nearest_neighbors = defaultdict(list)
    
    for i in range(n):
        city1 = cities[i]
        for j in range(n):
            if i != j:
                city2 = cities[j]
                dist = euclidean_distance((city1[1], city1[2]), (city2[1], city2[2]))
                matrix[city1[0]][city2[0]] = dist
                nearest_neighbors[city1[0]].append((city2[0], dist))
    
    for city_id in nearest_neighbors:
        nearest_neighbors[city_id].sort(key=lambda x: x[1])
    
    best_start = min(cities, key=lambda city: sum(dist for _, dist in nearest_neighbors[city[0]][:5]))[0]
    
    return matrix, nearest_neighbors, best_start


def greedy_tour_with_restarts(cities: List[Tuple[int, float, float]], 
                              penalty: int, 
                              distance_matrix: Dict[int, Dict[int, int]], 
                              nearest_neighbors: Dict[int, List[Tuple[int, int]]],
                              num_restarts: int = 10) -> Tuple[List[int], int, List[int]]:
    """Çoklu başlangıç noktası ile greedy tur oluşturur."""
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
            
            for neighbor, dist in nearest_neighbors[current][:5]:
                if neighbor in unvisited:
                    # Burada maliyet sınırı kaldırıldı, deneme için
                    tour.append(neighbor)
                    unvisited.remove(neighbor)
                    total_cost += dist
                    found = True
                    break
            
            if not found:
                break
        
        if tour[-1] != start_city:
            try:
                return_cost = distance_matrix[tour[-1]][start_city]
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


def try_insert_skipped(tour: List[int], skipped: List[int], distance_matrix: Dict[int, Dict[int, int]], penalty: int) -> List[int]:
    """Atlanan şehirleri tur içine eklemeyi dener, ceza maliyetinden daha iyi ise ekler."""
    for city in skipped:
        best_increase = float('inf')
        best_pos = None
        for i in range(1, len(tour)):
            increase = distance_matrix[tour[i-1]][city] + distance_matrix[city][tour[i]] - distance_matrix[tour[i-1]][tour[i]]
            if increase < best_increase:
                best_increase = increase
                best_pos = i
        if best_increase < penalty:
            tour.insert(best_pos, city)
    return tour


def calculate_cost(tour: List[int], 
                   distance_matrix: Dict[int, Dict[int, int]], 
                   penalty: int, 
                   all_cities: Set[int]) -> int:
    """Turun toplam maliyetini hesaplar."""
    cost = 0
    visited = set(tour)
    
    for i in range(len(tour)-1):
        cost += distance_matrix[tour[i]][tour[i+1]]
    
    skipped = all_cities - visited
    cost += len(skipped) * penalty
    
    return cost


def two_opt_swap(tour: List[int], i: int, k: int) -> List[int]:
    """2-opt swap işlemi uygular."""
    return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]


def tabu_search_optimized(initial_tour: List[int], 
                         distance_matrix: Dict[int, Dict[int, int]], 
                         penalty: int, 
                         all_cities: Set[int],
                         max_iter: int = 1000,
                         tabu_size: int = 30) -> Tuple[List[int], int]:
    """Optimize edilmiş tabu search algoritması."""
    current_tour = initial_tour[:]
    best_tour = initial_tour[:]
    best_cost = calculate_cost(best_tour, distance_matrix, penalty, all_cities)
    tabu_list = deque(maxlen=tabu_size)
    
    for _ in range(max_iter):
        best_move = None
        best_delta = 0
        
        for i in range(1, len(current_tour)-2):
            for k in range(i+1, len(current_tour)-1):
                a, b = current_tour[i-1], current_tour[i]
                c, d = current_tour[k], current_tour[k+1]
                delta = (distance_matrix[a][c] + distance_matrix[b][d]) - (distance_matrix[a][b] + distance_matrix[c][d])
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
                       distance_matrix: Dict[int, Dict[int, int]],
                       penalty: int,
                       all_cities: Set[int],
                       initial_temp: float = 10000,
                       cooling_rate: float = 0.995,
                       min_temp: float = 1) -> Tuple[List[int], int]:
    """Simulated annealing algoritması."""
    current_tour = initial_tour[:]
    current_cost = calculate_cost(current_tour, distance_matrix, penalty, all_cities)
    best_tour = current_tour[:]
    best_cost = current_cost
    temp = initial_temp
    
    while temp > min_temp:
        i = random.randint(1, len(current_tour)-3)
        k = random.randint(i+1, len(current_tour)-2)
        
        a, b = current_tour[i-1], current_tour[i]
        c, d = current_tour[k], current_tour[k+1]
        delta = (distance_matrix[a][c] + distance_matrix[b][d]) - (distance_matrix[a][b] + distance_matrix[c][d])
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_tour = two_opt_swap(current_tour, i, k)
            current_cost += delta
            
            if current_cost < best_cost:
                best_tour = current_tour[:]
                best_cost = current_cost
        
        temp *= cooling_rate
    
    return best_tour, best_cost


def hybrid_optimization(initial_tour: List[int],
                       distance_matrix: Dict[int, Dict[int, int]],
                       penalty: int,
                       all_cities: Set[int]) -> Tuple[List[int], int]:
    """Tabu search ve simulated annealing hibrid algoritması."""
    tabu_tour, tabu_cost = tabu_search_optimized(
        initial_tour, distance_matrix, penalty, all_cities,
        max_iter=500, tabu_size=30
    )
    
    final_tour, final_cost = simulated_annealing(
        tabu_tour, distance_matrix, penalty, all_cities,
        initial_temp=5000, cooling_rate=0.993
    )
    
    return final_tour, final_cost


def write_output(filename: str, tour: List[int], cost: int, all_cities: Set[int]):
    """Çıktı dosyasını yazar."""
    visited = set(tour)
    skipped = list(all_cities - visited)
    
    with open(filename, 'w') as f:
        f.write(f"{cost} {len(tour)} \n")
        for city in tour:
            f.write(f"{city}\n")
        f.write(" ")


def main():
    start_time = time.time()
    
    penalty, cities = read_input("input_3.txt")
    city_ids = [city[0] for city in cities]
    all_cities = set(city_ids)
    
    distance_matrix, nearest_neighbors, best_start = create_distance_matrix(cities)
    
    greedy_tour, greedy_cost, skipped = greedy_tour_with_restarts(
        cities, penalty, distance_matrix, nearest_neighbors, num_restarts=20
    )
    
    # Atlanan şehirleri tur içine ekle
    improved_tour = try_insert_skipped(greedy_tour, skipped, distance_matrix, penalty)
    
    optimized_tour, optimized_cost = hybrid_optimization(
        improved_tour, distance_matrix, penalty, all_cities
    )
    optimized_tour.pop()
    write_output("output_3.txt", optimized_tour, optimized_cost, all_cities)
    
    print(f"Toplam Maliyet: {optimized_cost}")
    print(f"Çalışma Süresi: {time.time() - start_time:.2f} saniye")
    print(f"Tour:{optimized_tour}")


if __name__ == "__main__":
    main() 