import math
import networkx as nx
from itertools import combinations
import time
from functools import lru_cache

@lru_cache(maxsize=None)
def euclidean_distance(c1, c2):
    return round(math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))

def parse_input_with_penalty(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    penalty = int(lines[0])
    cities = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 3:
            city_id, x, y = int(parts[0]), int(parts[1]), int(parts[2])
            cities[city_id] = (x, y)
    return penalty, cities

def total_tour_length(tour, cities):
    dist = 0
    num_cities = len(tour)
    for i in range(num_cities):
        c1 = cities[tour[i]]
        c2 = cities[tour[(i + 1) % num_cities]]
        dist += euclidean_distance(c1, c2)
    return dist

def two_opt(tour, cities, max_iterations=1000):
    num_cities = len(tour)
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for i in range(1, num_cities - 2):
            for j in range(i + 1, num_cities):
                if j - i == 1:
                    continue
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                new_length = total_tour_length(new_tour, cities)
                if new_length < total_tour_length(tour, cities):
                    tour = new_tour
                    improved = True
    return tour

def three_opt(tour, cities, max_iterations=1000):
    num_cities = len(tour)
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for i in range(num_cities - 2):
            for j in range(i + 2, num_cities - 1):
                for k in range(j + 2, num_cities):
                    moves = [
                        tour[:i+1] + tour[j:k] + tour[i+1:j] + tour[k:],
                        tour[:i+1] + tour[j:k][::-1] + tour[i+1:j] + tour[k:],
                        tour[:i+1] + tour[j:k] + tour[i+1:j][::-1] + tour[k:],
                        tour[:i+1] + tour[j:k][::-1] + tour[i+1:j][::-1] + tour[k:]
                    ]
                    for new_tour in moves:
                        if total_tour_length(new_tour, cities) < total_tour_length(tour, cities):
                            tour = new_tour
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return tour

def christofides_tour(cities):
    G = nx.Graph()
    for i, j in combinations(cities.keys(), 2):
        dist = euclidean_distance(cities[i], cities[j])
        G.add_edge(i, j, weight=dist)

    mst = nx.minimum_spanning_tree(G)
    odd_nodes = [v for v in mst.nodes if mst.degree(v) % 2 == 1]
    
    M = nx.Graph()
    for i, j in combinations(odd_nodes, 2):
        dist = euclidean_distance(cities[i], cities[j])
        M.add_edge(i, j, weight=dist)
    
    matching = nx.algorithms.matching.min_weight_matching(M)
    
    multigraph = nx.MultiGraph(mst)
    for u, v in matching:
        multigraph.add_edge(u, v, weight=euclidean_distance(cities[u], cities[v]))

    eulerian = list(nx.eulerian_circuit(multigraph))
    
    path = []
    visited = set()
    for u, v in eulerian:
        if u not in visited:
            visited.add(u)
            path.append(u)
    
    return path

def two_opt_fast(tour, cities, max_iterations=1000):
    num_cities = len(tour)
    improved = True
    best_length = total_tour_length(tour, cities)
    
    while improved and max_iterations > 0:
        max_iterations -= 1
        improved = False
        
        best_move = None
        best_move_length = best_length
        
        for i in range(1, num_cities - 2):
            for j in range(i + 1, num_cities):
                if j - i == 1:
                    continue
                
                old_edges = (euclidean_distance(cities[tour[i-1]], cities[tour[i]]) +
                           euclidean_distance(cities[tour[j]], cities[tour[(j+1)%num_cities]]))
                new_edges = (euclidean_distance(cities[tour[i-1]], cities[tour[j]]) +
                           euclidean_distance(cities[tour[i]], cities[tour[(j+1)%num_cities]]))
                
                if new_edges < old_edges:
                    new_length = best_length - old_edges + new_edges
                    if new_length < best_move_length:
                        best_move = (i, j)
                        best_move_length = new_length
        
        if best_move:
            i, j = best_move
            tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
            best_length = best_move_length
            improved = True
    
    return tour

def filter_cities_by_penalty(cities, penalty):
    threshold = penalty *1.2
    keep = {}
    for i in cities:
        close_count = 0
        for j in cities:
            if i != j and euclidean_distance(cities[i], cities[j]) <= threshold:
                close_count += 1
        if close_count >= 2:
            keep[i] = cities[i]
    return keep

def solve_with_penalty(input_file, output_file):
    start_time = time.time()
    
    penalty, cities = parse_input_with_penalty(input_file)
    filtered_cities = filter_cities_by_penalty(cities, penalty)
    
    try:
        tour = christofides_tour(filtered_cities)
        tour = two_opt_fast(tour, filtered_cities)
        
        visited = set(tour)
        skipped = set(cities) - visited
        total_cost = total_tour_length(tour, cities) + penalty * len(skipped)
        
        with open(output_file, 'w') as f:
            f.write(f"{total_cost} {len(tour)}\n")
            for cid in tour:
                f.write(f"{cid}\n")
            f.write("\n")
    except Exception as e:
        print("Hata:", e)
    
    print(f"Program çalışma süresi: {time.time() - start_time:.2f} saniye")


if __name__ == "__main__":
    solve_with_penalty("example-input-3.txt", "output3.txt")
