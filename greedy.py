import math
from typing import List, Tuple, Set, Dict
from collections import defaultdict

class City:
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y

def calculate_distance(city1: City, city2: City) -> float:
    """Calculate Euclidean distance between two cities"""
    return math.sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2)

def find_bridge_cities(cities: List[City], penalty_value: float) -> Dict[int, Set[int]]:
    """Find bridge cities that connect different clusters"""
    n = len(cities)
    if n <= 1:
        return {}
    
    # Sort cities by x-coordinate
    sorted_cities = sorted(enumerate(cities), key=lambda x: x[1].x)
    indices = [i for i, _ in sorted_cities]
    sorted_cities = [c for _, c in sorted_cities]
    
    # Track bridge cities (cities that connect different clusters)
    bridge_cities = defaultdict(set)
    
    def find_bridges_between(left: int, right: int):
        if right - left <= 1:
            return
        
        mid = (left + right) // 2
        mid_x = sorted_cities[mid].x
        
        # Find bridges in left and right halves
        find_bridges_between(left, mid)
        find_bridges_between(mid, right)
        
        # Find cities that cross the middle line (potential bridge cities)
        strip = []
        for i in range(left, right):
            if abs(sorted_cities[i].x - mid_x) < penalty_value:
                strip.append(i)
        
        # Sort strip by y-coordinate
        strip.sort(key=lambda i: sorted_cities[i].y)
        
        # Find bridge cities in strip
        for i in range(len(strip)):
            for j in range(i+1, min(i+7, len(strip))):
                dist = calculate_distance(sorted_cities[strip[i]], sorted_cities[strip[j]])
                if dist < penalty_value:
                    # Mark these cities as bridge cities
                    bridge_cities[indices[strip[i]]].add(indices[strip[j]])
                    bridge_cities[indices[strip[j]]].add(indices[strip[i]])
    
    # Find all bridge cities
    find_bridges_between(0, n)
    return bridge_cities

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

def main():
    # Example usage
    filename = "test_input.txt"  # Test input file
    penalty_value, cities = read_cities_from_file(filename)
    
    # Find bridge cities
    bridge_cities = find_bridge_cities(cities, penalty_value)
    
    # Print results
    print(f"Penalty value: {penalty_value}")
    print("\nBridge Cities:")
    for city_id, connected_cities in bridge_cities.items():
        print(f"City {city_id} connects to: {sorted(connected_cities)}")

if __name__ == "__main__":
    main() 
