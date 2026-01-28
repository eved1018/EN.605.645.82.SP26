from copy import deepcopy
from math import sqrt
from typing import Callable, Dict, List, Set, Tuple


def priority_enqueue(pq: List, element: Tuple[int, int], priority: int | float):
    pq.append((element, priority))
    pq.sort(key=lambda x: x[1])  # mutates pq (pass-by-ref is so weird in python)
    return


def priority_dequeue(pq: List):
    return pq.pop(0)[0]


def get_neighbors(world: List[List[str]], node: Tuple[int, int], moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    max_x = len(world[0]) - 1  # Assumes all rows are the same size
    max_y = len(world) - 1
    valid_moves: List[Tuple[int, int]] = []

    for dx, dy in moves:
        x: int = node[0] + dx
        y: int = node[1] + dy
        if 0 <= x <= max_x and 0 <= y <= max_y and world[y][x] != "🌋":
            valid_moves.append((dx, dy))
    # print(f"T({node[0]}, {node[1]}) -> {valid_moves}")
    return valid_moves


def heuristic(start: Tuple[int, int], goal: Tuple[int, int]) -> int | float:
    distance = sqrt((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2)  # euclidean distance?
    # distance = 0
    return distance


def a_star_search(
    world: List[List[str]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    costs: Dict[str, int],
    moves: List[Tuple[int, int]],
    heuristic: Callable,
) -> List[Tuple[int, int]]:
    ### YOUR SOLUTION HERE ###
    ### YOUR SOLUTION HERE ###
    pq: List[Tuple[int, int]] = []
    visited: Set[Tuple[int, int]] = set()
    actions: List[Tuple[int, int]] = []

    prev = start
    priority_enqueue(pq, start, 0)

    while pq:  # Check if more neighbors to traverse
        offset: Tuple[int, int] = priority_dequeue(pq)
        actions.append(offset)
        node = (prev[0] + offset[0], prev[1] + offset[1])
        x, y = node
        prev = node

        if node == goal:
            return actions

        visited.add(node)
        for dx, dy in get_neighbors(world, node, moves):
            neighbor = (x + dx, y + dy)
            if neighbor not in visited:
                priority: int | float = costs[world[neighbor[1]][neighbor[0]]] + heuristic(neighbor, goal)
                priority_enqueue(pq, (dx, dy), priority)
    return []


def pretty_print_path(world: List[List[str]], path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int]) -> int:
    ### YOUR SOLUTION HERE ###
    ### YOUR SOLUTION HERE ###
    total_cost = 0
    x = start[0]
    y = start[1]

    map = deepcopy(world)
    map[goal[1]][goal[0]] = "🎁"

    for dx, dy in path:
        node = (x + dx, y + dy)
        print(node)
        total_cost += costs[world[y][x]]
        tile = map[y][x]
        if (dx, dy) == (0, 1):
            tile = "⏬"
        elif (dx, dy) == (1, 0):
            tile = "⏩"
        elif (dx, dy) == (-1, 0):
            tile = "⏪"
        elif (dx, dy) == (0, -1):
            tile = "⏫"
        map[y][x] = tile
        x, y = node

    for y, row in enumerate(map):
        for x, tile in enumerate(row):
            print(tile, end="")
        print("\n")
    return total_cost


small_world = [
    ["🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🐊", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "⛰", "⛰", "⛰"],
    ["🌾", "🌾", "🌾", "🌾", "🌾", "⛰", "⛰", "🌾", "🌾", "🌾", "🌾", "🐊", "🐊", "🐊", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "⛰", "🌋", "🌋", "⛰"],
    ["🌾", "🌾", "🌾", "🌾", "🌾", "⛰", "⛰", "⛰", "🌾", "🌾", "🐊", "🐊", "🐊", "🐊", "🌾", "🌾", "🌋", "🌾", "🌾", "🌾", "⛰", "🌾", "🌾", "⛰", "⛰", "🌋", "🌾"],
    ["🌾", "🌾", "🌾", "🌾", "⛰", "⛰", "🌋", "⛰", "⛰", "🐊", "🐊", "🐊", "🐊", "🐊", "🌾", "🌋", "🌋", "🌋", "🌋", "🌾", "⛰", "🌾", "🌾", "🌾", "⛰", "🌋", "🌾"],
    ["🌾", "🌾", "🌋", "⛰", "⛰", "🌋", "🌋", "⛰", "⛰", "🐊", "🐊", "🐊", "🐊", "🐊", "🌋", "🌋", "🌲", "🌋", "🌋", "🌋", "⛰", "⛰", "🌾", "🌾", "⛰", "⛰", "🌾"],
    ["🌲", "🌾", "🌋", "🌋", "🌋", "🌋", "⛰", "⛰", "⛰", "🐊", "🐊", "🐊", "🌾", "🌾", "🌾", "🌋", "🌲", "🌲", "🌋", "🌋", "⛰", "⛰", "🌾", "⛰", "⛰", "🌾", "🌾"],
    ["🌲", "🌾", "🌲", "🌋", "🌋", "⛰", "⛰", "⛰", "🌾", "🌾", "🐊", "🌾", "🌾", "🌾", "🌾", "🌲", "🌲", "🌲", "🌲", "🌋", "🌋", "⛰", "⛰", "⛰", "🌾", "🌾", "🌾"],
    ["🌲", "🌲", "🌲", "🌋", "🌲", "⛰", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "⛰", "⛰", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌋", "🌋", "⛰", "⛰", "🌾", "🌾", "🌾"],
    ["🌲", "🌲", "🌲", "🌲", "🌲", "🌾", "🌾", "🌾", "🌾", "⛰", "⛰", "⛰", "⛰", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌋", "⛰", "⛰", "🌾", "🌾", "🌾"],
    ["🌲", "🌲", "🌲", "🌲", "🌾", "🌾", "🌾", "🌾", "🌾", "⛰", "⛰", "🌋", "🌋", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌋", "🌋", "⛰", "🌾", "🌾", "🌾"],
    ["🌲", "🌲", "🌲", "🌲", "🌾", "🌾", "🌾", "🌾", "🌾", "⛰", "🌋", "🌋", "🌋", "⛰", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌋", "🌾", "🌾", "⛰", "🌾"],
    ["🌲", "🌲", "🌲", "🌲", "🐊", "🌾", "⛰", "🌾", "🌾", "🌋", "🌋", "⛰", "⛰", "🌾", "⛰", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌋", "🌾", "🌋", "⛰", "🌾"],
    ["🌲", "🌲", "🌲", "🐊", "🐊", "🐊", "🌋", "🌾", "⛰", "🌋", "🌋", "🌾", "🌾", "🌾", "⛰", "🌋", "🌲", "🌲", "🌲", "🌲", "🌲", "🌋", "🌋", "🌾", "🌋", "🌋", "⛰"],
    ["🌲", "🌲", "🌲", "🐊", "🐊", "🐊", "🌋", "⛰", "⛰", "🌋", "⛰", "🌾", "🐊", "🌾", "⛰", "🌋", "🌲", "🌲", "🌲", "🌾", "🌾", "🌋", "🌾", "🌾", "🌋", "🌋", "⛰"],
    ["🌲", "🌲", "🌲", "🌲", "🐊", "🐊", "🌋", "🌋", "🌋", "🌋", "🌾", "🌾", "🐊", "🌾", "⛰", "🌋", "🌋", "🌲", "🌾", "🌾", "🌋", "🌾", "🌾", "🌾", "⛰", "🌋", "⛰"],
    ["🌾", "🌲", "🌲", "🌲", "🌲", "🐊", "🐊", "🌋", "🌋", "🌾", "🌾", "🌾", "🐊", "🐊", "🌾", "⛰", "🌋", "🌲", "🌾", "🌾", "🌾", "🌾", "🌾", "⛰", "⛰", "🌋", "⛰"],
    ["🌾", "🌾", "🌲", "🌲", "🌲", "🐊", "🐊", "🌋", "🌾", "🌾", "🌾", "🐊", "🐊", "🐊", "🐊", "⛰", "🌋", "🌋", "🌾", "🌾", "🌾", "🌾", "🌾", "⛰", "🌋", "⛰", "⛰"],
    ["🌾", "🌾", "🌋", "🌲", "🌲", "🌾", "🐊", "🐊", "🐊", "🌾", "🌾", "🐊", "🌾", "🐊", "🐊", "🌾", "🌾", "🌋", "⛰", "🌾", "🌾", "🌾", "⛰", "🌋", "🌋", "⛰", "🌾"],
    ["🌾", "🌋", "🌋", "🌲", "🌾", "🌾", "🌾", "🐊", "🐊", "🐊", "🌾", "🌾", "🌾", "🐊", "🐊", "🐊", "🌾", "🌋", "⛰", "🌾", "🌾", "🌾", "⛰", "🌋", "🌾", "⛰", "🌾"],
    ["🌾", "🌋", "🌋", "🌾", "🌾", "🌾", "🌾", "🐊", "🌾", "🌾", "⛰", "🌾", "🌾", "🌾", "🌾", "🌾", "🌋", "🌋", "🌾", "🌾", "🌾", "🌾", "🌾", "⛰", "🌋", "⛰", "🌾"],
    ["🌾", "🌋", "⛰", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "⛰", "🌋", "⛰", "⛰", "🌾", "🌾", "⛰", "🌋", "🌾", "🌾", "🌾", "🐊", "🐊", "🌾", "⛰", "🌋", "🌋", "🌾"],
    ["🌾", "🌋", "⛰", "⛰", "⛰", "🌾", "🌾", "🌾", "⛰", "⛰", "🌋", "🌋", "🌋", "⛰", "⛰", "🌋", "🌋", "🌾", "🌾", "🌾", "🐊", "🐊", "🐊", "🌾", "⛰", "🌋", "⛰"],
    ["🌾", "🌋", "⛰", "⛰", "🌋", "⛰", "🌾", "⛰", "⛰", "⛰", "🌋", "🌋", "⛰", "🌾", "🌋", "🌋", "🌾", "🌾", "🌾", "🌾", "🐊", "🐊", "🐊", "🐊", "⛰", "🌋", "⛰"],
    ["🌾", "🌋", "🌋", "🌋", "🌋", "🌋", "⛰", "⛰", "⛰", "🌾", "⛰", "⛰", "🌾", "🌾", "⛰", "⛰", "🌾", "🌾", "🌾", "🐊", "🐊", "🐊", "🐊", "🐊", "🐊", "🐊", "⛰"],
    ["🌾", "🌋", "🌋", "🌋", "🌋", "⛰", "🌾", "⛰", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🐊", "🐊", "🐊", "🐊", "🐊", "🐊", "🐊", "🌾"],
    ["🌾", "🌾", "⛰", "⛰", "⛰", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🐊", "🐊", "🐊", "🐊", "🐊", "🐊", "🐊", "🌾"],
    ["🌾", "🌾", "⛰", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🐊", "🐊", "🐊", "🐊", "🐊", "🐊", "🐊", "🌾"],
]

# small_world = [
#     ["🌾", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲"],
#     ["🌾", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲"],
#     ["🌾", "🌲", "🌲", "🌲", "🌲", "🌲", "🌲"],
#     ["🌾", "🌾", "🌾", "🌾", "🌾", "🌾", "🌾"],
#     ["🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌾"],
#     ["🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌾"],
#     ["🌲", "🌲", "🌲", "🌲", "🌲", "🌲", "🌾"],
# ]

MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]
COSTS = {"🌾": 1, "🌲": 3, "⛰": 5, "🐊": 7}

world = small_world
start = (0, 0)
goal = (len(small_world[0]) - 1, len(small_world) - 1)
costs = COSTS
moves = MOVES
actions = a_star_search(world, start, goal, costs, moves, heuristic)
print(actions)
total_cost = pretty_print_path(world, actions, start, goal, costs)
print(total_cost)
