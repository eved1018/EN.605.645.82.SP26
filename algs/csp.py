from typing import List, Set, Dict, Callable, Tuple
from math import inf
from copy import deepcopy


"""
Nodes: Ordered List of Strings - each represents a variable
Edges: List of Tuples - each tuple is index in nodes for 2 adjacent nodes
"""


def get_value_by_order(color_list):
    return sorted(color_list)[0]


def minimum_remaining_value(csp, assignments, color_list):  # pick var with least domain!
    n_colors = len(color_list)

    fewest_node = None
    fewest_node_index = -1
    fewest_domain = n_colors + 1

    for index, node in enumerate(csp["nodes"]):
        domain = color_list.copy()
        for neighbor_index in get_neighbors(csp, index):
            neighbor = csp["nodes"][neighbor_index]
            if neighbor in assignments:
                domain.remove(assignments[neighbor])
        if len(domain) < fewest_domain:
            fewest_node = node
            fewest_domain = len(domain)
            fewest_node_index = index

    return fewest_node, fewest_node_index

def contraint_value_overlaps(csp, variable_index, color):
    overlapping = 0
    for neighbor in get_neighbors(csp, variable_index):
        if color in csp["colors"][neighbor]:
            overlapping += 1
    return overlapping


def least_constraining_value(csp, variable_index, assignments, color_list):  # pick val
    return sorted(color_list, key = lambda x: contraint_value_overlaps(csp, variable_index, x))


def get_neighbors(csp, index):
    neighbors = []
    for edge in csp["edges"]:
        if index == edge[0]:
            neighbors.append(edge[1])
        elif index == edge[1]:
            neighbors.append(edge[0])
    return neighbors


def is_complete(csp, assignments):
    # correct number of assignments
    if len(assignments) != len(csp["nodes"]):
        return False
    return True


def degree_heuristic(csp, assignment):
    # degree hueristic
    best_node = None
    best_node_index = -1
    best_edges = 0
    for index, node in enumerate(csp["nodes"]):
        if node in assignment:
            continue
        n_edges = len(get_neighbors(csp, index))
        if n_edges > best_edges:
            best_node = node
            best_edges = n_edges
            best_node_index = index

    return best_node, best_node_index


def is_value_consistant(csp, variable, index, value, assignments):
    for neighbor_index in get_neighbors(csp, index):
        neighbor = csp["nodes"][neighbor_index]
        if neighbor in assignments and assignments[neighbor] == value:
            return False
    return True


def forward_checking(csp: Dict, variable: str, index: int, value: str, assignments: Dict[str, str], undo_list: List[Tuple[str,List[str]]]):
    for neighbor_index in get_neighbors(csp, index):
        neighbor = csp["nodes"][neighbor_index]
        if neighbor not in assignments:
            if value in csp["colors"][neighbor_index]:
                csp["colors"][neighbor_index].remove(value)
                undo_list.append((neighbor_index, [value]))

            if len(csp["colors"][neighbor_index]) == 0:  # pruned to much
                return False
    return True

def reverse_checking(csp, undo_list):  # reverse effect of forward_checking
    for index, values in undo_list:
        for value in values:
            if value not in csp["colors"][index]:
                csp["colors"][index].append(value)


def make_assignment(csp, variable, index, value, assignments, undo_list):
    undo_list.append((index, [c for c in csp["colors"][index] if c != value]))
    csp["colors"][index] = [value]
    assignments[variable] = value

def del_assignments(assignments, var, val):
    if var in assignments:
        c = assignments.pop(var)
        if c != val:
            assignments[var] = c
            return False
    return True


def color_map(planar_map, color_list: List[str], trace: bool = False):
    planar_map["colors"] = [list(color_list) for _ in range(len(planar_map["nodes"]))]
    return backtrack(planar_map, color_list, {})


def backtrack(planar_map, color_list, assignments) -> List[Tuple[str, str]] | None:
    if is_complete(planar_map, assignments):
        return assignments

    variable, variable_index = degree_heuristic(planar_map, assignments)
    for value in least_constraining_value(planar_map, variable_index, assignments, color_list):
        if is_value_consistant(planar_map, variable, variable_index, value, assignments):
            undo_list = []
            make_assignment(planar_map, variable, variable_index, value, assignments, undo_list)

            if forward_checking(planar_map, variable, variable_index, value, assignments, undo_list):
                result = backtrack(planar_map, color_list, assignments)

                if result is not None:
                    return result

            reverse_checking(planar_map, undo_list)
            if not del_assignments(assignments, variable, value):
                raise Exception("Could not remove assignment {} -> {}".format(variable, value))

    return None


connecticut = {
    "coordinates": [
        (46, 52),
        (217, 146),
        (65, 142),
        (147, 85),
        (162, 140),
        (104, 77),
        (197, 94),
        (123, 142),
    ],
    "edges": [
        (0, 2),
        (0, 5),
        (2, 5),
        (2, 7),
        (5, 7),
        (5, 3),
        (7, 3),
        (7, 4),
        (7, 6),
        (3, 6),
        (4, 6),
        (4, 1),
        (6, 1),
    ],
    "nodes": [
        "Fairfield",
        "Windham",
        "Litchfield",
        "Middlesex",
        "Tolland",
        "New Haven",
        "New London",
        "Hartford",
    ],
}


connecticut_colors = color_map(connecticut, ["red", "blue", "green", "yellow"])
print()
print("----"*20)
print(connecticut_colors)
