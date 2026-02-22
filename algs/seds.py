from typing import List, Set, Tuple

"""
The goal of this module is to implemend succesive iteration of dominant stratagies.
It takes in normal form game and wether to use strong or weakly dominant elimination
For striclty dominant elimination it returns the Nash Equilibrium stratagies,
for weakly dominant eliminiation it returns all possible Nash Equilibrium

The basics of the SEDS algorithm are straightforward.
Consider any two of my available strategies. If the payoff from strategy A is greater than the payoff from
strategy B no matter what my opponent does, Im never going to be pick Strategy B. In such a case,
Strategy A strongly dominates Strategy B and the dominated strategy can be eliminated from
consideration. Essentially, the range of strategies I need to consider just got smaller. Yay.
It may be the case that some of the payoffs for Strategies A and B are equal for a given strategy of the
opponent. If at least one is better than we can say Strategy A weakly dominates Strategy B and we
can still eliminate Strategy B from consideration.

This module uses BFS and state space search to search through all possible elmination paths
"""

def is_strongly_dominated(
    payoffs_c: List[int], payoffs_j: List[int]
) -> bool:  # true if C is dominated by j
    return all(p1 < p2 for p1, p2 in zip(payoffs_c, payoffs_j))


def is_weakly_dominated(
    payoffs_c: List[int], payoffs_j: List[int]
) -> bool:  # true if C dominated by j
    return all([p1 <= p2 for p1, p2 in zip(payoffs_c, payoffs_j)]) and any(
        [p1 < p2 for p1, p2 in zip(payoffs_c, payoffs_j)]
    )


def is_dominated(payoffs_c: List[int], payoffs_j: List[int], weak: bool) -> bool:
    if weak:
        return is_weakly_dominated(payoffs_c, payoffs_j)
    return is_strongly_dominated(payoffs_c, payoffs_j)


def get_payoffs(
    game: List[List[Tuple[int, int]]],
    player: int,
    strategy: int,
    opponent_indices: Tuple[int, ...],
) -> List[int]:
    if player == 0:
        return [game[strategy][c][player] for c in opponent_indices]

    return [game[r][strategy][player] for r in opponent_indices]


def dominated_strategies(
    game: List[List[Tuple[int, int]]],
    player: int,
    row_indices: Tuple[int, ...],
    col_indices: Tuple[int, ...],
    weak: bool,
) -> Set[int]:
    indices = row_indices if player == 0 else col_indices
    opponent_indices = col_indices if player == 0 else row_indices
    dominated = set()

    for c in indices:
        payoffs_c = get_payoffs(game, player, c, opponent_indices)

        for j in indices:
            if j == c:
                continue

            payoffs_j = get_payoffs(game, player, j, opponent_indices)

            if is_dominated(payoffs_c, payoffs_j, weak):
                dominated.add(c)
                break

    return dominated


def remove_one_strategy(indices: Tuple[int, ...], remove_index: int) -> Tuple[int, ...]:
    return tuple(i for i in indices if i != remove_index)


def successors(
    game: List[List[Tuple[int, int]]],
    strategies: Tuple[Tuple[int, ...], Tuple[int, ...]],
    weak: bool,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    valid_rows, valid_cols = strategies
    successor_strategies = []

    dominated_rows = dominated_strategies(game, 0, valid_rows, valid_cols, weak)
    for row in dominated_rows:
        successor_strategies.append((remove_one_strategy(valid_rows, row), valid_cols))

    dominated_cols = dominated_strategies(game, 1, valid_rows, valid_cols, weak)
    for col in dominated_cols:
        successor_strategies.append((valid_rows, remove_one_strategy(valid_cols, col)))

    return successor_strategies


def search(game: List[List[Tuple[int, int]]], weak: bool = False) -> List[Tuple[int, int]]:  # BFS over all possible elminiations
    rows = len(game)
    cols = len(game[0])

    solutions: Set[Tuple[int, int]] = set()
    strategies: Tuple[Tuple[int, ...], Tuple[int, ...]] = (
        tuple(range(rows)),
        tuple(range(cols)),
    )
    frontier: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = {strategies}
    visited: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()

    while frontier:
        strategies = frontier.pop(0)
        visited.add(strategies)

        next_strategies = successors(game, strategies, weak)

        if not next_strategies:
            for i in strategies[0]:
                for j in strategies[1]:
                    solutions.add((i, j))
            continue

        for child in next_strategies:
            if child not in visited:
                frontier.add(child)

    return list(solutions)


game1 = [
    [(10, 10), (14, 12), (14, 15)],
    [(12, 14), (20, 20), (28, 15)],
    [(15, 14), (15, 28), (25, 25)],
]


eql = search(game1, False)
print()
print(eql)
assert eql == [(1, 1)]

game2 = [[(3, 0), (2, 1), (0, 0)], [(1, 1), (1, 1), (5, 0)], [(0, 1), (4, 2), (0, 1)]]
print()
eql = search(game2)
print()
print(eql)
assert eql == [(2, 1)]

game3 = [
    [(1, 0), (3, 1), (1, 1)],
    [(1, 1), (3, 0), (0, 1)],
    [(2, 2), (3, 3), (0, 2)],
]


print()
eql = search(game3, True)
print()
print(eql)


prisoners_dilemma = [
 [( -5, -5), (-1,-10)],
 [(-10, -1), (-2, -2)]]

print()
eql = search(prisoners_dilemma, True)
print()
print(eql)
assert eql == [(0,0)]

#
# game2 = [[(3, 0), (2, 1), (0, 0)], [(1, 1), (1, 1), (5, 0)], [(0, 1), (4, 2), (0, 1)]]
# print()
# eql = search(game2)
# print()
# print("game 2: ", eql)
#
#
# game3 = [
#     [(1, 0), (3, 1), (1, 1)],
#     [(1, 1), (3, 0), (0, 1)],
#     [(2, 2), (3, 3), (0, 2)],
# ]
#
# print()
# eql = search(game3, True)
# print()
# print("game3: ", eql)
# # assert eql == [(0, 1), (0, 2)]
#
# game4 = [[(-5, -5), (-1, -10)], [(-10, -1), (-2, -2)]]
#
# print()
# eql = search(game4, True)
# print()
# print("game 4: ", eql)
# assert eql == [(0, 0)]
