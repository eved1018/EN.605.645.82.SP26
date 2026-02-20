import sys
from typing import List, Set, Tuple


def is_strongly_dominated( payoffs_c: List[int], payoffs_j: List[int]) -> bool:  # true if C is dominated by j
    return all(p1 < p2 for p1, p2 in zip(payoffs_c, payoffs_j))


def is_weakly_dominated( payoffs_c: List[int], payoffs_j: List[int]) -> bool:  # true if C dominated by j
    return all([p1 <= p2 for p1, p2 in zip(payoffs_c, payoffs_j)]) and any(
        [p1 < p2 for p1, p2 in zip(payoffs_c, payoffs_j)]
    )


def is_dominated(payoffs_c: List[int], payoffs_j: List[int], weak: bool) -> bool:
    if weak:
        return is_weakly_dominated(payoffs_c, payoffs_j)
    return is_strongly_dominated(payoffs_c, payoffs_j)


def get_payoffs( game: List[List[Tuple[int, int]]], player: int, strategy: int, indices: Tuple[int, ...],) -> List[int]:
    payoffs = []
    if player == 0:
        payoffs = [game[strategy][c][player] for c in indices]

    elif player == 1:
        payoffs = [game[r][strategy][player] for r in indices]

    return payoffs


def eliminate_dominated( game: List[List[Tuple[int, int]]], player: int, indices: Tuple[int, ...], weak: bool,) -> Tuple[int, ...]:
    dominated_stratagies = set()

    for c in indices:
        payoffs_c = get_payoffs(game, player, c, indices)

        for j in indices:
            if j == c:
                continue

            payoffs_j = get_payoffs(game, player, j,indices)

            if is_dominated(payoffs_c, payoffs_j, weak):
                # remove c from children and return
                dominated_stratagies.add(c)

    # take dominated_stratagies and remove from
    new_indices = set()
    for c in indices:
        if c not in dominated_stratagies:
            new_indices.add(c)

    return tuple(new_indices)


def succesors( game: List[List[Tuple[int, int]]], strategies: Tuple[Tuple[int, ...], Tuple[int, ...]], weak: bool) -> List[ Tuple[Tuple[int, ...], Tuple[int, ...]] ]:  # return stategies to be searched after removal of dominated stategies
    valid_rows = strategies[0]
    valid_cols = strategies[1]

    succesor_strategies = []

    new_rows = eliminate_dominated(game, 0, valid_rows, weak)
    succesor_strategies.append((new_rows, valid_cols))

    new_cols = eliminate_dominated(game, 1, valid_cols, weak)
    succesor_strategies.append((valid_rows, new_cols))
    return succesor_strategies


def is_goal_state(strategies: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> bool:
    # is there only one solution for a player left
    return len(strategies[0]) == 1 or len(strategies[1]) == 1


def search(game: List[List[Tuple[int, int]]], weak: bool = False) -> List[Tuple[int, int]]:  # BFS over all possible elminiations
    rows = len(game)
    cols = len(game[0])

    solutions: Set[Tuple[int, int]] = set()
    strategies: Tuple[Tuple[int, ...], Tuple[int, ...]] = (
        tuple(range(rows)),
        tuple(range(cols)),
    )
    frontier: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = [strategies]
    visited: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()

    depth = 0
    while frontier:
        strategies = frontier.pop(0)

        print(depth, strategies)
        visited.add(strategies)

        depth += 1

        next_strategies = succesors(game, strategies, weak)

        if is_goal_state(strategies):
            for i in strategies[0]:
                for j in strategies[1]:
                    solutions.add((i, j))
            continue

        for child in next_strategies:
            if child not in visited:
                frontier.append(child)

    return list(solutions)


prisoners_dilemma = [
    [(10, 10), (14, 12), (14, 15)],
    [(12, 14), (20, 20), (28, 15)],
    [(15, 14), (15, 28), (25, 25)],
]


eql = search(prisoners_dilemma)
print()
print(eql)


game2 = [[(3, 0), (2, 1), (0, 0)], [(1, 1), (1, 1), (5, 0)], [(0, 1), (4, 2), (0, 1)]]
print()
eql = search(game2)
print()
print(eql)

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


