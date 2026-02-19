def is_strongly_dominated(payoffs_c, payoffs_j):  # true if C dominates j
    for p1, p2 in zip(payoffs_c, payoffs_j):
        if not p1 > p2:
            return False
    return True

def is_weakly_dominated(payoffs_c, payoffs_j):  # true if C dominates j
    for p1, p2 in zip(payoffs_c, payoffs_j):
        if not p1 >= p2:
            return False
    return True

def get_payoffs(game, player, strategy, valid_rows, valid_cols):
    payoffs = []
    if player == 0:
        payoffs = [game[strategy][c][player] for c in valid_cols]

    elif player == 1:
        payoffs = [game[r][strategy][player] for r in valid_rows]

    return payoffs


def eliminate_dominated(game, player, valid_rows, valid_cols):
    indices = valid_rows if player == 0 else valid_cols

    new_indices = []

    for i, c in enumerate(indices):
        payoffs_c = get_payoffs(game, player, c, valid_rows, valid_cols)
        for j in indices:
            if j == c:
                continue
            payoffs_j = get_payoffs(game, player, j, valid_rows, valid_cols)
            if is_strongly_dominated(payoffs_j, payoffs_c):
                new_indices.append(indices[:i] + indices[i + 1 :])
                break

    return new_indices


def succesors(
    game, strategies
):  # return stategies to be searched after removal of dominated stategies
    valid_rows = strategies[0]
    valid_cols = strategies[1]

    succesor_strategies = []

    new_rows = eliminate_dominated(game, 0, valid_rows, valid_cols)
    for rows in new_rows:
        succesor_strategies.append((rows, valid_cols))

    new_cols = eliminate_dominated(game, 1, valid_rows, valid_cols)
    for cols in new_cols:
        succesor_strategies.append((valid_rows, cols))

    return succesor_strategies


def search(game):  # BFS over all possible elminiations
    rows = len(game)
    cols = len(game[0])

    nash_eql = []
    explored = set()
    strategies = (tuple(range(rows)), tuple(range(cols)))
    frontier = [strategies]

    while frontier:
        strategies = frontier.pop(0)
        next_strategies = succesors(game, strategies)
        if not next_strategies:
            if strategies not in explored:
                explored.add(strategies)
                for r in strategies[0]:
                    for c in strategies[1]:
                        nash_eql.append((r, c))
        else:
            frontier.extend(next_strategies)

    return nash_eql


prisoners_dilemma = [
    [(10, 10), (14, 12), (14, 15)],
    [(12, 14), (20, 20), (28, 15)],
    [(15, 14), (15, 28), (25, 25)],
]


eql = search(prisoners_dilemma)
print(eql)


game2 = [
    [(3, 0), (2, 1), (0, 0)],
    [(1, 1), (1, 1), (5, 0)],
    [(0, 1), (4, 2), (0, 1)]
]

eql = search(game2)
print(eql)

