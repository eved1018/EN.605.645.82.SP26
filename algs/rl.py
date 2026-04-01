from copy import deepcopy
from math import inf
from typing import List, Set, Dict, Tuple


"""
+ world: a `List` of `List`s of terrain (this is S from S, A, T, gamma, R)
+ costs: a `Dict` of costs by terrain (this is part of R)
+ goal: A `Tuple` of (x, y) stating the goal state.
+ reward: The reward for achieving the goal state.
+ actions: a `List` of possible actions, A, as offsets.
+ gamma: the discount rate

you will return a policy: 

`{(x1, y1): action1, (x2, y2): action2, ...}`

"""


"""
1 for each s in S, V[s] := 0
2 t := 0
3 do
    4 t := t + 1; Vlast[s] := V[s]
    5 for each s inS:
        6 for each a in A:
            7 Q[s, a] = R[s, a] + ɣ Σs'T(s,a,s') Vlast[s']
    8 π[s] := argmaxa: Q[s, a]
    9 V[s] := Q[s, π[s]]
10 until maxs |V[s] - Vlast[s]| < ε
"""


def get_states(world):
    states = []
    for r, row in enumerate(world):
        for c, val in enumerate(row):
            states.append((c, r))
    return states


def take_action(x, y, dx, dy, world):
    rows = len(world)
    cols = len(world[0])
    x_prime, y_prime = x + dx, y + dy
    if 0 <= x_prime < cols and 0 <= y_prime < rows and world[y_prime][x_prime] != "x":
        return x_prime, y_prime
    return x, y


def value_iteration(world, costs, goals, actions, gamma, transition=1.0, e=0.01):
    rows = len(world)
    cols = len(world[0])

    v = {state: 0.0 for state in get_states(world)}
    rewards = {state: goals.get(state, 0.0) for state in get_states(world)}
    t = 0
    while t < 100:
        v_last = deepcopy(v)
        policy = {state: "G" for state in goals}
        for x, y in get_states(world):
            if world[y][x] == "x":
                policy[(x, y)] = "x"
                continue

            if (x, y) in goals:
                v[(x, y)] = goals[(x, y)]
                policy[(x, y)] = "G"
                continue

            max_action = None
            max_val = -inf
            for dx, dy in actions:
                x_prime, y_prime = take_action(x, y, dx, dy, world)
                future_reward = transition * v_last[(x_prime, y_prime)]
                if transition < 1.0:
                    for ddx, ddy in actions:
                        other_transition = (1 - transition) / (len(actions) - 1)
                        if (dx, dy) != (ddx, ddy):
                            x_t, y_t = take_action(x, y, ddx, ddy, world)
                            future_reward += other_transition * v_last[(x_t, y_t)]

                val = rewards[(x, y)] - costs[world[y_prime][x_prime]] + gamma * future_reward
                if val > max_val:
                    max_val = val
                    max_action = (dx, dy)

            policy[(x, y)] = max_action
            v[(x, y)] = max_val

        delta = max([abs(v[state] - v_last[state]) for state in get_states(world)])
        if delta < e:
            return policy
        t += 1
    return None


def pretty_print_policy(cols, rows, policy, world):
    for r in range(rows):
        for c in range(cols):
            v = policy[(c, r)]
            if v == (-1, 0):
                v = "<"
            elif v == (1, 0):
                v = ">"
            elif v == (0, 1):
                v = "v"
            elif v == (0, -1):
                v = "^"

            print(v, end="")
        print()


world = [["o", "o", "o", "o", "o", "o", "o"]]
costs = {"o": 0}
goal = {(0, 0): 15, (6, 0): 10}
moves = [(-1, 0), (1, 0)]
gamma = 0.9


policy = value_iteration(world, costs, goal, moves, gamma)
pretty_print_policy(len(world[0]), len(world), policy, world)

world = [
    [".", ".", ".", ".", ".", "."],
    [".", "*", "*", "*", "*", "."],
    [".", "*", "*", "*", "*", "."],
    [".", "*", "*", "x", "*", "."],
    [".", "*", "*", "*", "*", "."],
    [".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", "."],
]


moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
costs = {".": -1, "*": -3, "^": -5, "~": -7}
goal = {(len(world[0]) - 1, len(world) - 1): 100}  # Lower Right Corner FILL ME IN
gamma = 0.9

print()
print()
policy = value_iteration(world, costs, goal, moves, gamma, 0.7)
pretty_print_policy(len(world[0]), len(world), policy, world)
