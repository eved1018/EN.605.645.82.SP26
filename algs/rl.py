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

World = List[List[str]]
State = Tuple[int, int]
Reward = List[List[float]]
Policy = Dict[State, State | str]

ACTION2TEXT = {(-1, 0): "<", (1, 0): ">", (0, 1): "v", (0, -1): "^"}

def get_states(world):
    states = []
    for r, row in enumerate(world):
        for c, _ in enumerate(row):
            states.append((c, r))
    return states

def get_actions(state, actions, world, impassible = ["x"], bounce_back = False):
    rows = len(world)
    cols = len(world[0])

    x, y = state
    moves = []
    for dx, dy in actions:
        x_prime, y_prime = x + dx, y + dy
        if 0 <= x_prime < cols and 0 <= y_prime < rows and world[y_prime][x_prime] not in impassible:
            moves.append((dx, dy))
        elif bounce_back:
            moves.append((0, 0))
    return moves

def get_action_value(world: World, state: State, action: State, actions: List[State], rewards: Reward, v_last: Reward, transition: float = 1.0, gamma: float = 1.0) -> Tuple[State, float]:
    x, y = state[0], state[1]
    x_prime, y_prime = x + action[0], y + action[1]
    future_reward = transition * v_last[y_prime][x_prime]

    if transition < 1.0:
        other_transition = (1 - transition) / (len(actions) - 1)
        for dx, dy in get_actions(state, actions, world, bounce_back=True):
            x_prob, y_prob = x + dx, y + dy
            if (x_prob, y_prob) == (x_prime, y_prime):
                continue
            future_reward += other_transition * v_last[y_prob][x_prob]
    val = rewards[y][x] + gamma * future_reward
    return action, val

def update_policy(world: World, actions: List[State], rewards: Reward, goals: Dict[State, float], v: Reward, policy: Policy, transition: float = 1.0, gamma: float = 1.0, impassible: List[str] = ["x"]) -> Tuple[Policy, Reward, Reward]:
    v_last = deepcopy(v)
    for x, y in get_states(world):
        if world[y][x] in impassible:
            policy[(x, y)] = "X"
            continue

        if (x, y) in goals:
            v[y][x] = goals[(x, y)]
            policy[(x, y)] = "G"
            continue
        
        action_values = (get_action_value(world, (x, y), move, actions, rewards, v_last, transition, gamma) for move in get_actions((x, y), actions, world))
        max_action, max_val = max(action_values, key = lambda x : x[1])
        policy[(x, y)] = max_action
        v[y][x] = max_val
    return policy, v, v_last

def value_iteration( world: World, costs: Dict[str, int], goals: Dict[State, float], actions: List[State], gamma: float = 1.0, transition: float = 1.0, e: float = 0.01, max_iters: int = 1000, debug: bool = False,) -> Policy:
    rows, cols = len(world), len(world[0])

    v = [[0.0] * cols for _ in range(rows)]
    policy: Policy = {state: "G" for state in goals}
    rewards = [[float(costs.get(x, 0.0)) for x in row] for row in world]
    for (x, y), r in goals.items():
        rewards[y][x] = r

    t = 0
    while max_iters < 1 or t < max_iters:
        policy, v, v_last = update_policy(world, actions, rewards, goals, v, policy, transition, gamma)
        delta = max([abs(v[y][x] - v_last[y][x]) for x, y in get_states(world)])
        print(t, v_last, v, policy, delta) if debug else None
        if delta < e:
            return policy
        t += 1
    return policy

def pretty_print_vi(v, v_last, rewards):
    print("Rewards:")
    for row in rewards:
        print(row)
    print("V Last:")
    for row in v_last:
        print(row)
    print("V:")
    for row in v:
        print(row)
    return

def pretty_print_policy(cols: int, rows: int, policy: Policy):
    for row in range(rows):
        for col in range(cols):
            action = policy[(col, row)]
            tile = ACTION2TEXT[action] if isinstance(action, tuple) else action
            print(tile, end="")
        print()


def test():
    world = [["o", "o", "o", "o", "o", "o", "o"]]
    costs = {"o": 0}
    goal = {(0, 0): 15, (6, 0): 10}
    moves = [(-1, 0), (1, 0)]
    gamma = 0.9

    policy = value_iteration(world, costs, goal, moves, gamma, transition=0.8, debug=True)

    print("Result:")
    pretty_print_policy(len(world[0]), len(world), policy)

    # world = [
    #     [".", ".", ".", ".", ".", "."],
    #     [".", "*", "*", "*", "*", "."],
    #     [".", "*", "*", "*", "*", "."],
    #     [".", "*", "*", "x", "*", "."],
    #     [".", "*", "*", "*", "*", "."],
    #     [".", ".", ".", ".", ".", "."],
    #     [".", ".", ".", ".", ".", "."],
    # ]

    # moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    # costs = {".": -1, "*": -3, "^": -5, "~": -7}
    # goal = {(len(world[0]) - 1, len(world) - 1): 100}  # Lower Right Corner FILL ME IN
    # gamma = 0.9

    # print()
    # print()
    # policy = value_iteration(world, costs, goal, moves, gamma)
    # pretty_print_policy(len(world[0]), len(world), policy)

    # print()
    # print()
    # policy = value_iteration(world, costs, goal, moves, gamma, 0.7)
    # pretty_print_policy(len(world[0]), len(world), policy)


test()
