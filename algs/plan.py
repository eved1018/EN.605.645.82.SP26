import tokenize
from io import StringIO
from typing import List, Dict, Set, Tuple
from copy import deepcopy


def is_variable(exp):
    return isinstance(exp, str) and exp[0] == "?"


def is_constant(exp):
    return isinstance(exp, str) and not is_variable(exp)


# http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def occurs_check(exp1, exp2):
    return exp1 in flatten(exp2)


def inconsistent_assignment(exp1, exp2, frame):
    if not exp1 in frame:
        return False
    return not frame[exp1] == exp2


def unification(exp1, exp2, frame=None):
    #    print( "expr1", exp1)
    #    print( "expr2", exp2)
    if frame == None:
        frame = {}
    if is_constant(exp1) and is_constant(exp2) or len(exp1) == 0 and len(exp2) == 0:
        if exp1 == exp2:
            return frame
        else:
            return False
    if is_variable(exp1):
        if occurs_check(exp1, exp2) or inconsistent_assignment(exp1, exp2, frame):
            return False
        else:
            frame[exp1] = exp2
            return frame
    if is_variable(exp2):
        if occurs_check(exp2, exp1) or inconsistent_assignment(exp2, exp1, frame):
            return False
        else:
            frame[exp2] = exp1
            return frame
    head1 = exp1[0]
    head2 = exp2[0]
    frame = unification(head1, head2, frame)
    if frame == False:
        return False
    return unification(exp1[1:], exp2[1:], frame)


# unification( "?x", "Fred")
# unification( "Fred", "Fred")
# unification( "Fred", "Barney")
# unification(["son", "Barney", "?y"], ["son", "Barney", "Bam-Bam"])
# unification(["son", "?x", "Bam-Bam"], ["son", "Barney", "?x"])
# unification(["son", "?x", "Bam-Bam"], ["son", "Barney", "?y"])
# unification(["son", "?x", "Bam-Bam"], ["son", "Barney", "?y"])
# unification(["son", "?x", "Bam-Bam"], ["son", "Barney", "?x"])
# unification(["son", "Barney", "Bam-Bam"], ["son", "Barney", "?y"])
# unification(["son?", "Barney", "?x"], ["son?", "?y", ["son", "Barney"]])
# unification(["son?", "Barney", "?x"], ["son?", "?y", ["son", "?y"]])
# unification( ["loves", "?x", "?y"], ["loves", "Fred", "Wilma"])
# unification( ["loves", "?x", "Wilma"], ["loves", "Fred",  "?y"])
# unification( ["loves", "?x", "Wilma"], ["loves", "Fred",  "?x"])

# adapted from http://effbot.org/zone/simple-iterator-parser.htm

import sys
import tokenize
from io import StringIO


def atom(next, token):
    if token[1] == "(":
        out = []
        token = next()
        while token[1] != ")":
            out.append(atom(next, token))
            token = next()
            if token[1] == " ":
                token = next()
        return out
    elif token[1] == "?":
        token = next()
        return "?" + token[1]
    else:
        return token[1]


def parse(exp):
    src = StringIO(exp).readline
    tokens = tokenize.generate_tokens(src)
    return atom(tokens.__next__, tokens.__next__())


def unify(exp1, exp2):
    return unification(parse(exp1), parse(exp2))


# print( unify( "?x", "Wilma"))
# print( unify( "(loves ?x ?x)", "(loves Wilma Fred)"))
# print( unify( "(loves (leftLegOf ?x) (rightLegOf Wilma))", "(loves (leftLegOf Wilma) (rightLegOf ?y))"))
# print( unify( '(father Barney ?x)', '(father Barney (son_of Barney))'))

goal = [
    "(item Saw)",    
    "(item Drill)",
    "(place Home)",
    "(place Store)",
    "(place Bank)",    
    "(agent Me)",
    "(at Me Home)",
    "(at Drill Me)",
    "(at Saw Store)"    
]
actions = {
    "drive": {
        "action": "(drive ?agent ?from ?to)",
        "conditions": [
            "(agent ?agent)",
            "(place ?from)",
            "(place ?to)",
            "(at ?agent ?from)"
        ],
        "add": [
            "(at ?agent ?to)"
        ],
        "delete": [
            "(at ?agent ?from)"
        ]
    },
    "buy": {
        "action": "(buy ?purchaser ?seller ?item)",
        "conditions": [
            "(item ?item)",
            "(place ?seller)",
            "(agent ?purchaser)",
            "(at ?item ?seller)",
            "(at ?purchaser ?seller)"
        ],
        "add": [
            "(at ?item ?purchaser)"
        ],
        "delete": [
            "(at ?item ?seller)"
        ]
    }
}

start_state = [
    "(item Saw)",
    "(item Drill)",
    "(place Home)",
    "(place Store)",
    "(place Bank)",
    "(agent Me)",
    "(at Me Home)",
    "(at Saw Store)",
    "(at Drill Store)"
]
def hash_state(state):
    return frozenset(deatomize(s) for s in state) # adapted from pa6


def is_goal_state(state, goal):
    return hash_state(state) == hash_state(goal)

#TODO might need to be recursive
def apply(substitutions: Dict, expression: List) -> List:
    return [substitutions[i] if is_variable(i) and i in substitutions else i for i in expression]


def deatomize(expression: str | List) -> str | None:
    if isinstance(expression, str):  # base-case
        return expression

    if isinstance(expression, list):  # recurse
        str_expression = []
        for subexpressions in expression:
            result = deatomize(subexpressions)
            if result is None:
                return None
            str_expression.append(result)
        return "(" + " ".join(str_expression) + ")"

    return None


def apply_action_to_state(state: List, substitution_list, add_list, del_list):
    new_state = deepcopy(state)
    for add_item in add_list:
        add_item = apply(substitution_list, add_item)
        if add_item not in new_state:
            new_state.append(add_item)

    for del_item in del_list:
        del_item = apply(substitution_list, del_item)
        if del_item in new_state:
            new_state.remove(del_item)
    return new_state


def get_single_unification(state, conditions):
    substitution_list = {}
    full_match = True
    for precondition in conditions:
        match = False
        for predicates in state:
            result = unification(predicates, precondition, deepcopy(substitution_list))
            if result is not False:
                match = True
                substitution_list = result
                break  # next precondition
        if not match:
            full_match = False
            break  # next action

    if not full_match:
        return None
    return substitution_list


def get_neighbors(state: List, actions: Dict):
    successors = []
    for action in actions.values():
        substitution_list = get_single_unification(state, action["conditions"])
        if substitution_list is None:
            continue
        planned_action = apply(substitution_list, action["action"])
        new_state = apply_action_to_state(state, substitution_list, action["add"], action["delete"])
        successors.append((new_state, planned_action))
    return successors


def get_all_unification(state, conditions):
    result = []
    stack = [(conditions, {})]
    # no visited list bc we want tree search
    while stack:
        preconditions, substitution_list = stack.pop()
        if not preconditions:  # fully matched
            result.append(substitution_list)
            continue
        condition, remainder = preconditions[0], preconditions[1:] # same idea from unification alg.
        for predicate in state:
            child_sub_list = unification(predicate, condition, deepcopy(substitution_list))
            if child_sub_list is not False:
                stack.append((remainder, child_sub_list))
    return result


def get_all_neighbors(state: List, actions: Dict):
    successors = []
    for action in actions.values():
        for substitution_list in get_all_unification(state, action["conditions"]):
            planned_action = apply(substitution_list, action["action"])
            new_state = apply_action_to_state(state, substitution_list, action["add"], action["delete"])
            successors.append((new_state, planned_action))
    return successors


def parse_inputs(start_state, goal, actions):
    if (not start_state) or (not goal):
        return None
    state = [parse(p) for p in start_state]
    goal = [parse(p) for p in goal]

    parsed_actions = {}
    for name, action in actions.items():
        expected_keys = ["action", "conditions", "add", "delete"]
        if not all(k in action for k in expected_keys):
            return None
        parsed_actions[name] =  {
            "action": parse(action["action"]),
            "conditions": [parse(c) for c in action["conditions"]],
            "add": [parse(a) for a in action["add"]],
            "delete": [parse(d) for d in action["delete"]],
        }
        
    return state, goal, parsed_actions



def forward_planner(start_state, goal, actions, intermediate=False):
    inputs = parse_inputs(start_state, goal, actions)
    if inputs is None:
        return None
    start_state, goal, actions = inputs
    
    stack = [(start_state, [], [start_state])]
    visited = set() # graph search 
        
    while stack:
        current_state, plan, path = stack.pop()

        if hash_state(current_state) in visited:
            continue
        visited.add(hash_state(current_state))

        if is_goal_state(current_state, goal):
            if not intermediate:            
                return plan
            full_plan = []
            for action, state in zip(plan, path):
                full_plan.append(state)
                full_plan.append(action)
            full_plan.append(current_state)
            return full_plan

        for new_state, action in get_all_neighbors(current_state, actions):
            if hash_state(new_state) not in visited:
                stack.append((new_state, plan + [action], path + [current_state]))
    return None



plan = forward_planner(start_state, goal, actions)
assert plan is not None
for el in plan:
    print(el)

print()
print()
print()
plan = forward_planner(start_state, goal, actions, True)
assert plan is not None
for el in plan:
    print(el)
