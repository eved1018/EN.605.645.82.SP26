


def evaluate(state):
    scores = [2.45 , 2.78 , 3.14 , 3.31 , 3.23 , 2.98 , 2.72 , 3.09 , 3.37 , 3.26]
    return scores[state -1]

def find_best_child(state, curr_val):
    child = state
    value = curr_val
    if state > 2 and evaluate(state -1) > value:
        child = state - 1 
        value = evaluate(state -1)
        
    if state < 10 and evaluate(state +1) > value:
        child = state + 1 
        value = evaluate(state + 1)

    return child, value


def main(initial_state):
    current = initial_state
    value = evaluate(current)
    while True: 
        print(current, value)
        candidate, cval = find_best_child(current, value)
        if cval <= value:
            return current
        current = candidate
        value = cval

result = main(2)
print(result)

