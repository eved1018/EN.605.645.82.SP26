#!/usr/bin/env python
# coding: utf-8

# Evan Edelstein
# EN.605.645.82.SP26

import random
from math import exp, log10
from typing import Dict, List, Tuple


clean_data = {
    "plains": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, "plains"]],
    "forest": [
        [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, "forest"],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, "forest"],
    ],
    "hills": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, "hills"],
    ],
    "swamp": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "swamp"], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, "swamp"]],
}


def blur(data):
    def apply_noise(value):
        if value < 0.5:
            v = random.gauss(0.30, 0.07)  # (0.10, 0.05)
            if v < 0.0:
                return 0.0
            if v > 0.75:
                return 0.75
            return v
        else:
            v = random.gauss(0.70, 0.07)  # (0.90, 0.10)
            if v < 0.25:
                return 0.25
            if v > 1.00:
                return 1.00
            return v

    noisy_readings = [apply_noise(v) for v in data[0:-1]]
    return noisy_readings + [data[-1]]


def slog(x: float) -> float:
    return log10(x) if x > 0.0 else 0.0


def parse_data(data: List[Tuple[List[float], int]]) -> Tuple[List[List[float]], List[int]]:
    features: List[List[float]] = [i[0] for i in data]
    ys: List[int] = [i[1] for i in data]

    if not all(float(i[0]) == 1.0 for i in features):
        for i in features:  # add in dummy variable
            i.insert(0, 1.0)
    return features, ys


def init_thetas(size: int, low: int, high: int) -> List[float]:
    return [random.uniform(low, high) for _ in range(size)]


def calc_yhats(thetas: List[float], features: List[List[float]]) -> List[float]:
    zs = [0.0] * len(features)

    for c, xs in enumerate(features):
        for t, x in zip(thetas, xs):
            zs[c] += t * x

    yhats: List[float] = [1.0 / (1.0 + exp(-1.0 * z)) for z in zs]
    return yhats


def calc_error(ys: List[int], yhats: List[float]) -> float:
    error = 0.0
    for yh, y in zip(yhats, ys):
        error += y * slog(yh) + (1 - y) * slog(1 - yh)

    error = (-1.0 / len(ys)) * error
    return error



def calc_thetas(thetas: List[float], features: List[List[float]], ys: List[int], yhats: List[float], alpha: float) -> List[float]:
    n = len(ys)
    ds = [(yh - y) for yh, y in zip(yhats, ys)]
    
    js = [0.0] * len(thetas)
    
    for c, xs in enumerate(features):
        for j, x in enumerate(xs):  
            js[j] += x * ds[c]      
    
    
    thetas = [t - alpha * j / n for t, j in zip(thetas, js)]
    return thetas


def generate_data(data, n, key_label):
    labels = list(data.keys())
    labels.remove(key_label)

    total_labels = len(labels)
    result = []
    # create n "not label" and code as y=0
    count = 1
    while count <= n:
        label = labels[count % total_labels]
        datum = blur(random.choice(data[label]))
        xs = datum[0:-1]
        result.append((xs, 0))
        count += 1

    # create n "label" and code as y=1
    for _ in range(n):
        datum = blur(random.choice(data[key_label]))
        xs = datum[0:-1]
        result.append((xs, 1))
    random.shuffle(result)
    return result


results = generate_data(clean_data, 5, "hills")
for result in results:
    print(result)


def learn_model(data: List[Tuple[List[float], int]], verbose: bool = False, epsilon: float = 1 * (10**-5), alpha: float = 0.1, print_freq: int = 1) -> List[float]:
    features, ys = parse_data(data)

    num_thetas = len(features[0])
    thetas: List[float] = init_thetas(num_thetas, -1, 1)

    prev_error: float = 0.0
    yhats: List[float] = calc_yhats(thetas, features)
    curr_error: float = calc_error(ys, yhats)

    if verbose:
        print("Iter\tError\tDelta")

    iterations = 0
    while abs(curr_error - prev_error) >= epsilon:
        thetas = calc_thetas(thetas, features, ys, yhats, alpha)
        prev_error = curr_error

        yhats: List[float] = calc_yhats(thetas, features)
        curr_error: float = calc_error(ys, yhats)

        if curr_error > prev_error:
            alpha /= 10

        if verbose and (iterations % print_freq == 0):
            print(f"{iterations}\t{curr_error:.6f}\t{abs(curr_error - prev_error):0.6}")

        iterations += 1

    return thetas


def apply_model(model: List[float], test_data: List[Tuple[List[float], int]]) -> List[Tuple[int, float]]:
    features, ys = parse_data(test_data)
    yhats = calc_yhats(model, features)
    result = [(y, yhat) for y, yhat in zip(ys, yhats)]
    return result


def evaluate(results: List[Tuple[int, float]]) -> Tuple[Dict[str, int], float]:
    confusion_matrix: Dict[str, int] = {"TN": 0, "TP": 0, "FN": 0, "FP": 0}

    threshold = 0.5
    for y, yhat in results:
        if yhat >= threshold and y == 1:
            confusion_matrix["TP"] += 1

        elif yhat < threshold and y == 0:
            confusion_matrix["TN"] += 1

        elif yhat >= threshold and y == 0:
            confusion_matrix["FP"] += 1

        elif yhat < threshold and y == 1:
            confusion_matrix["FN"] += 1

        else:
            raise Exception(f"Unable to classify yhat: {yhat}, y:{y}")

    # percent error rate
    percent_error = ((confusion_matrix["FN"] + confusion_matrix["FP"]) / len(results)) * 100 

    return confusion_matrix, percent_error


# ## Use your code

# Use `generate_data` to generate 100 blurred "hills" examples balanced with 100 "non hills" examples and use this as your test data. Print out the first 10 results, one per line.

train_data = generate_data(clean_data, 100, "hills")

for row in train_data[:10]:
    print(row)


test_data = generate_data(clean_data, 100, "hills")
for row in test_data[:10]:
    print(row)


# Use `learn_model` to learn a logistic regression model for classifying sensor images as "hills" or "not hills". Use your `generate_data` function to generate a training set of size 100 for "hills". **Set Verbose to True**

model = learn_model(train_data, True)


# Apply the model to the test data:

results = apply_model(model, test_data)


# Using the results above, print out your error rate (as a percent) and the confusion matrix:

confusion_matrix, percent_error = evaluate(results)
print(f"Error Rate: {percent_error}%")
print(f"Confusion Matrix: {confusion_matrix}")
