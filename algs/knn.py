import random
from typing import List, Dict, Tuple, Callable
from math import sqrt


def parse_data(file_name: str) -> List[List]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data


def create_folds(xs: List, n: int) -> List[List[List]]:
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def create_train_test(folds: List[List[List]], index: int) -> Tuple[List[List], List[List]]:
    training = []
    test = []
    for i, fold in enumerate(folds):
        if i == index:
            test = fold
        else:
            training = training + fold
    return training, test


"""
## Problem 1: kNN
Implement k Nearest Neighbors algorithm with k = 9.    
"""


def get_features(observation: List[float], label_idx: int = -1):
    if label_idx < 0:
        label_idx = len(observation) - 1
    return observation[:label_idx] + observation[label_idx + 1 :]


def average_label(nearest: List[Tuple[float, List[float]]], label_idx: int = -1):
    return sum([example[label_idx] for _, example in nearest]) / len(nearest)


def majority_label(nearest: List[Tuple[float, List[float]]], label_idx: int = -1):
    counter = {}
    for _, observation in nearest:
        label = observation[label_idx]
        if label in counter:
            counter[label] += 1
        else:
            counter[label] = 1
    return max(counter.items(), key=lambda x: x[1])[0]


def euclidean_distance(example: List[float], query: List[float], label_idx: int = -1):
    # print(example, query, label_idx)
    if label_idx < 0:
        label_idx = len(example) - 1
    distance = 0
    for i, (x, y) in enumerate(zip(example, query)):
        if i == label_idx:
            continue
        distance += (x - y) ** 2
    return sqrt(distance)


def print_distances(distances: List[Tuple[float, List[float]]], k: int, label_idx: int = -1):
    for i, (dist, example) in enumerate(distances):
        if i == k:
            print("-----" * 10)
        print(i, example, dist)


def knn(
    observations: List[List[float]],
    query: List[float],
    k: int,
    label_idx: int = -1,
    distance: Callable = euclidean_distance,
    processing: Callable = average_label,
    debug: bool = False,
):
    distances = [(distance(example, query, label_idx), example) for example in observations]
    distances = sorted(distances, key=lambda x: x[0])
    print_distances(distances, k, label_idx) if debug else None
    return processing(distances[:k], label_idx)


"""
## Problem 2: Evaluation vs. The Mean
Using Mean Squared Error (MSE) as your evaluation metric, evaluate your implement above and the Null model, the mean. 
"""


def evaluate(test, train, k: int, label_idx: int = -1):
    squared_error = 0
    for query in test:
        result = knn(train, query, k)
        squared_error += (query[label_idx] - result) ** 2
    return squared_error / len(test)


def cross_validate(dataset, k: int, label_idx: int = -1, folds: int = 10):
    for i in range(folds):
        train, test = create_train_test(dataset, i)
        mse = evaluate(test, train, k, label_idx)
        mean = sum([row[label_idx] for row in train]) / len(train)
        null_mse = sum((query[label_idx] - mean) ** 2 for query in test) / len(test)
        r2 = 1 - (mse/null_mse)
        print(i, mse, mean, r2)
    return mse

"""
## Problem 3: Hyperparameter Tuning
Tune the value of k.
"""

def hyperparameter_tuning(dataset, low_k, high_k, label_idx = -1, folds = 10):
    errors = {k:[] for k in range(low_k, high_k, 2)}
    for i in range(folds):
        train, test = create_train_test(dataset, i)
        for k in range(low_k, high_k, 2):
            error = evaluate(test, train, k)
            errors[k].append(error)
    for k, es in errors.items():
        a = sum(es) / len(es)
        print(k,a)
    return min(errors, key=lambda k: sum(errors[k]) / len(errors[k]))



def test():
    data = parse_data("Module12/concrete_compressive_strength-3.csv")
    folds = create_folds(data, 10)
    train, test = create_train_test(folds, 0)
    result = knn(train, test[0], 9)
    print(result, test[0][-1])
    print()
    cross_validate(folds, 9)
    print()
    k = hyperparameter_tuning(folds, 1, 13)
    print(f"best k={k}")


def test2():
    data = [[4.9, 3.0, 1.4, 0.2, "satosa"], [5.3, 3.7, 1.5, 0.2, "satosa"], [7.0, 3.2, 4.7, 1.4, "versicolor"], [4.9, 2.5, 4.5, 1.7, "virginica"]]
    query = [6.0, 2.7, 5.1, 1.6, "?"]

    r = knn(data, query, k=1, processing=majority_label, debug=True)
    print(r)


test()
# test2()
