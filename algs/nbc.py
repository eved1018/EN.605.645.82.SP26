import json
import random
from copy import deepcopy
from math import inf, log2
from typing import Dict, List, NamedTuple, Tuple, Callable, Set


def parse_data(file_name: str) -> list[list]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = line.rstrip().split(",")
        data.append(datum)
    random.shuffle(data)
    return data


def create_folds(xs: list, n: int) -> list[list[list]]:
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def parse_attributes(filename: str) -> Tuple[Dict[str, Tuple[int, List[str]]], Dict[str, Dict[str, str]]]:
    abrv2fullname: Dict[str, Dict[str, str]] = {}
    attributes: Dict[str, Tuple[int, List[str]]] = {}

    with open(filename, "r") as fh:
        data: Dict[str, Dict[str, str]] = json.load(fh)

    for idx, (feature, attrs) in enumerate(data.items()):
        abrv2fullname[feature] = {}
        attributes[feature] = (idx, [])
        for name, code in attrs.items():
            attributes[feature][1].append(name)
            abrv2fullname[feature][code] = name

    return attributes, abrv2fullname


def split_features(attributes: Dict[str, Tuple[int, List[str]]], label: str) -> List[str]:
    return [i for i in attributes if i != label]


def rename_data(data: List[List[str]], attributes: Dict[str, Tuple[int, List[str]]], abrv2name: Dict[str, Dict[str, str]]) -> List[List[str]] | None:
    new_data = []
    for row in data:
        if len(row) != len(attributes):
            return None

        new_row = []
        for value, attr in zip(row, attributes):
            if attr in abrv2name and value in abrv2name[attr]:
                new_row.append(abrv2name[attr][value])
            else:
                return None
        new_data.append(new_row)

    return new_data


def subset_data(data: List[List[str]], column_idx: int, value: str, shallow: bool = False) -> List[List[str]]:
    if shallow:
        return [row for row in data if row[column_idx] == value]
    return [deepcopy(row) for row in data if row[column_idx] == value]


NBC = NamedTuple("NBC", [("prior", Dict[str, float]), ("pf", Dict[str, Dict[str, Dict[str, float]]])])


def naive_bayes_classifier(data: List[List[str]], features: List[str], attributes: Dict[str, Tuple[int, List[str]]], label: str, smoothing: bool = True, trace: bool = False) -> NBC:
    smooth_factor = 1 if smoothing else 0
    pf = {}
    priors = {}
    label_idx, labels = attributes[label]
    for label_value in labels:
        label_rows = subset_data(data, label_idx, label_value)
        pc = len(label_rows) / len(data)
        pf[label_value] = {}
        priors[label_value] = pc
        for feature in features:
            feature_idx, domain = attributes[feature]
            pf[label_value][feature] = {}
            for attr in domain:
                fi = subset_data(label_rows, feature_idx, attr)
                score = (len(fi) + smooth_factor) / (len(label_rows) + smooth_factor)
                pf[label_value][feature][attr] = score

    model = NBC(priors, pf)
    return model


def train(data: List[List[str]], features: List[str], attributes: Dict[str, Tuple[int, List[str]]], label: str, trace=False) -> NBC:
    return naive_bayes_classifier(data, features, attributes, label, trace)


def safe_div(x, y):
    if x == 0 or y == 0:
        return 0
    return x / y


def calculate_estimates(model: NBC, observation: List[str], features: List[str], labels: List[str]):
    estimates = {}
    total_probability = 0
    for label in labels:  # c = p(c) * PI(p(fi|c)) for fi in features for c in labels
        probability = model.prior[label]
        for feature, attr in zip(features, observation):
            probability *= model.pf[label][feature][attr]

        estimates[label] = probability
        total_probability += probability

    for label in labels:  # divide prob for for each label by sum of all prob
        prob = estimates[label]
        normalized_probability = prob / total_probability
        estimates[label] = normalized_probability

    estimated_label = max(estimates.items(), key=lambda x: x[1])[0]  # label with highest norm prob
    return estimated_label, estimates


def classify(model: NBC, observations: List[List[str]], features: List[str], labels: List[str]) -> List[Tuple[str, Dict[str, float]]]:
    classifications = []
    for row in observations:
        estimate_label, estimates = calculate_estimates(model, row, features, labels)
        classifications.append((estimate_label, estimates))
    return classifications


def evaluate(truth_set: List[str], classifications: List[Tuple[str, Dict[str, float]]], labels: List[str]) -> Tuple[int, Dict[str, int]]:
    errors = 0
    fold_cm: Dict[str, int] = {"TN": 0, "TP": 0, "FN": 0, "FP": 0}
    for true_label, estimate_prob in zip(truth_set, classifications):
        estimate = estimate_prob[0]  # highest probability label
        if true_label == estimate:
            if estimate == labels[1]:
                fold_cm["TP"] += 1
            else:
                fold_cm["TN"] += 1

        elif true_label != estimate:
            errors += 1
            if true_label == labels[0]:
                fold_cm["FP"] += 1
            else:
                fold_cm["FN"] += 1
    return errors, fold_cm


def divide_folds(data: List[List[str]], n_folds: int = 10) -> List[Tuple[List[List[str]], List[List[str]]]]:
    random.shuffle(data)
    folds = create_folds(data, n_folds)

    k_folds = []
    for idx, test_fold in enumerate(folds):
        training_set = []
        for idx2, train_fold in enumerate(folds):
            if idx == idx2:
                continue
            training_set.extend(train_fold)

        k_folds.append((training_set, test_fold))
    return k_folds


def cross_validate(
    data: List[List[str]], features: List[str], attributes: Dict[str, Tuple[int, List[str]]], label: str, train_fn: Callable = train, classify_fn: Callable = classify, eval_fn: Callable = evaluate, n_folds: int = 10, trace: bool = False
) -> Tuple[float, Dict[str, int]]:
    confusion_matrices: List[Dict[str, int]] = []
    total_errors = []

    label_idx, label_values = attributes[label]

    for k, (training_set, test_set) in enumerate(divide_folds(data, n_folds)):
        model = train_fn(training_set, features, attributes, label, trace)
        masked_test_set = [[i for c, i in enumerate(row) if c != label_idx] for row in test_set]
        classifications = classify_fn(model, masked_test_set, features, label_values)
        truth_set = [row[label_idx] for row in test_set]
        errors, cm = eval_fn(truth_set, classifications, label_values)
        error_rate = errors / len(test_set)
        total_errors.append(error_rate)
        confusion_matrices.append(cm)
        print(f"Fold {k}\nTraining size: {len(training_set)} | Test size: {len(test_set)}\nError rate: {error_rate}\nConfusion matrix:\nTP={cm['TP']}  FP={cm['FP']}\nFN={cm['FN']}  TN={cm['TN']}\n")
    total_error_rate = sum(total_errors) / len(total_errors)
    return total_error_rate, {k: sum([cm[k] for cm in confusion_matrices]) for k in confusion_matrices[0]}


def run_model(data: List[List[str]], attributes: Dict[str, Tuple[int, List[str]]], label: str, trace: bool = False):
    n_folds = 10

    features = split_features(attributes, label)

    avrg_error_rate, cm = cross_validate(data, features, attributes, label, n_folds=n_folds)
    print(f"\nTotal Confusion Matrix ({n_folds}-fold CV):")
    print(f"TP={cm['TP']}  FP={cm['FP']}\nFN={cm['FN']}  TN={cm['TN']}")
    print(f"\nAverage Error Rate ({n_folds}-fold CV): {avrg_error_rate}")

    model = train(data, features, attributes, label, trace=trace)
    assert model is not None

    print("\nNBC model:")


attributes = {"Shape": (0, ["round", "square"]), "Size": (1, ["large", "small"]), "Color": (2, ["blue", "green", "red"]), "Safe?": (3, ["yes", "no"])}
label = "Safe?"

data = [
    ["round", "large", "blue", "no"],
    ["square", "large", "green", "yes"],
    ["square", "small", "red", "no"],
    ["round", "large", "red", "yes"],
    ["square", "small", "blue", "no"],
    ["round", "small", "blue", "no"],
    ["round", "small", "red", "yes"],
    ["square", "small", "green", "no"],
    ["round", "large", "green", "yes"],
    ["square", "large", "green", "yes"],
    ["square", "large", "red", "no"],
    ["square", "large", "green", "yes"],
    ["round", "large", "red", "yes"],
    ["square", "small", "red", "no"],
    ["round", "small", "green", "no"],
]

run_model(data, attributes, label)

trace = False
data = parse_data("Module9/agaricus-lepiota-1-2.data")
attributes, abrv2name = parse_attributes("Module9/agaricus-lepiota-3.attrs.json")
label = "mushroom-type"

data = rename_data(data, attributes, abrv2name)
assert data is not None

run_model(data, attributes, label)
