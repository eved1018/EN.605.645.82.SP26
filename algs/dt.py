import json
import random
from copy import deepcopy
from math import inf, log2
from typing import Callable, Dict, List, NamedTuple, Set, Tuple


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


Node = NamedTuple("Node", [("value", str), ("children", List)])


def create_node(value: str) -> Node:
    return Node(value, [])


def add_child(parent: Node, child: Node, edge_value: str) -> Node:
    if any(edge_value == v for v, _ in parent.children):
        return parent

    parent.children.append((edge_value, child))
    return parent


def pretty_print_tree(node: Node, indent: int = 0):
    tabs = " " * indent
    print(f"{tabs} - {node.value}")  # parent (attr to be split)
    for attribute, child in node.children:
        tabs = " " * (indent + 4)
        if not child.children:
            print(f"{tabs} | {attribute} -> {child.value}")  # leaf node - value is label
        else:
            print(f"{tabs} | {attribute}")  # internal node - just print partitioned attribute
            pretty_print_tree(child, indent + 8)
    return


def traverse_tree(node: Node, observation: List[str], feature_indices: Dict[str, int]) -> str | None:
    if len(node.children) == 0:
        return node.value  # label since node is leaf

    feature = node.value  # feature since node is not leaf
    if feature not in feature_indices:
        return None

    column_idx = feature_indices[feature]
    observed_attr = observation[column_idx]
    for attribute, child in node.children:
        if attribute == observed_attr:
            return traverse_tree(child, observation, feature_indices)
    return None


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


def split_features(attributes: Dict[str, Tuple[int, List[str]]], label: str) -> Set[str]:
    return {i for i in attributes if i != label}


def get_homogeneous_label(data: List[List[str]], label_idx: int) -> str | None:
    if len(data) == 0:
        return None

    if not all(len(row) and 0 <= label_idx < len(row) for row in data):
        return None

    if len(set(row[label_idx] for row in data)) == 1:
        return data[0][label_idx]
    return None


def get_majority_label(data: List[List[str]], label_idx: int, trace: bool = False) -> str | None:
    if not data or not all(0 <= label_idx < len(row) for row in data):
        return None

    counts = {}
    for row in data:
        label = row[label_idx]
        if label not in counts:
            counts[label] = 1
        else:
            counts[label] += 1

    label = max(counts, key=lambda k: counts[k])

    print(f"Majority label: {label}") if trace else None
    return label


def subset_data(data: List[List[str]], column_idx: int, value: str, shallow: bool = False) -> List[List[str]]:
    if shallow:
        return [row for row in data if row[column_idx] == value]
    return [deepcopy(row) for row in data if row[column_idx] == value]


def calculate_entropy(data: List[List[str]], feature: str, attributes: Dict[str, Tuple[int, List[str]]], label: str) -> float | None:
    if feature not in attributes or label not in attributes:
        return None

    total_entropy, total_size = 0, len(data)
    feature_idx, attribute_list = attributes[feature]
    label_idx, label_list = attributes[label]

    for attribute in attribute_list:
        s_a = subset_data(data, feature_idx, attribute, True)
        if len(s_a) == 0:
            continue

        probabilities = [len(subset_data(s_a, label_idx, label_value, True)) / len(s_a) for label_value in label_list]
        entropy = sum(-1 * p_l * log2(p_l) for p_l in probabilities if p_l > 0.0)
        total_entropy += (len(s_a) / total_size) * entropy
    return total_entropy


def pick_best_feature(data: List[List[str]], features: Set[str], attributes: Dict[str, Tuple[int, List[str]]], label: str, trace: bool = False) -> str | None:
    best_feature, best_entropy = None, inf

    for feature in features:
        entropy = calculate_entropy(data, feature, attributes, label)
        if entropy is None:
            print(f"entropy of feature {feature} is None") if trace else None
            continue

        print(f"entropy of feature {feature} = {entropy:.3f}") if trace else None

        if entropy < best_entropy:
            best_feature = feature
            best_entropy = entropy

    if trace:
        print(f"Highest entropy feature {best_feature} = {best_entropy:.3f}") if best_feature else print("Highest entropy feature not found")
    return best_feature


def domain(attributes: Dict[str, Tuple[int, List[str]]], feature: str, nans: List[str] | None = None) -> List[str]:
    if nans is None:
        nans = ["?"]
    return [i for i in attributes[feature][1] if i not in nans]


def remove_feature(features: Set[str], feature: str) -> Set[str]:
    return {f for f in features if f != feature}


def get_leaf_node(data: List[List[str]], features: Set[str], label_index: int, default_label: str, trace: bool = False) -> Node | None:
    if len(data) == 0:
        print("Base Case - empty data") if trace else None
        return create_node(default_label)

    homogenous_label = get_homogeneous_label(data, label_index)
    if homogenous_label is not None:
        print("Base Case - homogenous data") if trace else None
        return create_node(homogenous_label)

    if len(features) == 0:
        print("Base Case - empty features") if trace else None
        majority_label = get_majority_label(data, label_index)
        return create_node(majority_label)  # type: ignore - we already tested if data is empty
    return None


def id3(data: List[List[str]], features: Set[str], attributes: Dict[str, Tuple[int, List[str]]], label: str, default_label: str, trace: bool = False) -> Node:
    print(f"{features=}, {attributes=}") if trace else None
    result = get_leaf_node(data, features, attributes[label][0], default_label, trace)
    if result is not None:
        return result

    feature = pick_best_feature(data, features, attributes, label, trace)
    if feature is None:
        raise Exception(f"Cannot find a feature to partition")

    majority_label = get_majority_label(data, attributes[label][0], trace)
    if majority_label is None:
        raise Exception(f"Cannot find a majority label")

    node = create_node(feature)
    for value in domain(attributes, feature):
        subset = subset_data(data, attributes[feature][0], value)
        child = id3(subset, remove_feature(features, feature), attributes, label, majority_label, trace)
        node = add_child(node, child, value)
    return node


def train(data: List[List[str]], features: Set[str], attributes: Dict[str, Tuple[int, List[str]]], label: str, trace=False) -> Node | None:
    default_label = get_majority_label(data, attributes[label][0], trace)
    if default_label is None:
        raise Exception("default label is None")
    return id3(data, features, attributes, label, default_label, trace)


def classify(tree: Node, observations: List[List[str]], feature_indices: Dict[str, int]) -> List[str]:
    classifications = []
    for row in observations:
        label = traverse_tree(tree, row, feature_indices)
        classifications.append(label)
    return classifications


def evaluate(truth_set: List[str], classifications: List[str], labels: List[str]) -> Tuple[int, Dict[str, int]]:
    errors = 0
    fold_cm: Dict[str, int] = {"TN": 0, "TP": 0, "FN": 0, "FP": 0}
    for true_label, estimate in zip(truth_set, classifications):
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
    data: List[List[str]], features: Set[str], attributes: Dict[str, Tuple[int, List[str]]], label: str, train_fn: Callable = train, classify_fn: Callable = classify, eval_fn: Callable = evaluate, n_folds: int = 10, trace: bool = False
) -> Tuple[float, Dict[str, int]]:
    confusion_matrices: List[Dict[str, int]] = []
    total_errors = []
    feature_indices = {feature: idx for feature, (idx, _) in attributes.items()}
    label_idx, label_values = attributes[label]

    for k, (training_set, test_set) in enumerate(divide_folds(data, n_folds)):
        tree = train_fn(training_set, features, attributes, label, trace)
        classifications = classify_fn(tree, test_set, feature_indices)
        truth_set = [row[label_idx] for row in test_set]
        errors, cm = eval_fn(truth_set, classifications, label_values)
        total_errors.append(errors)
        confusion_matrices.append(cm)
        print(f"Fold {k}\nTraining size: {len(training_set)} | Test size: {len(test_set)}\nError rate: {errors / len(test_set)}\nConfusion matrix:\nTP={cm['TP']}  FP={cm['FP']}\nFN={cm['FN']}  TN={cm['TN']}\n")

    error_rate = sum(total_errors) / len(total_errors)
    return error_rate, {k: sum([cm[k] for cm in confusion_matrices]) for k in confusion_matrices[0]}


def run_model(data: List[List[str]], attributes: Dict[str, Tuple[int, List[str]]], label: str, trace: bool = False):
    n_folds = 10

    features = split_features(attributes, label)

    avrg_error_rate, cm = cross_validate(data, features, attributes, label, n_folds=n_folds)
    print(f"\nTotal Confusion Matrix ({n_folds}-fold CV):")
    print(f"TP={cm['TP']}  FP={cm['FP']}\nFN={cm['FN']}  TN={cm['TN']}")
    print(f"\nAverage Error Rate ({n_folds}-fold CV): {avrg_error_rate:0.4f}")

    tree = train(data, features, attributes, label, trace=trace)
    assert tree is not None

    print("\nDecision Tree:")
    pretty_print_tree(tree)


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
data = parse_data("Module8/agaricus-lepiota-3.data")
attributes, abrv2name = parse_attributes("Module8/agaricus-lepiota-3.attrs.json")
label = "mushroom-type"

data = rename_data(data, attributes, abrv2name)
assert data is not None

run_model(data, attributes, label)
