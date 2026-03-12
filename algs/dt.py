from copy import deepcopy
from math import inf, log2
import random
import json
from collections import OrderedDict

"""
1 def id3( data, attributes, default)  
2   if data is empty, return default  
3   if data is homogeneous, return class label.  
4   if attributes is empty, return majority-label( data)  
5   best_attr = pick_best_attribute( data, attributes)  
6   node = create_node( best_attribute)  
7   default_label = majority-label( data)  
8   for value in the domain of best_attr  
9     subset = examples in data where best_attr == value  
10     child = id3( subset, attributes - best_attr, default_label)  
11     add child to node  
12   end  
13   return node  
14 end 
"""


### Tree


def create_node(tree, attribute):
    if "nodes" not in tree:
        tree["nodes"] = set()
    tree["nodes"].add(attribute)
    return attribute


def add_child(tree, parent, child, value):
    if "edges" not in tree:
        tree["edges"] = {}

    if parent not in tree["edges"]:
        tree["edges"][parent] = {}

    tree["edges"][parent][value] = child
    return tree


def get_children(tree, node):
    return tree["edges"].get(node, {}).items()


def pretty_print_tree(root, tree):  # DFS
    if "nodes" not in tree or "edges" not in tree:
        raise Exception("Tree does not contain keys ['nodes', 'edges']")
    if len(tree["nodes"]) == 0:
        raise Exception("No nodes in tree")

    rows = []
    frontier = [(root, [])]

    while frontier:
        parent, path = frontier.pop()
        children = get_children(tree, parent)

        if not children:
            decision = []
            for feature, attr in path:
                decision.append(f"{feature}: {attr} ->")
            decision.append(f"| {parent} |")
            rows.append(" ".join(decision))

        for attr, child in children:
            child_path = path + [(parent, attr)]
            frontier.append((child, child_path))

    print("\n".join(rows))
    return


#### Data Parsing
def parse_data(file_name: str) -> list[list]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = line.rstrip().split(",")
        data.append(datum)
    random.shuffle(data)
    return data


def parse_attrs(filename):
    attr_map = {}
    attributes = {}
    with open(filename, "r") as fh:
        data = json.load(fh)
    for feature, attr in data.items():
        attr_map[feature] = {}
        attributes[feature] = []
        for single_letter, name in attr.items():
            attributes[feature].append(name)
            attr_map[feature][single_letter] = name
    return attributes, attr_map


def rename_data(data, attributes, attr_map):
    return [[attr_map[attr][i] for i, attr in zip(row, attributes)] for row in data]


### Alg
def is_homogeneous(data, label_index):
    if len(set(row[label_index] for row in data)) == 1:
        return True
    return False


def get_label(data, label_index) -> str:
    return data[0][label_index]


def get_majority_label(data, label_index):
    counts = {}
    for row in data:
        label = row[label_index]
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1

    return max(counts, key=lambda k: counts[k])


def calculate_entropy(data, attr_index, attr, attributes, label_index, labels):
    total_size = len(data)
    entropy = []
    for feature in attributes[attr]:
        subset = [i for i in data if i[attr_index] == feature]
        subset_size = len(subset)
        subset_entropy = 0
        for label in labels:
            l = len([i for i in subset if i[label_index] == label])
            if subset_size == 0:
                # print(f"[Entropy] {feature=} {attr=} {label=} {l=} NA NA {subset_entropy}")
                continue
            p = l / subset_size
            if p <= 0.0:
                # print(f"[Entropy] {feature=} {attr=} {label=} {l=} {p=} NA {subset_entropy}")
                se = 0.0
            else:
                se = -1 * p * log2(p)

            subset_entropy += se
            # print(f"[Entropy] {feature=} {attr=} {label=} {l=} {subset_size=} {p=} {se=} {subset_entropy=}")
            e = (subset_size/total_size) * subset_entropy
            entropy.append(round(e, 3))
            ee  = sum(entropy)
            print(f"[Subset Entropy] {attr=} {feature=} {label=} {l=} {subset_size=} {p=} {se=} {subset_size}/{total_size}*{subset_entropy:.3} = {e:.3}")
    print()
    entropy_sum = sum(entropy)
    print(f"[Entropy] {attr} {entropy} {entropy_sum:.3}\n")
    return entropy_sum


# def calculate_entropy(data, attr_index, attr, attributes, label_index, labels):
#     subset_sizes = {i: 0 for i in attributes[attr]}
#     label_counts = {l: {a: 0 for a in attributes[attr]} for l in labels}
#     total_size = 0

#     for row in data:
#         observation = row[attr_index]
#         label = row[label_index]
#         subset_sizes[observation] += 1
#         label_counts[label][observation] += 1
#         total_size += 1

#     entropy = 0
#     for feature in attributes[attr]:
#         subset_size = subset_sizes[feature]
#         subset_entropy = 0
#         for label in labels:
#             p = label_counts[label][feature]
#             if subset_size == 0:
#                 continue
#             p = p / subset_size
#             if p <= 0.0:
#                 continue
#             subset_entropy += -1 * p * log2(p)
#         entropy += (subset_size / total_size) * subset_entropy
#     return entropy


def pick_best_attribute(data, features, attributes, label_index, labels):
    best_attr = None
    best_entropy = inf
    best_index = -1

    for index, attr in features.items():
        entropy = calculate_entropy(data, index, attr, attributes, label_index, labels)

        if entropy < best_entropy:
            best_attr = attr
            best_entropy = entropy
            best_index = index

    return best_index, best_attr


def domain(attributes, feature):
    return attributes[feature]


def get_subset(data, attr_index, attr):
    return [deepcopy(row) for row in data if row[attr_index] == attr]


def remove_features(features, attr_index):
    new_features = deepcopy(features)
    new_features.pop(attr_index)
    return new_features


# node should always be a label or a feature (not observation)


def id3(data, features, attributes, label_index, labels, default, tree, trace=False):
    if trace:
        print(f"[DEBUG] {features=}, {attributes=}")

    if len(data) == 0:
        print("[DEBUG] Base Case - empty data")
        return create_node(tree, default)

    if is_homogeneous(data, label_index):
        print("[DEBUG] Base Case - homogenous data")
        return create_node(tree, get_label(data, label_index))

    if len(features) == 0:
        print("[DEBUG] Base Case - empty features")
        return create_node(tree, get_majority_label(data, label_index))
    print()
    index, attr = pick_best_attribute(data, features, attributes, label_index, labels)
    print(f" | Lowest Entropy Attr: {index=} {attr=}")

    node = create_node(tree, attribute=attr)
    default_label = get_majority_label(data, label_index)
    for value in domain(attributes, attr):
        subset = get_subset(data, index, value)
        print(f"Partitioning {value=} from {attr=}")
        new_features = remove_features(features, index)
        child = id3(subset, new_features, attributes, label_index, labels, default_label, tree, trace)
        tree = add_child(tree, node, child, value)
    return node


### Testing
def create_folds(xs: list, n: int) -> list[list[list]]:
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def train(training_data):
    decision_tree = None
    return decision_tree


def classify(tree, observations):
    classifications = []
    return classifications


def cross_validate(data):
    return


def build_tree(data, attributes, label, trace= False):
    features = {}
    labels = None
    label_index = -1
    for index, attr in enumerate(attributes):
        if attr == label:
            labels = attributes[attr]
            label_index = index
        else:
            features[index] = attr
    tree = {}
    root = id3(data, features, attributes, label_index, labels, get_majority_label(data, label_index), tree, trace)
    return root, tree

def test():
    data = parse_data("Module8/agaricus-lepiota-3.data")
    attributes, attr_map = parse_attrs("Module8/agaricus-lepiota-3.attrs")
    data = rename_data(data, attributes, attr_map)
    root, tree = build_tree(data, attributes, "mushroom-type")
    pretty_print_tree(root, tree)


def self_check():
    attributes = {
        "Shape": ["round", "square"],
        "Size": ["large", "small"],
        "Color": ["blue", "green", "red"],
        "Safe?": ["yes", "no"]
    }
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

    root, tree = build_tree(data, attributes, label, trace=True)
    # print(root, tree)
    pretty_print_tree(root, tree)


self_check()
# test()
