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


def parse_attrs(filename):
    attrs = {}
    with open(filename, "r") as fh:
        attrs = json.load(fh)
    return attrs


### Tree

#  node.attr = attr
#  node.children = [nodes]



def create_node(tree, attribute=None, label=None):
    node_id = len(tree["nodes"])
    tree["nodes"].append(node_id)
    tree["attributes"][node_id] = attribute
    tree["labels"][node_id] = label
    return node_id


def add_child(tree, parent_id, edge_value, child_id):
    tree["edges"][(parent_id, child_id)] = edge_value


def get_children(tree, parent):
    return [child[1] for child in tree["edges"] if child[0] == parent]

def pretty_print_tree(tree):
    paths = {tree["root"]: []}
    frontier = [tree["root"]]

    while frontier:
        node = frontier.pop()
        attr = tree["attributes"][node]
        children = get_children(tree, node)

        if not children:
            print(" -> ".join(paths[node]) + f" -> | {tree['labels'][node]} |")
            continue

        for child in reversed(children):
            edge_value = tree["edges"][(node, child)]
            paths[child] = paths[node] + [f"{attr} = {edge_value}"]
            frontier.append(child)

#### Data Parsing
def parse_data(file_name: str) -> list[list]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = line.rstrip().split(",")
        data.append(datum)
    random.shuffle(data)
    return data

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


# def calculate_entropy(data, attr, attr_index, attributes, label_index, labels):
#     total_size = len(data)
#     entropy = 0
#     for feature in attributes[attr]:
#         subset = [i for i in data if i[attr_index] == feature]
#         subset_size = len(subset)
#         subset_entropy = 0
#         for label in labels:
#             p = len([i for i in subset if i[label_index] == label])
#             p = p / subset_size
#             subset_entropy += -1 * p * log2(p)
#         entropy += (subset_size/total_size) * subset_entropy
#     return entropy


def calculate_entropy(data, attr_index, attr, attributes, label_index, labels):
    subset_sizes = {i: 0 for i in attributes[attr]}
    label_counts = {l: {a: 0 for a in attributes[attr]} for l in labels}
    total_size = 0

    for row in data:
        observation = row[attr_index]
        label = row[label_index]
        subset_sizes[observation] += 1
        label_counts[label][observation] += 1
        total_size += 1

    entropy = 0
    for feature in attributes[attr]:
        subset_size = subset_sizes[feature]
        subset_entropy = 0
        for label in labels:
            p = label_counts[label][feature]
            p = p / subset_size
            if p <= 0.0:
                continue
            subset_entropy += -1 * p * log2(p)
        entropy += (subset_size / total_size) * subset_entropy
    return entropy


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

def id3(data, features, attributes, label_index, labels, default, tree, trace = False):
    if trace:
        print(features, attributes)
    if len(data) == 0:
        return create_node(tree, label=default)

    if is_homogeneous(data, label_index):
        return create_node(tree, label=get_label(data, label_index))

    if len(attributes) == 0:
        return create_node(tree, label=get_majority_label(data, label_index))
    
    index, attr = pick_best_attribute(data, features, attributes, label_index, labels)
    node = create_node(tree, attribute=attr)
    default_label = get_majority_label(data, label_index)
    for value in domain(attributes, attr):
        subset = get_subset(data, index, value)
        new_features = remove_features(features, index)
        child = id3(subset, new_features, attributes, label_index, labels, default_label, tree, trace)
        add_child(tree, node, value, child)
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

def build_tree(data, attributes, label_index, labels):
    features = {c:k for c, k in enumerate(attributes)}
    tree = {"nodes": [], "edges": {}, "attributes": {}, "labels": {}, "root": None}
    root = id3(data, features, attributes, label_index, labels, get_majority_label(data, label_index), tree, True)
    tree["root"] = root
    return tree

# data = parse_data("Module8/agaricus-lepiota-3.data")
# attrs = parse_attrs("Module8/agaricus-lepiota-3.attrs")
# print(len(data))

# id3(data, attrs, "p", {"nodes": [], "edges": []})

feature_names = ["Shape", "Size", "Color", "Safe?"]
attributes = {
    "Shape": ["round", "square"],
    "Size":  ["large", "small"],
    "Color": ["blue", "green", "red"],
}
label_index = 3
labels = ["yes", "no"]

data = [
    ["round", "large", "blue",  "no"],
    ["square","large", "green", "yes"],
    ["square","small", "red",   "no"],
    ["round", "large", "red",   "yes"],
    ["square","small", "blue",  "no"],
    ["round", "small", "blue",  "no"],
    ["round", "small", "red",   "yes"],
    ["square","small", "green", "no"],
    ["round", "large", "green", "yes"],
    ["square","large", "green", "yes"],
    ["square","large", "red",   "no"],
    ["square","large", "green", "yes"],
    ["round", "large", "red",   "yes"],
    ["square","small", "red",   "no"],
    ["round", "small", "green", "no"],
]

tree = build_tree(data, attributes, label_index, labels)
pretty_print_tree(tree)