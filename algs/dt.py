from copy import deepcopy
import random


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

#  node.attr = attr
#  node.children = [nodes]

def create_node(attr):
    node = {"attr": attr, "children": []}
    return node


def add_child(node, child):
    node["children"].append(child)
    return node


def pretty_print_tree(tree):
    # BFS 
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


### Alg
def is_homogeneous(data):
    return False


def get_label(data):
    return


def get_majority_label(data):
    return


def pick_best_attribute(data, attributes):
    return


def domain(attr):
    return []


def get_subset(data, value):
    return


def remove_attrs(attributes, attr):
    # use deepcopy
    return


def id3(data, attributes, default):
    if len(data) == 0:
        return default

    if is_homogeneous(data):
        return get_label(data)

    if len(attributes) == 0:
        return get_majority_label(data)

    best_attr = pick_best_attribute(data, attributes)
    node = create_node(best_attr)
    default_label = get_majority_label(data)
    for value in domain(best_attr):
        subset = get_subset(data, value)
        new_attributes = remove_attrs(attributes, best_attr)
        child = id3(subset, new_attributes, default_label)
        node = add_child(node, child)
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
