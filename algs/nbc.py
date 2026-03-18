from typing import List, Set, Dict, Tuple


def evaluate(truth_set: List[str], classifications: List[str], labels: List[str]) -> Tuple[int, Dict[str, int]]:
    errors = 0
    fold_cm: Dict[str, int] = {"TN": 0, "TP": 0, "FN": 0, "FP": 0}
    for true_label, estimate_prob in zip(truth_set, classifications):
        estimate = estimate_prob[0]
        print(true_label, estimate)

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


def train(data, attributes, label):
    pf = {}
    pcs = {}
    label_idx, labels = attributes[label]
    for label in labels:
        label_rows = [row for row in data if row[label_idx] == label]
        pc = len(label_rows) / len(data)
        pf[label] = {}
        pcs[label] = pc
        for feature, (feature_idx, domain) in attributes.items():
            pf[label][feature] = {}
            for attr in domain:
                fi = [row for row in label_rows if row[feature_idx] == attr]
                pf[label][feature][attr] = (len(fi) + 1) / (len(label_rows) + 1)

    return pf, pcs


def classify(data, attributes, label, pf, pc):
    classification = []
    label_idx, labels = attributes[label]
    for row in data:
        estimates = {}
        for label in labels:
            c = pc[label]
            for feature, attr in zip(attributes, row):
                c *= pf[label][feature][attr]
            estimates[label] = c
        estimate_label = max(estimates.items(), key=lambda x: x[1])[0]
        classification.append((estimate_label, estimates))
    return classification


def run_model(data, attributes, label):
    split = len(data) // 2
    training_set = data[0:split]
    test_set = data[split + 1 :]
    pf, pc = train(training_set, attributes, label)
    classification = classify(test_set, attributes, label, pf, pc)
    print(classification)
    label_idx, labels = attributes[label]
    truth_set = [row[label_idx] for row in test_set]
    errors, cm = evaluate(truth_set, classification, labels)
    k = 0
    print(f"Fold {k}\nTraining size: {len(training_set)} | Test size: {len(test_set)}\nError rate: {errors / len(test_set)}\nConfusion matrix:\nTP={cm['TP']}  FP={cm['FP']}\nFN={cm['FN']}  TN={cm['TN']}\n")


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
