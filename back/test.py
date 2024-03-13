from functools import reduce
from loader import DefaultLoader
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

def Fn(y_pred, y_true):
    """
    Calculate the number of false negatives.

    Parameters:
    y_pred (list): Predicted labels.
    y_true (list): True labels.

    Returns:
    int: Number of false negatives.
    """
    return reduce(lambda a, e: a if e in y_pred else a + 1, y_true, 0)

def Fp(y_pred, y_true):
    """
    Calculate the number of false positives.

    Parameters:
    y_pred (list): Predicted labels.
    y_true (list): True labels.

    Returns:
    int: Number of false positives.
    """
    return reduce(lambda a, e: a if e in y_true else a + 1, y_pred, 0)

def Tn(y_pred, y_true, docs):
    """
    Calculate the number of true negatives.

    Parameters:
    y_pred (list): Predicted labels.
    y_true (list): True labels.
    docs (list): Documents.

    Returns:
    int: Number of true negatives.
    """
    return reduce(lambda a, e: a if e in y_pred or e in y_true else a + 1, docs, 0)

def Tp(y_pred, y_true):
    """
    Calculate the number of true positives.

    Parameters:
    y_pred (list): Predicted labels.
    y_true (list): True labels.

    Returns:
    int: Number of true positives.
    """
    return reduce(lambda a, e: a if not e in y_pred else a + 1, y_true, 0)

def accuracy(y_pred, y_true):
    """
    Calculate the accuracy of predictions.

    Parameters:
    y_pred (list): Predicted labels.
    y_true (list): True labels.

    Returns:
    float: Accuracy of predictions.
    """
    tp = Tp(y_pred, y_true)
    fp = Fp(y_pred, y_true)
    sm = tp + fp

    return 0 if sm == 0 else tp / sm

def fallout(y_pred, y_true, docs):
    """
    Calculate the fallout of predictions.

    Parameters:
    y_pred (list): Predicted labels.
    y_true (list): True labels.
    docs (list): Documents.

    Returns:
    float: Fallout of predictions.
    """
    fp = Fp(y_pred, y_true)
    tn = Tn(y_pred, y_true, docs)
    sm = fp + tn

    return 0 if sm == 0 else fp / sm

def fb(y_pred, y_true, b=3, r=30):
    """
    Calculate the F-beta score of predictions.

    Parameters:
    y_pred (list): Predicted labels.
    y_true (list): True labels.
    b (int): Beta parameter.
    r (int): Number of documents to consider.

    Returns:
    float: F-beta score of predictions.
    """
    p = r_accuracy(y_pred, y_true, r)
    r = recall(y_pred, y_true)

    n = (1 + b ** 2) * p * r
    d = (b ** 2) * p + r

    return 0 if d == 0 else n / d

def f1(y_pred, y_true):
    """
    Calculate the F1 score of predictions.

    Parameters:
    y_pred (list): Predicted labels.
    y_true (list): True labels.

    Returns:
    float: F1 score of predictions.
    """
    p = accuracy(y_pred, y_true)
    r = recall(y_pred, y_true)

    n = 2 * p * r
    d = p + r

    return 0 if d == 0 else n / d

def r_accuracy(y_pred, y_true, r=15):
    """
    Calculate the recall-based accuracy of predictions.

    Parameters:
    y_pred (list): Predicted labels.
    y_true (list): True labels.
    r (int): Number of documents to consider.

    Returns:
    float: Recall-based accuracy of predictions.
    """
    y_pred = list(y_pred)
    subset = set(y_pred[:r])

    return accuracy(subset, y_true)

def recall(y_pred, y_true):
    """
    Calculate the recall of predictions.

    Parameters:
    y_pred (list): Predicted labels.
    y_true (list): True labels.

    Returns:
    float: Recall of predictions.
    """
    tp = Tp(y_pred, y_true)
    fn = Fn(y_pred, y_true)
    sm = tp + fn

    return 0 if sm == 0 else tp / sm

def programs():
    """
    Main program to load data, calculate metrics, and print results.
    """
    loader = DefaultLoader()

    loader.load_boolean_corpora()

    dataset = loader.dataset
    documents = loader.boolean_ids

    metricers = [
        ('accuracy', lambda y_pred, y_true: accuracy(y_pred, y_true)),
        ('f1', lambda y_pred, y_true: f1(y_pred, y_true)),
        ('fallout', lambda y_pred, y_true: fallout(y_pred, y_true, documents)),
        ('fb', lambda y_pred, y_true: fb(y_pred, y_true)),
        ('r-accuracy', lambda y_pred, y_true: r_accuracy(y_pred, y_true)),
        ('recall', lambda y_pred, y_true: recall(y_pred, y_true)),
    ]

    searchers = [
        ('boolean', lambda q: loader.search_boolean(q)),
        ('extended', lambda q: loader.search_extended(q)),
        ('lsi', lambda q: loader.search_lsi(q)),
    ]

    queries = {}

    for rel in dataset.qrels_iter():
        if not queries.get(rel.query_id):
            queries[rel.query_id] = set([rel.doc_id])
        else:
            queries[rel.query_id].add(rel.doc_id)

    metrics = [{name: [] for name, _ in metricers} for i in range(len(searchers))]

    for k, query in enumerate(dataset.queries_iter()):
        matches = [set(search(query.text)) for name, search in searchers]
        gotchas = [False for i in range(len(matches))]
        expected = queries[query.query_id]

        gotchas = [not set.isdisjoint(match, expected) for match in matches]
        messages = [f'{"ok!" if gotcha else "missed!"}' for gotcha in gotchas]
        debug = any([not got for got in gotchas])

        print(f'testing query {query.query_id}, { " ".join(messages) }, {"" if not debug else f"query: {query.text}"}')

        for metric, y_pred in zip(metrics, matches):
            for name, func in metricers:
                metric[name].append(func(y_pred, expected))

        if (k > 10): break

    for metric in metrics:
        for name, _ in metricers:
            acc = metric[name]
            metric[name] = sum(acc) / len(acc)

    for (model_name, _), metric in zip(searchers, metrics):
        print('')
        print(f'Metrics: ({model_name} model)')

        for metric_name, value in metric.items():
            print(f'{metric_name}: {value}')

programs()
