def precision_recall(spans_true, spans_pred):
    indices_true = as_indices(spans_true)
    indices_pred = as_indices(spans_pred)

    tp = len(indices_true.intersection(indices_pred))
    fp = len(indices_pred.difference(indices_true))
    fn = len(indices_true.difference(indices_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall
    

def as_indices(spans):
    indices = set()
    for start, end in spans:
        indices.update(range(start, end))
    return indices