import numpy as np


def precision(act, perfect_act, threshold=0.5):
    perfect_bool = perfect_act > 0.5
    positives = act >= threshold
    n_positives = np.count_nonzero(positives)
    n_true_positives = np.count_nonzero(perfect_bool & positives)
    return (n_true_positives / n_positives) if n_positives else 0.


def recall(act, perfect_act, threshold=0.5):
    perfect_bool = perfect_act > 0.5
    n_true = np.count_nonzero(perfect_bool)
    positives = act >= threshold
    n_true_positives = np.count_nonzero(perfect_bool & positives)
    return n_true_positives / n_true