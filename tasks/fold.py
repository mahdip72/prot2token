import numpy as np
import pandas as pd
from utils.utils import random_pick, calculate_class_weights


def check_characters(char_list, s):
    return any(char in s for char in char_list)


def prepare_samples(df):
    """
    Convert a pandas DataFrame with 'sequence' and 'label' columns into a list of tuples.

    Parameters:
    df (pandas.DataFrame): DataFrame with at least two columns 'sequence' and 'label'.

    Returns:
    list: A list of tuples, where each tuple is (sequence, label).
    """
    return list(zip(df['sequence'], df['label']))

def count_samples_by_class(samples):
    """Count the number of samples for each class."""
    class_counts = {}

    # Iterate over the samples
    for _, label in samples:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    return class_counts


def independent_exponential_smoothing(weights_dict, alpha=0.5):
    """
    Apply independent exponential smoothing to the weights.
    Each weight is reduced by an exponential decay factor.
    alpha is the smoothing factor where 0 < alpha <= 1.
    """
    # Apply independent exponential smoothing to each weight
    smoothed_weights = {k: v ** alpha for k, v in weights_dict.items()}

    return smoothed_weights


def prepare_fold_samples(dataset_path, task_token, max_length, max_samples, random_seed, logging):
    df = pd.read_csv(dataset_path)
    samples = prepare_samples(df)

    class_weights = calculate_class_weights(count_samples_by_class(samples))
    class_weights = independent_exponential_smoothing(class_weights)

    new_samples = []
    for sequence, label in samples:
        new_samples.append(
            (task_token, sequence[:max_length-2], [label+1], 'null', np.full(2, class_weights[label]))
        )

    new_samples = random_pick(new_samples, max_samples, random_seed)
    logging.info(f'{task_token}: remaining samples: {len(new_samples)}')

    train_label_index_mapping = {f'{key}': i for i, key in enumerate(sorted(set(token for sample in new_samples for token in sample[2])))}

    return new_samples, train_label_index_mapping
