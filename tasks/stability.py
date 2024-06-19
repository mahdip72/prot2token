import numpy as np
import pandas as pd
from utils.utils import random_pick, calculate_class_weights, independent_exponential_smoothing


def map_labels(original_label, original_min=-1.97, original_max=3.4, new_min=0.0001, new_max=0.9999):
    """
    Maps an original label value from its original range to a new specified range linearly.

    Parameters:
    original_label (float): The original label value to be mapped.
    original_min (float, optional): The minimum value of the original range. Defaults to 1.283.
    original_max (float, optional): The maximum value of the original range. Defaults to 4.123.
    new_min (float, optional): The minimum value of the new range.
    new_max (float, optional): The maximum value of the new range.

    Returns:
    float: The new label value mapped to the new range.
    """
    # Linearly map the original label to the new range
    return ((original_label - original_min) / (original_max - original_min)) * (new_max - new_min) + new_min


def prepare_samples(df):
    """
    Convert a pandas DataFrame with 'sequence' and 'label' columns into a list of tuples.
    The 'label' column values are rounded to two decimal places.

    Parameters:
    df (pandas.DataFrame): DataFrame with at least two columns 'sequence' and 'label'.

    Returns:
    list: A list of tuples, where each tuple is (sequence, label).
    """
    # Round the 'label' column values to two decimal places
    df['label'] = df['label'].round(4)
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


def prepare_stability_samples(dataset_path, task_token, max_length, max_samples, random_seed, logging):
    df = pd.read_csv(dataset_path)
    samples = prepare_samples(df)

    # class_weights = calculate_class_weights(count_samples_by_class(samples))
    # class_weights = independent_exponential_smoothing(class_weights)

    new_samples = []
    for sequence, label in samples:
        label = round(map_labels(label), 4)
        new_samples.append(
            (task_token, sequence[:max_length-2], list(str(label)), 'null', np.full(2, 1))
        )
    new_samples = random_pick(new_samples, max_samples, random_seed)
    logging.info(f'{task_token}: remaining samples: {len(new_samples)}')

    train_labels = list(range(10))
    train_labels = [str(i) for i in train_labels] + ['.']
    # train_labels = ['.', '-']

    return new_samples, train_labels
