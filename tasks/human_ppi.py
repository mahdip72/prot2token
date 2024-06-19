import numpy as np
import pandas as pd
from utils.utils import random_pick, calculate_class_weights


def prepare_samples(df):
    """
    Convert a pandas DataFrame with 'sequence_1', 'sequence_2', and 'label' columns into a list of tuples,
    replacing '0' with 'connect' and '1' with 'disconnect' in the 'label' column.

    Parameters:
    df (pandas.DataFrame): DataFrame with at least two columns 'sequence_1', 'sequence_2', and 'label'.

    Returns:
    list: A list of tuples, where each tuple is (sequence_1, sequence_2, label).
    """
    # Replace '0' with 'connect' and '1' with 'disconnect' in the 'label' column
    df['label'] = df['label'].replace({0: 'disconnect', 1: 'connect'})

    return list(zip(df['sequence_1'], df['sequence_2'], df['label']))


def count_samples_by_class(samples):
    """Count the number of samples for each class."""
    class_counts = {}

    # Iterate over the samples
    for _, _, label in samples:
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


def prepare_human_ppi_samples(dataset_path, task_token, max_length, max_samples, random_seed, logging):
    df = pd.read_csv(dataset_path)
    samples = prepare_samples(df)

    train_label_index_mapping = {'disconnect': 0, 'connect': 1}

    class_weights = calculate_class_weights(count_samples_by_class(samples))
    class_weights = independent_exponential_smoothing(class_weights)

    new_samples = []
    for sequence_1, sequence_2, label in samples:
        while len(sequence_1) + len(sequence_2) > max_length - 3:
            sequence_1 = sequence_1[:int(len(sequence_1)/1.1)]
            sequence_2 = sequence_2[:int(len(sequence_2)/1.1)]
        new_samples.append(
            (task_token, sequence_1 + "." + sequence_2, [label], 'null', np.full(2, class_weights[label]))
        )

    new_samples = random_pick(new_samples, max_samples, random_seed)
    logging.info(f'{task_token}: remaining samples: {len(new_samples)}')

    return new_samples, train_label_index_mapping
