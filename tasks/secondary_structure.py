import numpy as np
import pandas as pd
from utils.utils import random_pick, calculate_class_weights


def process_sequences(file_path):
    """
    Processes a CSV file containing concatenated protein sequences and their labels.

    This function reads a CSV file where each row in the 'sequence label' column
    contains a protein sequence followed by its corresponding label. The function
    separates each sequence from its label, ensuring that the lengths of the sequence
    and the label are identical. It returns a list of tuples, each containing a sequence
    and its matching label.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    list of tuples: A list where each tuple contains a sequence and its corresponding label.
    """

    data = pd.read_csv(file_path)
    sequence_label_pairs = []
    mapping = {'0': 'alpha_helix', '1': 'beta_sheet', '2': 'coil_or_loop'}
    for row in data["sequence label"]:
        # Separating alphabetic characters (sequence) from numeric characters (label)
        sequence = ''.join(filter(str.isalpha, row))
        label = ''.join(filter(str.isdigit, row))

        # Ensure the sequence and label lengths are identical
        if len(sequence) == len(label):
            label = [mapping[element] for element in label]
            sequence_label_pairs.append((sequence, label))
        else:
            # If lengths are not identical, print the sequence and label for inspection
            print(f"Sequence: {sequence}, Label: {label} (Lengths not equal)")

    return sequence_label_pairs


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


def prepare_secondary_structure_samples(dataset_path, task_token, max_length, max_samples, random_seed, logging):
    samples = process_sequences(dataset_path)

    # class_weights = calculate_class_weights(count_samples_by_class(samples))
    # class_weights = independent_exponential_smoothing(class_weights)

    new_samples = []
    for sequence, label in samples:
        samples_weights = np.full(len(label[:max_length-2])+2, 1)
        # samples_weights[-1] = 1
        new_samples.append(
            (task_token, sequence[:max_length-2], label[:max_length-2], 'null', samples_weights)
        )

    new_samples = random_pick(new_samples, max_samples, random_seed)
    logging.info(f'{task_token}: remaining samples: {len(new_samples)}')

    train_label_index_mapping = {'alpha_helix': 0, 'beta_sheet': 1, 'coil_or_loop': 2}

    return new_samples, train_label_index_mapping
