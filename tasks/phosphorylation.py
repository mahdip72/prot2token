import numpy as np
import pandas as pd
from utils.utils import random_pick, calculate_class_weights


def convert_ptm_positions(ptm_array):
    """
    Convert the PTM positions represented as 1 in the array to their corresponding
    numerical positions in the protein sequence.

    :param ptm_array: A list or array indicating PTM positions with 1.
    :return: A list of positions (1-indexed) where PTMs occur.
    """
    # Find the indices where the value is 1 in the ptm_array
    ptm_positions = [index + 1 for index, value in enumerate(ptm_array) if value == 1]
    return ptm_positions


def convert_samples(samples):
    new_list = []
    for sequence, ptm_array in samples:
        positive_positions = convert_ptm_positions(ptm_array)
        # print([sequence[i-1] for i in positive_positions])
        new_list.append((sequence, positive_positions))
    return new_list


def prepare_samples(df):
    """
    Convert a pandas DataFrame with 'sequence' and 'label' columns into a list of tuples.

    Parameters:
    df (pandas.DataFrame): DataFrame with at least two columns 'sequence' and 'label'.

    Returns:
    list: A list of tuples, where each tuple is (sequence, label).
    """
    return convert_samples(list(zip(df['x'], df['label'])))


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


def find_indexes(string, chars, exclude_indexes):
    """
    Returns the 1-indexed positions of specified characters in a string, excluding certain positions.

    Parameters:
    - string (str): The input string in which to search for characters.
    - chars (list of str): A list of characters to search for in the string.
    - exclude_indexes (list of int): A list of 1-indexed positions to exclude from the search.

    Returns:
    - list of int: A list of 1-indexed positions where the characters from 'chars' appear in 'string',
                   excluding positions from 'exclude_indexes'.

    """
    # Convert the 1-indexed positions to 0-indexed positions for Python
    exclude_indexes = [i - 1 for i in exclude_indexes]

    indexes = []
    for i, char in enumerate(string):
        # Check if the character is in the chars list and not in the exclude list
        if char in chars and i not in exclude_indexes:
            # Convert the 0-indexed position back to 1-indexed for the result
            indexes.append(i + 1)
    return indexes


def check_ptm_site(sequence, positions, allowed_ptm_sites):
    for i in positions.copy():
        if i > len(sequence):
            positions.remove(i)
            continue
        elif not sequence[i - 1] in allowed_ptm_sites:
            positions.remove(i)
    return positions


def prepare_phosphorylation_samples(dataset_path, positive_amino_acids, task_token, max_length, max_samples, random_seed, logging):
    samples_array = np.load(dataset_path, allow_pickle=True)

    samples = prepare_samples(samples_array)

    new_samples = []
    for sequence, positions in samples:
        if len(sequence) > max_length - 2:
            continue

        positive_positions = check_ptm_site(sequence, positions, positive_amino_acids)
        negative_positions = find_indexes(sequence, positive_amino_acids, positions)
        all_positions = sorted(negative_positions + positive_positions) + ['<sep>'] + positive_positions

        sample_weights = np.ones(len(all_positions))
        if not len(positive_positions) == 0:
            positive_negative_ratio = len(negative_positions) / len(positive_positions)
            sample_weights[len(negative_positions + positive_positions):] = positive_negative_ratio
            sample_weights = sample_weights ** 0.5  # Apply exponential smoothing

        new_samples.append(
            (task_token, sequence, all_positions, 'null', sample_weights)
        )

    new_samples = random_pick(new_samples, max_samples, random_seed)
    logging.info(f'{task_token}: remaining samples: {len(new_samples)}')

    return new_samples
