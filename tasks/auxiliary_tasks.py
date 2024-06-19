import numpy as np
import pandas as pd
import os
import random
from utils.utils import random_pick, calculate_class_weights


def check_characters(char_list, s):
    return any(char in s for char in char_list)


def count_samples_by_class(samples):
    """Count the number of samples for each class."""
    class_counts = {}

    # Iterate over the samples
    for _, label in samples:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    return class_counts


def build_self_supervised_labels(sequence):
    """
    Find occurrences of each specified amino acid in the given sequence and
    return their positions in a dictionary, starting from 1 instead of 0.

    :param sequence: A string representing a sequence of amino acids.
    :return: A dictionary with amino acids as keys and lists of indices as values.
    """
    # amino_acids = 'STYKRN'  # List of amino acids to check
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # List of amino acids to check
    positions_dict = {acid: [] for acid in amino_acids}  # Initialize dictionary

    for i, acid in enumerate(sequence, start=1):  # Start counting from 1
        if acid in amino_acids:
            positions_dict[acid].append(i)

    return positions_dict


def mask_amino_acids(sequence, mask_percent=30) -> str:
    """
    Randomly replaces approximately mask_percent% of amino acids in the sequence with <mask>.

    :param sequence: The amino acid sequence as a string.
    :param mask_percent: The percentage of amino acids to be replaced.
    :return: The sequence with masked amino acids.
    """
    sequence_length = len(sequence)
    num_to_replace = int(sequence_length * mask_percent / 100)
    positions = random.sample(range(sequence_length), num_to_replace)

    masked_sequence = list(sequence)
    for pos in positions:
        masked_sequence[pos] = '<mask>'

    return ''.join(masked_sequence)


def create_pair_samples(grouped_samples, num_pairs_per_group, max_seq_length):
    new_pair_samples = []

    for task_name, samples in grouped_samples.items():
        # Filter samples by max_seq_length
        filtered_samples = [s for s in samples if len(s[1]) <= max_seq_length]

        # Check if there are enough samples in the filtered list to create pairs
        if len(filtered_samples) < 2:
            continue  # Skip to next group if not enough samples

        # Generate pairs
        for _ in range(min(num_pairs_per_group, len(filtered_samples)//2)):
            # Randomly select two distinct samples
            sample1, sample2 = random.sample(filtered_samples, 2)
            # Create a new pair sample using the sequence from sample1 and labels from sample2
            new_sample = (sample1[0], [[sample1[1], sample2[1]]], sample1[2] + ['<sep>'] + sample2[2],
                          sample1[3], np.full(2, 1))
            new_pair_samples.append(new_sample)

    return new_pair_samples


def prepare_auxiliary_samples(dataset_path, max_length, max_samples, random_seed, logging):
    df = pd.read_csv(dataset_path, usecols=['Amino Acid Sequence'])
    samples = df['Amino Acid Sequence'].tolist()

    # class_weights = calculate_class_weights(count_samples_by_class(samples))
    # class_weights = independent_exponential_smoothing(class_weights)

    new_samples = []
    for sequence in samples:
        auxiliary_tasks_dict = build_self_supervised_labels(sequence[:max_length-2])
        sequence = sequence[:max_length - 2]
        # sequence = mask_amino_acids(sequence, mask_percent=30)
        for amino_acid_name, labels in auxiliary_tasks_dict.items():
            if len(labels) == 0:
                continue

            new_samples.append(
                (f"<task_{amino_acid_name.lower()}>", sequence, labels, 'null', np.full(2, 1))
            )

    # Sort new_samples based on the length of each sequence
    new_samples = sorted(new_samples, key=lambda x: len(x[1]))

    new_samples = random_pick(new_samples, max_samples, random_seed)
    logging.info(f'auxiliary tasks: remaining samples: {len(new_samples)}')

    return new_samples
