import numpy as np
import pandas as pd
from utils.utils import random_pick, calculate_class_weights


def separate_train_validation(dataset_path, partition_number=4):
    """
    Separates a protein dataset into training and validation sets based on partition numbers.

    This function reads a dataset of protein sequences and their cellular location labels,
    then separates the data into training and validation sets. The training set comprises
    samples from partitions 0 to 3, and the validation set comprises samples from partition 4.
    Each set is a list of tuples, where each tuple contains a protein sequence and a list of its
    corresponding binary labels for various cellular locations.

    Parameters:
    dataset_path (str): Path to the CSV file containing the dataset.

    Returns:
    tuple: A tuple containing two lists:
           - The first list is the training set, with each element being a tuple of
             (protein sequence, list of labels).
           - The second list is the validation set, structured in the same way.
    """

    # Load the dataset
    data = pd.read_csv(dataset_path)

    # Split the dataset into training and validation based on the 'Partition' column
    train_data = data[data['Partition'] != partition_number]
    validation_data = data[data['Partition'] == partition_number]

    # Function to extract labels for a given row
    def extract_labels(row):
        label_columns = ['Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane',
                         'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole',
                         'Golgi apparatus', 'Peroxisome']
        return [int(row[col]) for col in label_columns]

    # Create lists for training and validation sets
    train_set = [(row['Sequence'], extract_labels(row)) for _, row in train_data.iterrows()]
    validation_set = [(row['Sequence'], extract_labels(row)) for _, row in validation_data.iterrows()]

    return train_set, validation_set


def calculate_label_frequencies(data, partition_number=4):
    # Columns indicating labels
    label_columns = ['Cytoplasm', 'Nucleus', 'Extracellular',
                     'Cell membrane', 'Mitochondrion', 'Plastid',
                     'Endoplasmic reticulum', 'Lysosome/Vacuole',
                     'Golgi apparatus', 'Peroxisome']

    data = data[data['Partition'] != partition_number]

    # Count the frequencies of each label
    label_frequencies = data[label_columns].sum()

    # Convert to dictionary
    frequencies_dict = label_frequencies.to_dict()

    return frequencies_dict


def load_test_dataset(dataset_path):
    """
    Load the test set with complete label columns.

    This function reads a dataset of protein sequences and their cellular location labels,
    then loads the data into a test set. It ensures that all expected label columns are present,
    filling missing columns with zeros. Each set is a list of tuples, where each tuple contains
    a protein sequence and a list of its corresponding binary labels for various cellular locations.

    Parameters:
    dataset_path (str): Path to the CSV file containing the dataset.

    Returns:
    list: The test set, with each element being a tuple of (protein sequence, list of labels).
    """

    try:
        # Load the dataset
        data = pd.read_csv(dataset_path)

        # Define all possible label columns
        all_label_columns = ['Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane',
                             'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole',
                             'Golgi apparatus', 'Peroxisome']

        # Fill missing label columns with zeros
        for col in all_label_columns:
            if col not in data.columns:
                data[col] = 0

        # Create the test set
        test_set = [(row['fasta'], row[all_label_columns].tolist()) for _, row in data.iterrows()]

        return test_set
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


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


def prepare_localization_deeploc_samples(dataset_path, task_token, max_length, max_samples, random_seed, logging,
                                         mode='train'):
    class_weights = {'Cytoplasm': 1, 'Nucleus': 1, 'Extracellular': 1, 'Cell membrane': 1,
                     'Mitochondrion': 1, 'Plastid': 1, 'Endoplasmic reticulum': 1,
                     'Lysosome/Vacuole': 1, 'Golgi apparatus': 1, 'Peroxisome': 1}
    if mode == 'train':
        samples, _ = separate_train_validation(dataset_path, partition_number=4)
        label_frequencies_dict = calculate_label_frequencies(pd.read_csv(dataset_path),
                                                             partition_number=4)
        class_weights = calculate_class_weights(label_frequencies_dict)
    elif mode == 'valid':
        _, samples = separate_train_validation(dataset_path, partition_number=4)
    elif mode == 'test':
        samples = load_test_dataset(dataset_path)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    label_index_mapping = {'Cytoplasm': 0, 'Nucleus': 1, 'Extracellular': 2, 'Cell membrane': 3,
                           'Mitochondrion': 4, 'Plastid': 5, 'Endoplasmic reticulum': 6,
                           'Lysosome/Vacuole': 7, 'Golgi apparatus': 8, 'Peroxisome': 9}
    index_label_mapping = {v: k for k, v in label_index_mapping.items()}
    new_samples = []
    for sequence, labels in samples:
        new_samples.append(
            (task_token, sequence[:max_length - 2],
             [index_label_mapping[i] for i, label in enumerate(labels) if label != 0],
             'null', np.array([class_weights[index_label_mapping[i]] for i, label in enumerate(labels) if label != 0]))
        )

    new_samples = random_pick(new_samples, max_samples, random_seed)
    logging.info(f'{task_token}: remaining samples: {len(new_samples)}')

    return new_samples, label_index_mapping
