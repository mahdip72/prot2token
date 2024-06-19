import numpy as np
import pandas as pd
import csv
import itertools
from utils.utils import random_pick, calculate_class_weights


def load_ec_seq_annot(file_seq, file_annot):
    # Load EC annotations """
    # Example:
    # file_seq: EC_valid.csv
    # file_annot: EC_annot.csv
    prot2seq = {}
    prot2annot = {}

    # gain the annotation
    with open(file_annot, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        # molecular function
        next(reader, None)  # skip the headers
        ec_numbers = {'ec': next(reader)}
        next(reader, None)  # skip the headers
        counts = {'ec': np.zeros(len(ec_numbers['ec']), dtype=float)}
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            ec_indices = [ec_numbers['ec'].index(ec_num) for ec_num in prot_ec_numbers.split(',')]
            prot2annot[prot] = {'ec': np.zeros(len(ec_numbers['ec']), dtype=float)}
            prot2annot[prot]['ec'][ec_indices] = 1.0
            counts['ec'][ec_indices] += 1

    with open(file_seq, mode="r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # molecular function
        next(reader, None)
        for row in reader:
            prot, prot_seq = row[0], row[1]
            prot2seq[prot] = {'seq': prot_seq}

    return prot2seq, prot2annot, ec_numbers, counts


def prepare_samples(f_seq, f_annot):  # attention QShao Dec-7-2023
    """
    Revised by QShao Dec-7-2023.
    f_seq: EC_test.csv
    f_annot: nrPDB-EC_annot.tsv
    read seq from f_seq and label from f_annot and merge them
    into a pandas dataframe df
    Returns:
    list: A list of tuples, where each tuple is (sequence, label).
    """
    prot2seq, prot2annot, ec_numbers, counts = load_ec_seq_annot(f_seq, f_annot)

    samples = []
    max_length = 0
    for item in prot2seq:
        ont_indices = np.where(prot2annot[item]['ec'] == 1)[0]
        labels = [ec_numbers['ec'][index] for index in ont_indices]
        # print(sorted(labels))
        if len(labels) > max_length:
            max_length = len(labels)
        samples.append((prot2seq[item]['seq'], labels))
    # print(f'max_length: {max_length}')
    return samples


def calculate_label_frequencies(data):
    """
    Calculate the frequency of each label in the dataset.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing a 'Label' column with comma-separated labels.

    Returns:
    dict: A dictionary with labels as keys and their frequencies as values.
    """
    label_freq = {}

    # Iterate through each label entry in the dataset
    for labels in data['Label']:
        # Split the labels by comma
        split_labels = labels.split(',')
        # Update the frequency of each label in the dictionary
        for label in split_labels:
            if label in label_freq:
                label_freq[label] += 1
            else:
                label_freq[label] = 1

    return label_freq


def independent_exponential_smoothing(weights_dict, alpha=0.5):
    """
    Apply independent exponential smoothing to the weights.
    Each weight is reduced by an exponential decay factor.
    alpha is the smoothing factor where 0 < alpha <= 1.
    """
    # Apply independent exponential smoothing to each weight
    smoothed_weights = {k: v ** alpha for k, v in weights_dict.items()}

    return smoothed_weights


def extract_unique_terms(samples):
    unique_terms = set()
    for item in samples:
        labels = item[2]
        unique_terms.update(labels)

    terms_to_index = {term: idx for idx, term in enumerate(unique_terms)}
    numbers_to_index = {str(idx + 1): idx for idx, term in enumerate(unique_terms)}
    return terms_to_index, numbers_to_index


# def remove_terms_from_samples(samples, terms_to_remove):
#     updated_samples = []
#     for item in samples:
#         task, sequence, label, prot_id, sample_weight = item
#         # Filter out the terms to be removed
#         filtered_labels = [term for term in label if term not in terms_to_remove]
#         updated_samples.append((task, sequence, filtered_labels, prot_id, sample_weight))
#     return updated_samples


# def map_terms_to_numbers(samples, label_index):
#     updated_samples = []
#     for item in samples:
#         task, sequence, labels, prot_id, sample_weight = item
#         labels = [int(label_index[label] + 1) for label in labels]
#         labels = sorted(labels)
#         labels = [str(num) for num in labels]
#         updated_samples.append((task, sequence, labels, prot_id, sample_weight))
#     return updated_samples


def convert_labels_to_string(labels):
    # Flatten the list by splitting each label and including each part as a separate element
    return [element for label in labels for part in label.split('.') for element in (part.split('-') if part != '-' else [part])]


def prepare_enzyme_commission_samples(dataset_path, label_path, task_token, max_length, max_samples, random_seed, logging):
    samples = prepare_samples(dataset_path, label_path)

    # Create a sorted list of unique labels using a single line of code
    unique_labels = sorted(set(label for labels in pd.read_csv(dataset_path)['Label'] for label in labels.split(',')))

    train_label_index_mapping = {label: index for index, label in enumerate(unique_labels)}

    broken_labels = [i.split('.') for i in unique_labels]
    broken_labels = set(list(itertools.chain.from_iterable(broken_labels)))
    broken_label_index_mapping = {label: index for index, label in enumerate(broken_labels)}

    class_weights = calculate_class_weights(calculate_label_frequencies(pd.read_csv(dataset_path)))
    class_weights = independent_exponential_smoothing(class_weights)

    new_samples = []
    for sequence, labels in samples:
        sample_weight = [1]
        if len(labels) > 0:
            sample_weight = [class_weights[label] for label in labels]
            sample_weight = [item for item in sample_weight for _ in range(4)]

        new_samples.append(
            (task_token, sequence[:max_length - 2], convert_labels_to_string(labels), 'null', sample_weight))

    new_samples = random_pick(new_samples, max_samples, random_seed)
    logging.info(f'{task_token}: remaining samples: {len(new_samples)}')

    return new_samples, train_label_index_mapping, broken_label_index_mapping
