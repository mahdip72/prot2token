import numpy as np
import pandas as pd
import csv
import os
from utils.utils import random_pick, calculate_class_weights


def load_go_seq_annot(file_seq, file_annot):
    # Load GO annotations """
    # Example:
    # file_seq: GO_valid.csv
    # file_annot: nrPDB-GO_annot.tsv
    # mf: 489, bp: 1943, cc: 320
    prot2seq = {}
    with open(file_seq, mode="r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # molecular function
        next(reader, None)
        for row in reader:
            prot, prot_seq = row[0], row[1]
            prot2seq[prot] = {'seq': prot_seq}

    onts = ['mf', 'bp', 'cc']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}

    with open(file_annot, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        counts = {ont: np.zeros(len(goterms[ont]), dtype=float) for ont in onts}
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if
                                  goterm != '']
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
                counts[onts[i]][goterm_indices] += 1.0

    return prot2seq, prot2annot, goterms, gonames, counts


def prepare_samples(f_seq, f_annot):
    prot2seq, prot2annot, goterms, gonames, counts = load_go_seq_annot(f_seq, f_annot)

    samples = []
    for item in prot2seq:
        tasks_terms = {}
        for ont in ['bp', 'mf', 'cc']:
            ont_indices = np.where(prot2annot[item][ont] == 1)[0]
            ont_terms = [goterms[ont][index] for index in ont_indices]
            tasks_terms[ont] = ont_terms
        samples.append((prot2seq[item]['seq'], tasks_terms))

    return samples


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


def extract_unique_terms(samples):
    unique_terms = set()
    for item in samples:
        labels = item[2]
        unique_terms.update(labels)

    terms_to_index = {term: idx for idx, term in enumerate(unique_terms)}
    numbers_to_index = {str(idx+1): idx for idx, term in enumerate(unique_terms)}
    return terms_to_index, numbers_to_index


def remove_terms_from_samples(samples, terms_to_remove):
    updated_samples = []
    for item in samples:
        task, sequence, label, prot_id, sample_weight = item
        # Filter out the terms to be removed
        filtered_labels = [term for term in label if term not in terms_to_remove]
        updated_samples.append((task, sequence, filtered_labels, prot_id, sample_weight))
    return updated_samples


def map_terms_to_numbers(samples, label_index):
    updated_samples = []
    for item in samples:
        task, sequence, labels, prot_id, sample_weight = item
        labels = [int(label_index[label]+1) for label in labels]
        labels = sorted(labels)
        labels = [str(num) for num in labels]
        updated_samples.append((task, sequence, labels, prot_id, sample_weight))
    return updated_samples


def calculate_gene_ontology_label_frequencies(samples):
    """
    Calculate the frequency of each label in the dataset.

    Parameters:
    data (pd.DataFrame): A list containing samples (tuples).

    Returns:
    dict: three dictionary with labels as keys and their frequencies as values.
    """
    label_freq_bp = {}
    label_freq_mf = {}
    label_freq_cc = {}

    # Iterate through each label entry in the dataset
    for _, labels in samples:
        for label in labels['bp']:
            if label in label_freq_bp:
                label_freq_bp[label] += 1
            else:
                label_freq_bp[label] = 1
        for label in labels['mf']:
            if label in label_freq_mf:
                label_freq_mf[label] += 1
            else:
                label_freq_mf[label] = 1
        for label in labels['cc']:
            if label in label_freq_cc:
                label_freq_cc[label] += 1
            else:
                label_freq_cc[label] = 1

    return label_freq_bp, label_freq_mf, label_freq_cc


def prepare_gene_ontology_samples(dataset_path, label_path, max_length, max_samples, random_seed, logging, configs,
                                  task_type=('bp', 'mf', 'cc')):
    samples = prepare_samples(dataset_path, label_path)

    class_weights_bp, class_weights_mf, class_weights_cc = calculate_gene_ontology_label_frequencies(samples)

    class_weights_bp = calculate_class_weights(class_weights_bp)
    class_weights_mf = calculate_class_weights(class_weights_mf)
    class_weights_cc = calculate_class_weights(class_weights_cc)

    class_weights_bp = independent_exponential_smoothing(class_weights_bp)
    class_weights_mf = independent_exponential_smoothing(class_weights_mf)
    class_weights_cc = independent_exponential_smoothing(class_weights_cc)

    train_label_index = {task: {} for task in ['bp', 'mf', 'cc']}

    new_samples_bp = []
    new_samples_mf = []
    new_samples_cc = []
    for sequence, labels in samples:
        if 'bp' in task_type:
            if len(labels['bp']) < configs.prot2token_model.decoder.max_len:
                sample_weight = [1]
                if len(labels['bp']) > 0:
                    sample_weight = [class_weights_bp[label] for label in labels['bp']]
                new_samples_bp.append(('<task_gene_ontology_bp>', sequence[:max_length-2], sorted(labels['bp']), 'null', sample_weight))
        if 'mf' in task_type:
            if len(labels['mf']) < configs.prot2token_model.decoder.max_len:
                sample_weight = [1]
                if len(labels['mf']) > 0:
                    sample_weight = [class_weights_mf[label] for label in labels['mf']]
                new_samples_mf.append(('<task_gene_ontology_mf>', sequence[:max_length-2], sorted(labels['mf']), 'null', sample_weight))
        if 'cc' in task_type:
            if len(labels['cc']) < configs.prot2token_model.decoder.max_len:
                sample_weight = [1]
                if len(labels['cc']) > 0:
                    sample_weight = [class_weights_cc[label] for label in labels['cc']]
                new_samples_cc.append(('<task_gene_ontology_cc>', sequence[:max_length-2], sorted(labels['cc']), 'null', sample_weight))

    if 'bp' in task_type:
        new_samples_bp = remove_terms_from_samples(new_samples_bp, ['GO:0034286', 'GO:0034289', 'GO:0034288'])
    if 'mf' in task_type:
        new_samples_mf = remove_terms_from_samples(new_samples_mf, ['GO:0048513'])

    if 'bp' in task_type:
        train_term_bp_index, train_label_bp_index = extract_unique_terms(new_samples_bp)
        new_samples_bp = map_terms_to_numbers(new_samples_bp, train_term_bp_index)
        new_samples_bp = random_pick(new_samples_bp, max_samples, random_seed)
        logging.info(f'<task_gene_ontology_bp> remaining samples: {len(new_samples_bp)}')
        train_label_index['bp'] = train_label_bp_index

    if 'mf' in task_type:
        train_term_mf_index, train_label_mf_index = extract_unique_terms(new_samples_mf)
        new_samples_mf = map_terms_to_numbers(new_samples_mf, train_term_mf_index)
        new_samples_mf = random_pick(new_samples_mf, max_samples, random_seed)
        logging.info(f'<task_gene_ontology_mf> remaining samples: {len(new_samples_mf)}')
        train_label_index['mf'] = train_label_mf_index

    if 'cc' in task_type:
        train_term_cc_index, train_label_cc_index = extract_unique_terms(new_samples_cc)
        new_samples_cc = map_terms_to_numbers(new_samples_cc, train_term_cc_index)
        new_samples_cc = random_pick(new_samples_cc, max_samples, random_seed)
        logging.info(f'<task_gene_ontology_cc> remaining samples: {len(new_samples_cc)}')
        train_label_index['cc'] = train_label_cc_index

    new_samples = new_samples_bp + new_samples_mf + new_samples_cc
    return new_samples, train_label_index
