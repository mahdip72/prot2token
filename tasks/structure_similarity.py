import numpy as np
import pandas as pd
import os
from utils.utils import random_pick, calculate_class_weights


def create_sequence_similarity_list(samples, indices, targets):
    combined_list = []
    for idx_pair, target in zip(indices, targets):
        # print('index', idx_pair)
        seq1 = samples[idx_pair[0]]
        seq2 = samples[idx_pair[1]]
        combined_list.append([seq1, seq2, target])
    return combined_list


def prepare_samples(mode, dataset_path):
    from proteinshake.tasks import StructureSimilarityTask
    task = StructureSimilarityTask(root=os.path.join(dataset_path, 'structure_similarity'))
    proteins = task.proteins

    samples = []

    for protein_dict in proteins:
        samples.append([protein_dict['protein']['sequence']])

    a = StructureSimilarityTask(root=os.path.join(dataset_path, 'structure_similarity'),
                                split='structure', split_similarity_threshold=0.7)

    if mode == 'train':
        samples = create_sequence_similarity_list(samples, a.train_index, a.train_targets)
    elif mode == 'valid':
        samples = create_sequence_similarity_list(samples, a.val_index, a.val_targets)
    elif mode == 'test':
        samples = create_sequence_similarity_list(samples, a.test_index, a.test_targets)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    return samples


def independent_exponential_smoothing(weights_dict, alpha=0.5):
    """
    Apply independent exponential smoothing to the weights.
    Each weight is reduced by an exponential decay factor.
    alpha is the smoothing factor where 0 < alpha <= 1.
    """
    # Apply independent exponential smoothing to each weight
    smoothed_weights = {k: v ** alpha for k, v in weights_dict.items()}

    return smoothed_weights


def prepare_structure_similarity_samples(dataset_path, task_token, max_length, max_samples, random_seed, logging, mode):
    samples = prepare_samples(mode, dataset_path)

    new_samples = []
    for sequence_1, sequence_2, label in samples:
        sequence_1, sequence_2 = sequence_1[0], sequence_2[0]
        label = round(label, 2)
        while len(sequence_1) + len(sequence_2) > max_length - 3:
            sequence_1 = sequence_1[:int(len(sequence_1)/1.1)]
            sequence_2 = sequence_2[:int(len(sequence_2)/1.1)]

        new_samples.append(
            (task_token, [[sequence_1, sequence_2]], list(str(label)), 'null', np.full(2, 1))
        )
    new_samples = random_pick(new_samples, max_samples, random_seed)
    logging.info(f'{task_token}: remaining samples: {len(new_samples)}')

    train_labels = list(range(10))
    train_labels = [str(i) for i in train_labels] + ['.']

    return new_samples, train_labels
