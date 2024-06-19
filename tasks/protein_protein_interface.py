import numpy as np
import os
from itertools import chain
from utils.utils import random_pick, load_list_from_file
from proteinshake.tasks import ProteinProteinInterfaceTask


def tuple_list_to_string_list_with_commas(tuple_list):
    """
    Convert a list of tuples into a flattened list of strings with commas
    separating the elements of different tuples.

    Args:
    tuple_list (list of tuples): A list containing tuples of integers or
    floats.

    Returns:
    list: A list of strings, where each number in the tuples is converted to
    a string and tuples are separated by a comma.
    """
    # Using list comprehension and flattening with a comma after each tuple
    flattened_list = []
    for tup in tuple_list:
        flattened_list.extend([str(item) for item in tup] + [','])
    return flattened_list[:-1]  # Remove the last comma


def prepare_target(target):
    # Find the indices of elements that are 1
    rows, cols = np.where(target == 1)

    # Adjust indices to start from 1 and combine them
    adjusted_sorted_indices = sorted([(r + 1, c + 1) for r, c in zip(rows, cols)])
    new_target = tuple_list_to_string_list_with_commas(adjusted_sorted_indices)
    first_part_of_label = [str(target.shape[0]), str(target.shape[1]), '<sep>']
    first_part_of_label += new_target
    return new_target


def create_pairs_list(samples, indices, targets, max_length):
    combined_list = []
    counter = 0
    for idx_pair, target in zip(indices, targets):
        seq1 = samples[idx_pair[0]]
        seq2 = samples[idx_pair[1]]

        # remove samples that longer than max_length
        # if len(seq1[0]) + len(seq2[0]) > max_length - 3:
        #     counter += 1
            # print(len(seq1[0]) + len(seq2[0]))
            # continue

        if len(seq1[0]) + len(seq2[0]) > max_length:
            max_length = len(seq1[0]) + len(seq2[0])
        target = prepare_target(target)
        combined_list.append([seq1[0], seq2[0], target])
    print('removed samples:', counter)
    return combined_list


def prepare_samples(mode, dataset_path, max_length):
    task = ProteinProteinInterfaceTask()
    proteins = task.proteins

    samples = []

    for protein_dict in proteins:
        samples.append([protein_dict['protein']['sequence']])

    a = ProteinProteinInterfaceTask(root=os.path.join("./", 'protein_protein_interface'),
                                    split='structure', split_similarity_threshold=0.7)

    if mode == 'train':
        samples = create_pairs_list(samples, a.train_index, a.train_targets, max_length)
    elif mode == 'valid':
        samples = create_pairs_list(samples, a.val_index, a.val_targets, max_length)
    elif mode == 'test':
        samples = create_pairs_list(samples, a.test_index, a.test_targets, max_length)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    return samples


def process_list(input_list):
    # Remove commas from the list
    filtered_list = [item for item in input_list if item != ',']

    # Find the index of '<sep>'
    if '<sep>' in filtered_list:
        sep_index = filtered_list.index('<sep>')
        # Move the segment from '<sep>' onwards to the beginning, ensuring the correct order
        segment_to_move = filtered_list[sep_index + 1:]  # Move only the part after '<sep>'
        rest_of_list = filtered_list[:sep_index]  # The rest of the list before '<sep>'
        return segment_to_move + [filtered_list[sep_index]] + rest_of_list
    else:
        return filtered_list


def prepare_samples_from_pickle(dataset_path, max_length):
    samples = load_list_from_file(dataset_path)
    new_samples = []
    counter = 0
    for sample in samples:
        # remove samples that longer than max_length
        if len(sample[0]) + len(sample[1]) > max_length - 3:
            counter += 1
            # print(len(sample[0]) + len(sample[0]))
            continue
        sample[-1] = process_list(sample[-1])
        new_samples.append(sample)
    print('removed samples:', counter)
    return new_samples


def prepare_protein_protein_interface_samples(dataset_path, task_token, max_length, max_samples, random_seed, logging):
    # train set max length: 12520
    # valid set max length: 1470
    # test set max length: 1579

    # mode = 'train'
    # samples = prepare_samples(mode, dataset_path, max_length)
    samples = prepare_samples_from_pickle(dataset_path, max_length)

    train_label_list = list(set(chain.from_iterable(third_item for _, _, third_item in samples)))

    new_samples = []
    for sequence_1, sequence_2, label in samples:
        # label = [element for element in label if element != ","]
        new_samples.append(
            (task_token, [[sequence_1, sequence_2]], label, 'null', np.full(2, 1))
        )

    new_samples = random_pick(new_samples, max_samples, random_seed)
    logging.info(f'{task_token}: remaining samples: {len(new_samples)}')

    return new_samples, train_label_list
