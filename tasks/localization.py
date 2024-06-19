import os
import random
import numpy as np
from utils.utils import Tokenizer, random_pick, calculate_class_weights, k_batches, merge_lists


def count_samples_by_class(samples):
    """Count the number of samples for each class."""
    class_counts = {}

    # Iterate over the samples
    for sample in samples:
        label = sample[2][0]
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


def add_sample_weights(train_samples: list, valid_samples: list) -> (list, list):
    class_weights = count_samples_by_class(train_samples)
    class_weights = calculate_class_weights(class_weights)
    # class_weights = independent_exponential_smoothing(class_weights)

    new_train_samples = []
    for sample in train_samples:
        new_train_samples.append((sample[0], sample[1], [sample[2][0], sample[2][1]], sample[3],
                                  np.full((len(sample[2]) + 1), class_weights[sample[2][0]])))

    new_valid_samples = []
    for sample in valid_samples:
        new_valid_samples.append((sample[0], sample[1], [sample[2][0], sample[2][1]], sample[3],
                                  np.full((len(sample[2]) + 1), 1)))

    return new_train_samples, new_valid_samples


def check_characters(char_list, s):
    return any(char in s for char in char_list)


def prepare_localization_samples(data_path, task_token, max_length, max_samples, random_seed, logging):
    annotations_file = os.path.join(data_path, "localization/data/swissprot_annotated_proteins.tab")
    seq_file = os.path.join(data_path, "localization/data/idmapping_2023_08_25.tsv")

    f_anno = open(annotations_file)
    id_labels = f_anno.readlines()
    f_anno.close()
    f_seq = open(seq_file)
    seq = f_seq.readlines()
    f_seq.close()

    print('preprocess data')
    samples = []
    counter = 0
    for idx in range(len(seq)):
        sequence = seq[idx].split("\t")[2]
        label = id_labels[idx].split("\t")[1].lower()
        position = id_labels[idx].split("\t")[2].rstrip()
        # if len(sequence) > max_length - 2:
        #     counter += 1
        #     continue
        # if check_characters(['U', 'X', 'Z', 'B'], sequence):
        #     counter += 1
        #     continue
        samples.append((task_token, sequence, [label, int(position)], 'null'))

    logging.info(f'{task_token}: number of removed sequences bigger than {max_length - 2}: {counter}')

    train_samples, valid_samples = prepare_localization_train_valid_samples(samples, random_seed)
    train_samples = random_pick(train_samples, max_samples, random_seed)

    train_samples, valid_samples = add_sample_weights(train_samples, valid_samples)

    logging.info(f'{task_token}: remaining train samples: {len(train_samples)}')
    logging.info(f'{task_token}: remaining valid samples: {len(valid_samples)}')

    return train_samples, valid_samples



def prepare_localization_train_valid_samples(samples, random_seed):
    random.seed(random_seed)

    # Shuffle the list
    random.shuffle(samples)

    samples_batches = k_batches(samples, 5)

    valid_samples = samples_batches.pop(0)
    train_samples = merge_lists(samples_batches)

    return train_samples, valid_samples