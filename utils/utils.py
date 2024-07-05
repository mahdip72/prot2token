import os
from timm import optim
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import logging as log
from io import StringIO
from box import Box
import shutil
from pathlib import Path
import datetime
import random
from accelerate import Accelerator
from transformers import get_scheduler
import pickle


def remove_prefix_from_keys(dictionary, prefix):
    new_dict = {}
    for key, value in dictionary.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    return new_dict


def add_prefix_to_keys(dictionary, prefix):
    new_dict = {}
    for key, value in dictionary.items():
        new_key = prefix + key  # Concatenate the prefix and the original key
        new_dict[new_key] = value  # Store the value with the new key in the new dictionary
    return new_dict


def save_list_to_file(my_list, file_name):
    """
    Saves a list to a file.

    Parameters:
    my_list (list): The list to be saved.
    file_name (str): The name of the file where the list will be saved.
    """
    with open(file_name, 'wb') as file:
        pickle.dump(my_list, file)


def load_list_from_file(file_name):
    """
    Loads a list from a file.

    Parameters:
    file_name (str): The name of the file from which to load the list.

    Returns:
    list: The list loaded from the file.
    """
    with open(file_name, 'rb') as file:
        return pickle.load(file)


def independent_exponential_smoothing(weights_dict, alpha=0.5):
    """
    Apply independent exponential smoothing to the weights.
    Each weight is reduced by an exponential decay factor.
    alpha is the smoothing factor where 0 < alpha <= 1.
    """
    # Apply independent exponential smoothing to each weight
    smoothed_weights = {k: v ** alpha for k, v in weights_dict.items()}

    return smoothed_weights


def multiply_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply tensor A with tensor B element-wise for the shared columns.
    The rest of the columns in tensor A remain unchanged.

    Args:
        a (torch.Tensor): A tensor of shape (m, n).
        b (torch.Tensor): A tensor of shape (m, k) where k <= n.

    Returns:
        torch.Tensor: A tensor of shape (m, n) where the first k columns are the result
                      of multiplying A's first k columns with B, and the rest of the columns
                      are from A unchanged.

    Raises:
        RuntimeError: If the number of rows of A and B don't match.
    """
    multiplier = torch.ones_like(a)
    multiplier[:, :b.shape[1]] = b
    return a * multiplier


def calculate_class_weights(class_samples):
    total_samples = sum(class_samples.values())
    class_weights = {}

    # Calculate weights using the inverse of the class frequencies
    for class_name, samples in class_samples.items():
        class_weights[class_name] = total_samples / samples

    # Normalize the weights so that the largest class has a weight of 1
    min_weight = min(class_weights.values())
    for class_name, weight in class_weights.items():
        class_weights[class_name] = weight / min_weight

    return class_weights


def calculate_class_weights_normalized(samples_dict):
    """
    Calculate the weights for each class based on the number of samples.

    :param samples_dict: Dictionary containing classes as keys and number of samples as values.
    :return: Dictionary containing classes as keys and their respective weights as values.
    """
    min_samples = min(samples_dict.values())

    weights_dict = {}
    for key, value in samples_dict.items():
        weights_dict[key] = min_samples / value

    return weights_dict


def merge_lists(list_of_lists):
    merged_list = []
    for sublist in list_of_lists:
        for item in sublist:
            merged_list.append(item)
    return merged_list


def k_batches(data, k):
    """
    Split a list into k batches.

    Parameters:
    - data: List of items to be split.
    - k: Number of batches.

    Returns:
    - List of k batches.
    """
    if k <= 0:
        raise ValueError("k should be a positive integer.")

    batch_size = len(data) // k
    batches = []

    for i in range(k):
        start = i * batch_size
        if i == k - 1:
            # For the last batch, take all remaining items
            end = len(data)
        else:
            end = (i + 1) * batch_size
        batches.append(data[start:end])

    return batches


def load_configs(config, inference=False):
    """
        Load the configuration file and convert the necessary values to floats.

        Args:
            config (dict): The configuration dictionary.
            inference (bool): A boolean flag to indicate if the configuration is for inference.

        Returns:
            The updated configuration dictionary with float values.
        """

    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)

    if not inference:
        # Convert the necessary values to floats.
        tree_config.optimizer.lr = float(tree_config.optimizer.lr)
        tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
        tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
        tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    return tree_config


def prepare_saving_dir(configs, config_file_path, inference_result=False):
    """
    Prepare a directory for saving a training results.

    Args:
        configs: A python box object containing the configuration options.
        config_file_path: Directory of configuration file.
        inference_result: A boolean flag to indicate if the results are for inference.

    Returns:
        str: The path to the directory where the results will be saved.
    """
    # Create a unique identifier for the run based on the current time.
    run_id = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    # Add '_evaluation' to the run_id if the 'evaluate' flag is True.
    # if configs.evaluate:
    #     run_id += '_evaluation'

    # Create the result directory and the checkpoint subdirectory.
    result_path = os.path.abspath(os.path.join(configs.result_path, run_id))
    checkpoint_path = os.path.join(result_path, 'checkpoints')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    if not inference_result:
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file to the result directory.
    shutil.copy(config_file_path, result_path)

    # Return the path to the result directory.
    return result_path, checkpoint_path


def prepare_optimizer(net, configs, num_train_samples, logging):
    optimizer, scheduler = load_opt(net, configs, logging)
    if scheduler is None:
        whole_steps = np.ceil(
            num_train_samples / configs.train_settings.grad_accumulation
        ) * configs.train_settings.num_epochs / configs.optimizer.decay.num_restarts
        first_cycle_steps = np.ceil(whole_steps / configs.optimizer.decay.num_restarts)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=1.0,
            max_lr=configs.optimizer.lr,
            min_lr=configs.optimizer.decay.min_lr,
            warmup_steps=configs.optimizer.decay.warmup,
            gamma=configs.optimizer.decay.gamma)

    return optimizer, scheduler


def load_opt(model, config, logging):
    scheduler = None
    if config.optimizer.name.lower() == 'adabelief':
        opt = optim.AdaBelief(model.parameters(), lr=config.optimizer.lr, eps=config.optimizer.eps,
                              decoupled_decay=True,
                              weight_decay=config.optimizer.weight_decay, rectify=False)
    elif config.optimizer.name.lower() == 'adam':
        if config.optimizer.use_8bit_adam:
            import bitsandbytes
            logging.info('use 8-bit adamw')
            opt = bitsandbytes.optim.AdamW8bit(
                model.parameters(), lr=float(config.optimizer.lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                weight_decay=float(config.optimizer.weight_decay),
                eps=float(config.optimizer.eps),
            )
        else:
            opt = torch.optim.AdamW(
                model.parameters(), lr=float(config.optimizer.lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                weight_decay=float(config.optimizer.weight_decay),
                eps=float(config.optimizer.eps)
            )

    else:
        raise ValueError('wrong optimizer')
    return opt, scheduler


def load_checkpoints(configs, optimizer, scheduler, logging, net, accelerator):
    """
    Load saved checkpoints from a previous training session.

    Args:
        configs: A python box object containing the configuration options.
        optimizer (Optimizer): The optimizer to resume training with.
        scheduler (Scheduler): The learning rate scheduler to resume training with.
        logging (Logger): The logger to use for logging messages.
        net (nn.Module): The neural network model to load the saved checkpoints into.

    Returns:
        tuple: A tuple containing the loaded neural network model and the epoch to start training from.
    """
    start_epoch = 1

    # If the 'resume' flag is True, load the saved model checkpoints.
    if configs.resume.resume:
        model_checkpoint = torch.load(configs.resume.resume_path, map_location='cpu')
        # state_dict = model_checkpoint['model_state_dict']
        pretrained_state_dict = model_checkpoint['model_state_dict']
        pretrained_state_dict = {k.replace('_orig_mod.', ''): v for k, v in pretrained_state_dict.items()}
        model_state_dict = net.state_dict()

        if configs.resume.handle_shape_missmatch:
            if accelerator.is_main_process:
                logging.info(f'Consider handling shape miss match to reload the checkpoint.')

        for name, param in pretrained_state_dict.items():
            if name in model_state_dict:
                if model_state_dict[name].size() == param.size():
                    model_state_dict[name].copy_(param)
                elif configs.resume.handle_shape_missmatch:
                    # Copy only the overlapping parts of the tensor
                    # Assumes the mismatch is in the first dimension
                    if len(model_state_dict[name].size()) == 2:
                        min_size = min(model_state_dict[name].size(0), param.size(0))
                        model_state_dict[name][:min_size].copy_(param[:min_size])
                    else:
                        min_size = min(model_state_dict[name].size(1), param.size(1))
                        model_state_dict[name][:, :min_size].copy_(param[:, :min_size])
                    if accelerator.is_main_process:
                        logging.info(
                            f'Copied overlapping parts of this layer: {name}, Checkpoint shape: {param.size()}, Model shape: {model_state_dict[name].size()}')
                else:
                    if accelerator.is_main_process:
                        logging.info(
                            f'Ignore {name} layer, missmatch: Checkpoint shape: {param.size()}, Model shape: {model_state_dict[name].size()}')
            else:
                if accelerator.is_main_process:
                    logging.info(f'Ignore {name} layer, missmatch name')

        loading_log = net.load_state_dict(model_state_dict, strict=False)
        if accelerator.is_main_process:
            logging.info(f'Loading checkpoint log: {loading_log}')

        # If the saved checkpoint contains the optimizer and scheduler states and the epoch number,
        # resume training from the last saved epoch.
        if 'optimizer_state_dict' in model_checkpoint and 'scheduler_state_dict' in model_checkpoint and 'epoch' in model_checkpoint:
            if not configs.resume.restart_optimizer:
                optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
                if accelerator.is_main_process:
                    logging.info('Optimizer is loaded to resume training!')

                # scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
                # if accelerator.main_process:
                #     logging.info('Scheduler is loaded to resume training!')

            # start_epoch = model_checkpoint['epoch'] + 1
            start_epoch = 1
        if accelerator.is_main_process:
            logging.info('Model is loaded to resume training!')
    return net, start_epoch


def load_checkpoints_inference(checkpoint_path, logging, net):
    """
    Load a PyTorch checkpoint from a specified path into the provided model.

    Args:
        checkpoint_path (str): The file path to the checkpoint.
        logging (Logger): Logger for logging messages.
        net (torch.nn.Module): The model into which the checkpoint will be loaded.

    Returns:
        torch.nn.Module: The model with loaded checkpoint weights.
    """
    # Check if the checkpoint file exists
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint file does not exist at {checkpoint_path}")
        return net

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Removing the prefix "_orig_mod." from the keys of the model checkpoint if it exists.
    checkpoint['model_state_dict'] = remove_prefix_from_keys(checkpoint['model_state_dict'],
                                                             '_orig_mod.')

    # Load state dict into the model
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)

    logging.info(f"Checkpoint loaded successfully from {checkpoint_path}")

    return net


def save_checkpoint(epoch: int, model_path: str, tools: dict, accelerator: Accelerator):
    """
    Save the model checkpoints during training.

    Args:
        epoch (int): The current epoch number.
        model_path (str): The path to save the model checkpoint.
        tools (dict): A dictionary containing the necessary tools for saving the model checkpoints.
        accelerator (Accelerator): Accelerator object.

    Returns:
        None
    """

    # Save the model checkpoint.
    torch.save({
        'epoch': epoch,
        'model_state_dict': accelerator.unwrap_model(tools['net']).state_dict(),
        'optimizer_state_dict': accelerator.unwrap_model(tools['optimizer'].state_dict()),
        'scheduler_state_dict': accelerator.unwrap_model(tools['scheduler'].state_dict()),
    }, model_path)


def test_gpu_cuda():
    print('Testing gpu and cuda:')
    print('\tcuda is available:', torch.cuda.is_available())
    print('\tdevice count:', torch.cuda.device_count())
    print('\tcurrent device:', torch.cuda.current_device())
    print(f'\tdevice:', torch.cuda.device(0))
    print('\tdevice name:', torch.cuda.get_device_name(), end='\n\n')


class Tokenizer:
    def __init__(self, tasks_tokens_list: list,
                 amino_to_fold_seek_label_index_mapping: dict,
                 localization_deeploc_label_index_mapping: dict,
                 er_label_index_mapping: dict, fold_label_index_mapping: dict,
                 secondary_structure_label_index_mapping: dict,
                 gene_ontology_label_index_mapping: dict,
                 human_ppi_label_index_mapping: dict,
                 protein_protein_interface_label_list: list,
                 stability_label: list, structure_similarity_label: list, fluorescence_label: list,
                 label_tokens: list,
                 configs, max_label_index=500, **kwargs):
        self.amino_acid_chars = 'ACDEFGHIKLMNPQRSTVWYUXZB'
        self.max_label_index = max_label_index

        self.er_label_index_mapping = er_label_index_mapping

        self.ec_label_index_mapping = kwargs["ec_label_index_mapping"]
        self.broken_ec_label_index_mapping = kwargs['broken_ec_label_index_mapping']

        self.human_ppi_label_index_mapping = human_ppi_label_index_mapping
        self.protein_protein_interface_label_list = protein_protein_interface_label_list
        self.stability_label = stability_label
        self.protein_ligand_affinity_label = kwargs['protein_ligand_affinity_label']
        self.structure_similarity_label = structure_similarity_label
        self.fluorescence_label = fluorescence_label
        self.amino_to_fold_seek_label_index_mapping = amino_to_fold_seek_label_index_mapping
        self.localization_deeploc_label_index_mapping = localization_deeploc_label_index_mapping
        self.secondary_structure_label_index_mapping = secondary_structure_label_index_mapping
        self.fold_label_index_mapping = fold_label_index_mapping
        if gene_ontology_label_index_mapping != {}:
            self.gene_ontology_label_index_mapping = gene_ontology_label_index_mapping

        self.tokens_dict = dict()
        self.index_token_dict = dict()
        self.tokens_dict['<pad>'] = 0
        self.tokens_dict['<bos>'] = 1
        self.tokens_dict['<eos>'] = 2
        self.tokens_dict['<sep>'] = 3
        # self.tokens_dict['-'] = 4

        for task_token in tasks_tokens_list:
            self.tokens_dict[task_token] = max(list(self.tokens_dict.values())) + 1

        # # for dummy tokens
        # for i in range(5):
        #     self.tokens_dict[f"dummy_{i}"] = max(list(self.tokens_dict.values())) + 1

        task_attributes = [
            configs.tasks.phosphorylation,
            configs.tasks.localization,
            configs.tasks.auxiliary,
            configs.tasks.protein_protein_interface,
        ]

        if configs.tasks.fold:
            for i in range(1, 1195 + 1):
                if f'{i}' not in self.tokens_dict.keys():
                    self.tokens_dict[f'{i}'] = max(list(self.tokens_dict.values())) + 1

        if any(task_attributes):
            for i in range(max_label_index + 1):
                if f'{i}' not in self.tokens_dict.keys():
                    self.tokens_dict[f'{i}'] = max(list(self.tokens_dict.values())) + 1

        for class_token in label_tokens:
            if class_token not in self.tokens_dict.keys():
                self.tokens_dict[class_token] = max(list(self.tokens_dict.values())) + 1
            else:
                pass

        self.vocab_size = len(self.tokens_dict)
        self.index_token_dict = self.update_index_token_dict()

    def update_index_token_dict(self):
        return {value: key for key, value in self.tokens_dict.items()}

    def fit(self, samples_list):
        concatenated_str = ''.join(samples_list)
        unique_chars = set(concatenated_str)
        unique_chars_string = ''.join(unique_chars)
        sorted_string = ''.join(sorted(unique_chars_string))
        for a in sorted_string:
            if a not in self.tokens_dict.keys():
                self.tokens_dict[a] = max(list(self.tokens_dict.values())) + 1

        self.index_token_dict = self.update_index_token_dict()

    def __call__(self,
                 labels: list,
                 task_name: int,
                 max_target_len: int):

        encoded_target = [self.tokens_dict['<bos>'], self.tokens_dict[task_name]]
        for p in labels:
            encoded_target.append(self.tokens_dict[str(p)])
        encoded_target.append(self.tokens_dict['<eos>'])

        encoded_target += [self.tokens_dict['<pad>'] for i in range(max_target_len - len(encoded_target))]

        return encoded_target


class InferenceTokenizer:
    def __init__(self, token_dict):
        self.tokens_dict = token_dict
        self.index_token_dict = self.update_index_token_dict()
        self.vocab_size = len(self.tokens_dict)

    def update_index_token_dict(self):
        return {value: key for key, value in self.tokens_dict.items()}

    def __call__(self,
                 task_name: int):
        encoded_target = [self.tokens_dict['<bos>'], self.tokens_dict[task_name]]
        return encoded_target


def prepare_tensorboard(result_path):
    train_path = os.path.join(result_path, 'train')
    val_path = os.path.join(result_path, 'val')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)

    train_log_path = os.path.join(train_path, 'tensorboard')
    train_writer = SummaryWriter(train_log_path)

    val_log_path = os.path.join(val_path, 'tensorboard')
    val_writer = SummaryWriter(val_log_path)

    return train_writer, val_writer


def get_logging(result_path):
    logger = log.getLogger(result_path)
    logger.setLevel(log.INFO)

    fh = log.FileHandler(os.path.join(result_path, "logs.txt"))
    formatter = log.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = log.StreamHandler()
    logger.addHandler(sh)

    return logger


def get_dummy_logger():
    # Create a logger object
    logger = log.getLogger('dummy')
    logger.setLevel(log.INFO)

    # Create a string buffer to hold the logs
    log_buffer = StringIO()

    # Create a stream handler that writes to the string buffer
    handler = log.StreamHandler(log_buffer)
    formatter = log.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Optionally disable propagation to prevent logging on the parent logger
    logger.propagate = False

    # Return both logger and buffer so you can inspect logs as needed
    return logger, log_buffer


def save_model(epoch, model, opt, result_path, scheduler, description='best_model'):
    Path(os.path.join(result_path, 'checkpoints')).mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'epoch': epoch
    }, os.path.join(result_path, 'checkpoints', description + '.pth'))


def random_pick(input_list, num_to_pick, seed):
    # Set the random seed
    random.seed(seed)

    # Check if num_to_pick is greater than the length of the input_list
    if num_to_pick > len(input_list):
        print("Number to pick is greater than the length of the input list")
        return input_list

    # Use random.sample to pick num_to_pick items from the input_list
    random_items = random.sample(input_list, num_to_pick)

    return random_items


if __name__ == '__main__':
    # For test utils modules
    print('done')
