import os
import yaml
import torch
from box import Box


def prepare_config_and_checkpoint(name):
    """
    Prepare the configuration dictionary, the checkpoint file path for the model with the given name.

    Args:
        name (str): The name of the model.

    Returns:
        configs: The configuration in a box object.
        checkpoint_path: The file path to the checkpoint file.
        decoder_tokenizer_dict: The dictionary containing the decoder tokenizer.
    """
    from huggingface_hub import hf_hub_download
    if name == 'stability':
        checkpoint_path = hf_hub_download(
            repo_id="Mahdip72/prot2token",
            filename="stability/2024-07-05__17-35-31/checkpoints/best_valid_stability_spearman.pth"
        )
        decoder_tokenizer_path = hf_hub_download(
            repo_id="Mahdip72/prot2token",
            filename="stability/2024-07-05__17-35-31/decoder_tokenizer.yaml"
        )
        config_file_path = hf_hub_download(
            repo_id="Mahdip72/prot2token",
            filename="stability/2024-07-05__17-35-31/config.yaml"
        )
    elif name == 'fluorescence':
        checkpoint_path = hf_hub_download(
            repo_id="Mahdip72/prot2token",
            filename="fluorescence/2024-04-23__18-20-05/checkpoints/best_valid_fluorescence_spearman.pth"
        )
        decoder_tokenizer_path = hf_hub_download(
            repo_id="Mahdip72/prot2token",
            filename="fluorescence/2024-04-23__18-20-05/decoder_tokenizer.yaml"
        )
        config_file_path = hf_hub_download(
            repo_id="Mahdip72/prot2token",
            filename="fluorescence/2024-04-23__18-20-05/config.yaml"
        )
    elif name == 'kinase_group':
        checkpoint_path = hf_hub_download(
            repo_id="Mahdip72/prot2token",
            filename="kinase_group/2025-02-03__23-14-25/checkpoints/best_valid_kinase_group_macro_f1.pth"
        )
        decoder_tokenizer_path = hf_hub_download(
            repo_id="Mahdip72/prot2token",
            filename="kinase_group/2025-02-03__23-14-25/decoder_tokenizer.yaml"
        )
        config_file_path = hf_hub_download(
            repo_id="Mahdip72/prot2token",
            filename="kinase_group/2025-02-03__23-14-25/config.yaml"
        )
    elif name == 'kinase_phosphorylation_site':
        checkpoint_path = hf_hub_download(
            repo_id="Mahdip72/prot2token",
            filename="kinase_phosphorylation_site/2025-01-21__22-15-51/checkpoints/best_valid_kinase_phosphorylation_site_f1.pth"
        )
        decoder_tokenizer_path = hf_hub_download(
            repo_id="Mahdip72/prot2token",
            filename="kinase_phosphorylation_site/2025-01-21__22-15-51/decoder_tokenizer.yaml"
        )
        config_file_path = hf_hub_download(
            repo_id="Mahdip72/prot2token",
            filename="kinase_phosphorylation_site/2025-01-21__22-15-51/config.yaml"
        )

    else:
        raise ValueError(f"Model with name '{name}' is not supported.")

    # Load the configuration file
    with open(config_file_path) as file:
        dict_config = yaml.full_load(file)

    configs = load_configs(dict_config)

    with open(decoder_tokenizer_path) as file:
        decoder_tokenizer_dict = yaml.full_load(file)

    return configs, checkpoint_path, decoder_tokenizer_dict


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


def load_configs(config, inference=True):
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
    return tree_config


def load_checkpoints_inference(checkpoint_path, net):
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
        print(f"Checkpoint file does not exist at {checkpoint_path}")
        return net

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Removing the prefix "_orig_mod." from the keys of the model checkpoint if it exists.
    checkpoint['model_state_dict'] = remove_prefix_from_keys(checkpoint['model_state_dict'],
                                                             '_orig_mod.')

    # Load state dict into the model
    load_log = net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f'Loading checkpoint log: {load_log}')

    return net


if __name__ == '__main__':
    # For test utils modules
    print('done')
