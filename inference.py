import os
import numpy as np
import transformers
import yaml
import argparse
import torch
from time import time
from tqdm import tqdm
import pandas as pd
from utils.utils import load_configs, test_gpu_cuda, prepare_tensorboard, prepare_optimizer, save_checkpoint, \
    get_logging, load_checkpoints_inference, prepare_saving_dir, remove_prefix_from_keys
from dataset import prepare_dataloaders
from model import prepare_models
from accelerate import Accelerator
from transformers import AutoTokenizer
from accelerate import DataLoaderConfiguration
from metrics import prepare_metrics_dict, compute_metrics, initializing_monitoring_metrics, compute_all_metrics
from utils.log import evaluation_log


class Tokenizer:
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


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, configs, **kwargs):
        self.configs = configs
        self.inference_configs = kwargs["inference_configs"]
        self.molecule_encoder_tokenizer = kwargs["molecule_encoder_tokenizer"]
        self.protein_encoder_tokenizer = kwargs["protein_encoder_tokenizer"]
        self.decoder_tokenizer = kwargs["decoder_tokenizer"]
        self.max_protein_encoder_length = configs.prot2token_model.protein_encoder.max_len
        self.max_molecule_encoder_length = configs.prot2token_model.molecule_encoder.max_len
        self.max_decoder_length = configs.prot2token_model.decoder.max_len
        self.items = self.get_input_task_pairs(self.inference_configs.data_path)[:20]

        self.max_protein_encoder_length = configs.prot2token_model.protein_encoder.max_len
        self.max_molecule_encoder_length = configs.prot2token_model.molecule_encoder.max_len
        self.max_decoder_length = configs.prot2token_model.decoder.max_len

    def __len__(self):
        return len(self.items)

    @staticmethod
    def get_input_task_pairs(csv_path):
        """
        Reads a CSV file and returns a list of (input, task_name) pairs.

        Parameters:
        csv_path (str): The directory path to the CSV file.

        Returns:
        list: A list containing (input, task_name) pairs.
        """
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Create the list of (input, task_name) pairs
        pairs = [(row['input'], '<task_stability>') for _, row in df.iterrows()]

        return pairs

    def __getitem__(self, idx):
        sequence, task_name = self.items[idx]

        encoded_target = self.decoder_tokenizer(task_name=task_name)

        if len(sequence) == 3:
            smiles_sequence = sequence[2]
            sequence = sequence[0]
        else:
            smiles_sequence = ""

        if self.protein_encoder_tokenizer:
            encoded_protein_sequence = self.protein_encoder_tokenizer(
                sequence, max_length=self.max_protein_encoder_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            encoded_protein_sequence['input_ids'] = torch.squeeze(encoded_protein_sequence['input_ids'])
            encoded_protein_sequence['attention_mask'] = torch.squeeze(encoded_protein_sequence['attention_mask'])
        else:
            encoded_protein_sequence = torch.LongTensor(torch.zeros(1, 64, 320))

        if self.molecule_encoder_tokenizer:
            encoded_molecule_sequence = self.molecule_encoder_tokenizer(smiles_sequence,
                                                                        max_length=self.max_molecule_encoder_length,
                                                                        padding='max_length',
                                                                        truncation=True,
                                                                        return_tensors="pt",
                                                                        add_special_tokens=True
                                                                        )

            encoded_molecule_sequence['input_ids'] = torch.squeeze(encoded_molecule_sequence['input_ids'])
            encoded_molecule_sequence['attention_mask'] = torch.squeeze(encoded_molecule_sequence['attention_mask'])
        else:
            encoded_molecule_sequence = torch.LongTensor(torch.zeros(1, 64, 320))

        encoded_target = torch.LongTensor(encoded_target)

        return encoded_protein_sequence, encoded_target, encoded_molecule_sequence, sequence, task_name


def prepare_inference_dataloader(configs, inference_configs):
    protein_encoder_tokenizer = AutoTokenizer.from_pretrained(configs.prot2token_model.protein_encoder.model_name)

    molecule_encoder_tokenizer = AutoTokenizer.from_pretrained("gayane/BARTSmiles",
                                                               add_prefix_space=True)
    molecule_encoder_tokenizer.pad_token = '<pad>'
    molecule_encoder_tokenizer.bos_token = '<s>'
    molecule_encoder_tokenizer.eos_token = '</s>'
    molecule_encoder_tokenizer.mask_token = '<unk>'

    with open(inference_configs.decoder_tokenizer_path) as file:
        decoder_tokenizer_dict = yaml.full_load(file)

    decoder_tokenizer = Tokenizer(decoder_tokenizer_dict)

    inference_dataset = InferenceDataset(configs,
                                         inference_configs=inference_configs,
                                         molecule_encoder_tokenizer=molecule_encoder_tokenizer,
                                         protein_encoder_tokenizer=protein_encoder_tokenizer,
                                         decoder_tokenizer=decoder_tokenizer)

    if inference_configs.batch_size > 1:
        RuntimeError("Does not support batch size > 1 for inference. Please set batch size to 1.")

    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=inference_configs.batch_size,
        shuffle=False,
        num_workers=inference_configs.num_workers,
    )
    return inference_dataloader, protein_encoder_tokenizer, molecule_encoder_tokenizer, decoder_tokenizer


def inference(net, dataloader, configs, decoder_tokenizer, mode):
    counter = 0
    results = []
    net.eval()
    for i, data in enumerate(tqdm(dataloader, desc=f'{mode}', total=len(dataloader),
                                  leave=False, disable=not configs.tqdm_progress_bar)):
        protein_sequence, target, molecule_sequence, sequence, task_name = data

        batch = {"protein_sequence": protein_sequence, "molecule_sequence": molecule_sequence,
                 "target_input": target}

        with torch.inference_mode():
            preds = net(batch, mode="inference_greedy")
            preds = preds.detach().cpu().numpy().tolist()[0]
            preds = [decoder_tokenizer.index_token_dict[pred] for pred in preds[2:-1]]
            results.append([sequence[0], task_name[0], ''.join(preds)])
        counter += 1

    return results


def main(dict_inference_config, dict_config, inference_config_file_path):
    configs = load_configs(dict_config)
    inference_configs = load_configs(dict_inference_config, inference=True)
    transformers.logging.set_verbosity_error()

    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()
    test_gpu_cuda()

    result_path, _ = prepare_saving_dir(inference_configs, inference_config_file_path, inference_result=True)

    logging = get_logging(result_path)

    inference_dataloader, protein_encoder_tokenizer, molecule_encoder_tokenizer, decoder_tokenizer = prepare_inference_dataloader(
        configs, inference_configs)
    logging.info('preparing dataloaders are done')

    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        dataloader_config=dataloader_config
    )

    net = prepare_models(configs, protein_encoder_tokenizer, decoder_tokenizer, logging, accelerator, inference=True)
    logging.info('preparing model is done')

    net = load_checkpoints_inference(inference_configs.checkpoint_path, logging, net)
    logging.info('loading model weights is done')

    # Compile model to predict faster and efficiently on GPU
    if inference_configs.compile_model:
        net = torch.compile(net)
        if accelerator.is_main_process:
            logging.info('compile model is done')

    net, inference_dataloader = accelerator.prepare(net, inference_dataloader)
    net.to(accelerator.device)

    if accelerator.is_main_process:
        inference_steps = len(inference_dataloader)
        logging.info(f'number of inference steps: {int(inference_steps)}')

    torch.cuda.empty_cache()

    start_time = time()
    inference_results = inference(net, inference_dataloader, inference_configs, decoder_tokenizer, mode='inference')
    end_time = time()
    inference_time = end_time - start_time

    if accelerator.is_main_process:
        logging.info(
            f'inference dataset 1 - steps {len(inference_dataloader)} - time {np.round(inference_time, 2)}s')

    inference_results = pd.DataFrame(inference_results, columns=['input', 'task_name', 'predicted'])
    inference_results.to_csv(os.path.join(result_path, 'inference_results.csv'), index=False)

    accelerator.free_memory()
    del net, inference_dataloader, accelerator
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Doing inference of a pre-trained Prot2Token model.")
    parser.add_argument("--config_path", "-c", help="The location of inference config file",
                        default='./configs/inference_config.yaml')
    args = parser.parse_args()
    inference_config_path = args.config_path

    with open(inference_config_path) as file:
        inference_config_file = yaml.full_load(file)
    result_config_path = inference_config_file['result_config_path']

    with open(result_config_path) as file:
        result_config_file = yaml.full_load(file)
    main(inference_config_file, result_config_file, inference_config_path)
    print('done!')
