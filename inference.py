import os
import numpy as np
import transformers
import yaml
import argparse
import torch
from time import time
from tqdm import tqdm
import pandas as pd
from utils.utils import load_configs, test_gpu_cuda, get_logging, load_checkpoints_inference, prepare_saving_dir
from dataset import prepare_inference_dataloader
from model import prepare_models
from accelerate import Accelerator
from accelerate import DataLoaderConfiguration


def inference(net, dataloader, configs, decoder_tokenizer, mode):
    counter = 0
    results = []
    net.eval()

    inference_config = {
        "beam_width": (configs.beam_search.beam_width,),
        "temperature": (configs.beam_search.temperature,),
        "top_k": configs.beam_search.top_k
    }
    for i, data in enumerate(tqdm(dataloader, desc=f'{mode}', total=len(dataloader),
                                  leave=False, disable=not configs.tqdm_progress_bar)):
        protein_sequence, target, molecule_sequence, sequence, task_name = data

        batch = {"protein_sequence": protein_sequence, "molecule_sequence": molecule_sequence,
                 "target_input": target}

        with torch.inference_mode():
            preds = net(batch, mode=configs.inference_type, inference_config=inference_config)
            preds = preds.detach().cpu().numpy().tolist()[0]
            preds = [decoder_tokenizer.index_token_dict[pred] for pred in preds[2:-1]]
            results.append([sequence[0], task_name[0], configs.merging_character.join(preds)])
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
