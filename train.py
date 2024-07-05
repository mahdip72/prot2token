import os
import numpy as np
import transformers
import yaml
import argparse
import torch
from time import time, sleep
from tqdm import tqdm
from utils.utils import load_configs, test_gpu_cuda, prepare_tensorboard, prepare_optimizer, save_checkpoint, \
    get_logging, load_checkpoints, prepare_saving_dir, remove_prefix_from_keys
from dataset import prepare_dataloaders
from model import prepare_models
from accelerate import Accelerator
from accelerate import DataLoaderConfiguration
from metrics import prepare_metrics_dict, compute_metrics, initializing_monitoring_metrics, compute_all_metrics
from utils.log import training_tensorboard_log, validation_tensorboard_log, evaluation_log, training_log


def train(epoch, accelerator, tools, global_step, configs):
    tools["optimizer"].zero_grad()

    train_loss = 0.0
    epoch_loss = 0
    counter = 0

    progress_bar = tqdm(range(global_step, int(np.ceil(len(tools['train_loader']) / tools['accum_iter']))),
                        disable=not configs.tqdm_progress_bar, leave=False)
    progress_bar.set_description("Steps")

    metrics_dict = prepare_metrics_dict(accelerator)
    for i, data in enumerate(tools['train_loader']):
        with accelerator.accumulate(tools['net']):
            protein_sequence, target, sample_weight, molecule_sequence = data
            target_input = target[:, :-1]
            target_expected = target[:, 1:]

            batch = {"protein_sequence": protein_sequence, "molecule_sequence": molecule_sequence,
                     "target_input": target_input}

            preds = tools['net'](batch)
            # loss = tools['loss_function'](preds.reshape(-1, preds.shape[-1]), target_expected.reshape(-1))

            # Flatten the tensors
            preds_flatten = preds.view(-1, preds.size(2))
            target_expected_flatten = target_expected.contiguous().view(-1)

            # if configs.train_settings.loss == 'crossentropy':
            losses = tools['loss_function'](preds_flatten, target_expected_flatten)

            # Reshape losses to [batch_size, seq_length]
            losses = losses.view(preds.size(0), preds.size(1))

            if configs.train_settings.sample_weight:
                # Multiply each sequence loss by the sample weight
                losses = losses * sample_weight

            # add zero weight to the first token (class token)
            weights = torch.ones(losses.shape).to(accelerator.device)
            weights[:, 0] = 0
            loss = torch.mean(losses * weights)

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(tools["train_batch_size"])).mean()
            train_loss += avg_loss.item() / tools['accum_iter']

            if epoch > configs.train_settings.start_metric_epoch:
                metrics_dict = compute_metrics(
                    accelerator.gather_for_metrics(preds.detach()),
                    accelerator.gather_for_metrics(target_expected.detach()),
                    tools['decoder_tokenizer'], metrics_dict,
                    accelerator
                )

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(tools['net'].parameters(), tools['grad_clip'])

            tools['optimizer'].step()
            tools['scheduler'].step()
            tools['optimizer'].zero_grad()

        if accelerator.sync_gradients:
            tools['train_writer'].add_scalar('step loss', train_loss, global_step)
            tools['train_writer'].add_scalar('learning rate', tools['optimizer'].param_groups[0]['lr'], global_step)

            progress_bar.update(1)
            global_step += 1

            counter += 1
            epoch_loss += train_loss
            train_loss = 0

        logs = {"step_loss": loss.detach().item(),
                "lr": tools['optimizer'].param_groups[0]['lr']}
        progress_bar.set_postfix(**logs)

    train_loss = epoch_loss / counter

    return train_loss, metrics_dict


def evaluation(accelerator, dataloader, tools, name, configs, mode):
    valid_loss = 0
    counter = 0
    metrics_dict = prepare_metrics_dict(accelerator)
    for i, data in enumerate(tqdm(dataloader, desc=f'{mode} {name}', total=len(dataloader),
                                  leave=False, disable=not configs.tqdm_progress_bar)):
        protein_sequence, target, sample_weight, molecule_sequence = data
        target_input = target[:, :-1]
        target_expected = target[:, 1:]

        batch = {"protein_sequence": protein_sequence, "molecule_sequence": molecule_sequence,
                 "target_input": target_input}

        with torch.inference_mode():
            preds = tools['net'](batch, mode="prediction")
            loss = tools['loss_function'](preds.reshape(-1, preds.shape[-1]), target_expected.reshape(-1))
            weights = torch.ones(loss.shape).to(accelerator.device)
            weights[..., 0] = 0
            loss = torch.mean(loss * weights, dim=-1)

        metrics_dict = compute_metrics(
            accelerator.gather_for_metrics(preds.detach()),
            accelerator.gather_for_metrics(target_expected.detach()),
            tools['decoder_tokenizer'], metrics_dict,
            accelerator
        )
        counter += 1
        valid_loss += loss.data.item()

    valid_loss = valid_loss / len(dataloader.dataset)

    return valid_loss, metrics_dict


def main(dict_config, config_file_path):
    configs = load_configs(dict_config)
    transformers.logging.set_verbosity_error()

    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()
    test_gpu_cuda()

    result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)

    logging = get_logging(result_path)

    dataloaders_dict, encoder_tokenizer, decoder_tokenizer = prepare_dataloaders(configs, logging, result_path)
    logging.info('preparing dataloaders are done')

    dataloader_config = DataLoaderConfiguration(dispatch_batches=True)
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
    )

    net = prepare_models(configs, encoder_tokenizer, decoder_tokenizer, logging, accelerator)
    logging.info('preparing model is done')

    optimizer, scheduler = prepare_optimizer(net, configs, len(dataloaders_dict["train"]), logging)
    logging.info('preparing optimizer is done')

    net, optimizer, dataloaders_dict["train"], scheduler = accelerator.prepare(
        net, optimizer, dataloaders_dict["train"], scheduler
    )
    net, start_epoch = load_checkpoints(configs, optimizer, scheduler, logging, net, accelerator)

    for name, dataset in dataloaders_dict["valids"].items():
        dataloaders_dict["valids"][name] = accelerator.prepare(dataset)

    if configs.test_settings.enable:
        for name, dataset in dataloaders_dict["tests"].items():
            dataloaders_dict["tests"][name] = accelerator.prepare(dataset)

    net.to(accelerator.device)

    # compile model to train faster and efficiently
    if configs.prot2token_model.compile_model:
        net = torch.compile(net)
        if accelerator.is_main_process:
            logging.info('compile model is done')

    # initialize tensorboards
    train_writer, valid_writer = prepare_tensorboard(result_path)

    # prepare loss function
    if configs.train_settings.loss == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=decoder_tokenizer.tokens_dict['<pad>'],
            reduction="none"
        )
    else:
        logging.error("wrong loss!")
        raise ValueError("wrong loss!")

    tools = {
        'net': net,
        'train_loader': dataloaders_dict["train"],
        'valid_loader': dataloaders_dict["valids"],
        'train_batch_size': configs.train_settings.batch_size,
        'valid_batch_size': configs.valid_settings.batch_size,
        'optimizer': optimizer,
        'mixed_precision': configs.train_settings.mixed_precision,
        'tensorboard_log': configs.tensorboard_log,
        'train_writer': train_writer,
        'valid_writer': valid_writer,
        'accum_iter': configs.train_settings.grad_accumulation,
        'loss_function': criterion,
        'grad_clip': configs.optimizer.grad_clip_norm,
        'checkpoints_every': configs.checkpoints_every,
        'scheduler': scheduler,
        'result_path': result_path,
        'checkpoint_path': checkpoint_path,
        'encoder_tokenizer': encoder_tokenizer,
        'decoder_tokenizer': decoder_tokenizer,
        'logging': logging,
    }

    if accelerator.is_main_process:
        train_steps = np.ceil(len(tools["train_loader"]) / configs.train_settings.grad_accumulation)
        logging.info(f'number of train steps per epoch: {int(train_steps)}')
        for name, val_loader in tools['valid_loader'].items():
            logging.info(f'number of {name} valid steps per epoch: {int(len(val_loader))}')

    global_step = 0
    monitoring_metrics_dict = initializing_monitoring_metrics(configs)
    for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
        if configs.train_settings.skip:
            logging.info(f'skip training')
            break
        tools['epoch'] = epoch
        tools['net'].train()
        start_time = time()
        train_loss, metrics_dict = train(epoch, accelerator, tools, global_step, configs)
        end_time = time()
        training_time = end_time - start_time
        training_tensorboard_log(epoch, train_loss, metrics_dict, tools, configs)
        training_log(epoch, train_loss, accelerator, tools, configs, metrics_dict, training_time, logging)

        if epoch > configs.train_settings.start_metric_epoch:
            if epoch % configs.valid_settings.do_every == 0:
                tools['net'].eval()
                for i, (task_name, dataloader) in enumerate(dataloaders_dict['valids'].items()):
                    start_time = time()
                    valid_loss, metrics_dict = evaluation(accelerator, dataloader, tools, task_name, configs,
                                                          mode='valid')
                    end_time = time()
                    evaluation_time = end_time - start_time
                    validation_tensorboard_log(task_name, epoch, valid_loss, metrics_dict, tools, configs)
                    evaluation_log(i, task_name, valid_loss, dataloader, accelerator, metrics_dict, evaluation_time,
                                   logging, mode='valid')

                    if task_name in monitoring_metrics_dict.keys():
                        metric_name = list(monitoring_metrics_dict[task_name].keys())[0]
                        if metrics_dict[task_name]['correct_prediction'] > 0:
                            if metric_name in ['rmse', 'mse', 'mae']:
                                condition = monitoring_metrics_dict[task_name][metric_name] > metrics_dict[task_name][
                                    metric_name]
                            else:
                                condition = monitoring_metrics_dict[task_name][metric_name] < metrics_dict[task_name][
                                    metric_name]
                            if condition:
                                monitoring_metrics_dict[task_name][metric_name] = metrics_dict[task_name][metric_name]

                                # Set the path to save the model checkpoint.
                                model_path = os.path.join(tools['result_path'],
                                                          'checkpoints', f'best_valid_{task_name}_{metric_name}.pth')
                                accelerator.wait_for_everyone()
                                save_checkpoint(epoch, model_path, tools, accelerator)
                                if accelerator.is_main_process:
                                    logging.info(
                                        f"\tnew best {metric_name} for {task_name}: {monitoring_metrics_dict[task_name][metric_name]: .4}")
                                    logging.info(f'\tsaving the best model in {model_path}')

        if epoch % configs.checkpoints_every == 0:
            # Set the path to save the model checkpoint.
            model_path = os.path.join(tools['result_path'], 'checkpoints', f'checkpoint_{epoch}.pth')
            accelerator.wait_for_everyone()
            save_checkpoint(epoch, model_path, tools, accelerator)

    train_writer.close()
    valid_writer.close()

    # pause 20 second to make sure the best validation checkpoint is ready on the disk
    sleep(20)

    for param in tools['net'].parameters():
        param.requires_grad = False
    torch.cuda.empty_cache()

    if configs.test_settings.enable:
        if configs.train_settings.skip and configs.resume.resume:
            resume_path = configs.resume.resume_path
            tools['result_path'] = os.path.abspath(resume_path[:resume_path.index("checkpoints")])
            logging.info(f"consider checkpoints from {tools['result_path']} to use for the test sets")

        if accelerator.is_main_process:
            logging.info('\n\nstart testing the best validation checkpoints')

        for i, (task_name, dataloader) in enumerate(dataloaders_dict['tests'].items()):
            if task_name in monitoring_metrics_dict.keys():
                metric_name = list(monitoring_metrics_dict[task_name].keys())[0]
                model_path = os.path.join(tools['result_path'], 'checkpoints',
                                          f'best_valid_{task_name}_{metric_name}.pth')
                if not os.path.exists(model_path):
                    if accelerator.is_main_process:
                        logging.info(f'\n\t{model_path} does not exist')
                    continue
                else:
                    if accelerator.is_main_process:
                        logging.info(f'\n\ttesting this checkpoint: {model_path}')

                model_checkpoint = torch.load(model_path, map_location='cpu')

                if list(model_checkpoint['model_state_dict'].keys())[0].split('.')[0] != list(tools['net'].state_dict().keys())[0].split('.')[0]:
                    print(list(model_checkpoint['model_state_dict'].keys())[0])

                    # Removing the prefix "_orig_mod." from the keys of the model checkpoint if it exists.
                    model_checkpoint['model_state_dict'] = remove_prefix_from_keys(model_checkpoint['model_state_dict'],
                                                                                   '_orig_mod.')

                    print(list(model_checkpoint['model_state_dict'].keys())[0])

                tools['net'] = accelerator.unwrap_model(tools['net'])
                loading_log = tools['net'].load_state_dict(model_checkpoint['model_state_dict'], strict=True)
                logging.info(f'Loading checkpoint log: {loading_log}')
                tools['net'] = accelerator.prepare(tools['net'])

                del model_checkpoint

                tools['net'].eval()
                start_time = time()
                valid_loss, metrics_dict = evaluation(accelerator, dataloader, tools, task_name, configs, mode='test')
                end_time = time()
                evaluation_time = end_time - start_time
                metrics_dict = compute_all_metrics(metrics_dict, configs)
                evaluation_log(i, task_name, valid_loss, dataloader, accelerator, metrics_dict, evaluation_time,
                               logging, mode='test')

    accelerator.free_memory()
    del tools, net, dataloaders_dict, accelerator, optimizer, scheduler
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Prot2Token model via joint training of multiple tasks.")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./configs/config.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
    print('done!')
