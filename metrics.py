import numpy as np
import torch.nn.functional as F
import torch
import torchmetrics
from tasks.protein_ligand_affinity import unmap_labels

ptms_list = ['phosphorylation']


def initializing_monitoring_metrics(configs):
    monitoring_metrics = {}
    for task, metric in configs.test_settings.monitoring_metrics.items():
        if task == 'gene_ontology' and getattr(configs.tasks, task, False):
            monitoring_metrics['gene_ontology_bp'] = {}
            monitoring_metrics['gene_ontology_mf'] = {}
            monitoring_metrics['gene_ontology_cc'] = {}
            monitoring_metrics['gene_ontology_bp'][metric] = -1.0
            monitoring_metrics['gene_ontology_mf'][metric] = -1.0
            monitoring_metrics['gene_ontology_cc'][metric] = -1.0
        elif metric in ['rmse', 'mse', 'mae'] and getattr(configs.tasks, task, False):
            monitoring_metrics[task] = {}
            monitoring_metrics[task][metric] = 10000.0
        else:
            if getattr(configs.tasks, task, False):
                monitoring_metrics[task] = {}
                monitoring_metrics[task][metric] = -1.0

    return monitoring_metrics


def compute_all_metrics(metrics_dict, configs):
    for ptm in ptms_list:
        metrics_dict[ptm] = compute_ptm_metrics(metrics_dict[ptm])
    if configs.tasks.auxiliary:
        metrics_dict['auxiliary'] = compute_auxiliary_metrics(metrics_dict['auxiliary'])
    if configs.tasks.localization:
        metrics_dict['localization'] = compute_localization_metrics(metrics_dict['localization'])
    if configs.tasks.localization_deeploc:
        metrics_dict['localization_deeploc'] = compute_localization_deeploc_metrics(
            metrics_dict['localization_deeploc'])
    if configs.tasks.fold:
        metrics_dict['fold'] = compute_fold_metrics(metrics_dict['fold'])
    if configs.tasks.amino_to_fold_seek:
        metrics_dict['amino_to_fold_seek'] = compute_amino_to_fold_seek_metrics(metrics_dict['amino_to_fold_seek'])
    if configs.tasks.secondary_structure:
        metrics_dict['secondary_structure'] = compute_sequence_prediction_metrics(metrics_dict['secondary_structure'])
    if configs.tasks.gene_ontology:
        metrics_dict['gene_ontology_bp'] = compute_gene_ontology_metrics(metrics_dict['gene_ontology_bp'])
        metrics_dict['gene_ontology_mf'] = compute_gene_ontology_metrics(metrics_dict['gene_ontology_mf'])
        metrics_dict['gene_ontology_cc'] = compute_gene_ontology_metrics(metrics_dict['gene_ontology_cc'])

    if configs.tasks.enzyme_commission:
        metrics_dict['enzyme_commission'] = compute_enzyme_commission_metrics(metrics_dict['enzyme_commission'])

    if configs.tasks.enzyme_reaction:
        metrics_dict['enzyme_reaction'] = compute_enzyme_reaction_metrics(metrics_dict['enzyme_reaction'])
    if configs.tasks.human_ppi:
        metrics_dict['human_ppi'] = compute_human_ppi_metrics(metrics_dict['human_ppi'])
    if configs.tasks.structure_similarity:
        metrics_dict['structure_similarity'] = compute_structure_similarity_metrics(
            metrics_dict['structure_similarity'])
    if configs.tasks.protein_protein_interface:
        metrics_dict['protein_protein_interface'] = compute_protein_protein_interface_metrics(
            metrics_dict['protein_protein_interface'])
    if configs.tasks.fluorescence:
        if metrics_dict['fluorescence']['correct_prediction'] > 0:
            metrics_dict['fluorescence'] = compute_fluorescence_metrics(metrics_dict['fluorescence'])
    if configs.tasks.stability:
        if metrics_dict['stability']['correct_prediction'] > 0:
            metrics_dict['stability'] = compute_stability_metrics(metrics_dict['stability'])
    if configs.tasks.protein_ligand_affinity:
        if metrics_dict['protein_ligand_affinity']['correct_prediction'] > 0:
            metrics_dict['protein_ligand_affinity'] = compute_protein_ligand_affinity_metrics(
                metrics_dict['protein_ligand_affinity'])
    return metrics_dict


def compute_ptm_metrics(ptm_dict):
    new_dict = {'accuracy': ptm_dict['accuracy'].compute().cpu().item(),
                'f1': ptm_dict['f1'].compute().cpu().item(),
                'length': ptm_dict['length'],
                'correct_prediction': ptm_dict['correct_prediction']}
    return new_dict


def compute_auxiliary_metrics(auxiliary_dict):
    new_dict = {}
    for name, metric_dict in auxiliary_dict.items():
        new_dict[name] = {}
        new_dict[name]['accuracy'] = auxiliary_dict[name]['accuracy'].compute().cpu().item()
        new_dict[name]['f1'] = auxiliary_dict[name]['f1'].compute().cpu().item()
        new_dict[name]['length'] = auxiliary_dict[name]['length']
    return new_dict


def compute_localization_metrics(localization_dict):
    new_dict = {
        'accuracy': localization_dict['accuracy'].compute().cpu().item(),
        'f1': localization_dict['f1'].compute().cpu().item(),
        'regression': localization_dict['regression'],
        'correct_prediction': localization_dict['correct_prediction']
    }
    return new_dict


def compute_localization_deeploc_metrics(localization_deeploc_dict):
    new_dict = {
        'accuracy': localization_deeploc_dict['accuracy'].compute().cpu().item(),
        'macro_f1': localization_deeploc_dict['macro_f1'].compute().cpu().item(),
        'f1': localization_deeploc_dict['f1'].compute().cpu(),
        'correct_prediction': localization_deeploc_dict['correct_prediction'],
        'wrong_token': localization_deeploc_dict['wrong_token'],
    }
    return new_dict


def compute_fold_metrics(fold_dict):
    new_dict = {
        'accuracy': fold_dict['accuracy'].compute().cpu().item(),
        'f1': fold_dict['f1'].compute().cpu().item(),
        'correct_prediction': fold_dict['correct_prediction']
    }
    return new_dict


def compute_amino_to_fold_seek_metrics(fold_dict):
    new_dict = {
        'accuracy': fold_dict['accuracy'].compute().cpu().item(),
        'precision': fold_dict['precision'].compute().cpu().item(),
        'recall': fold_dict['recall'].compute().cpu().item(),
        'f1': fold_dict['f1'].compute().cpu().item(),
        'correct_prediction': fold_dict['correct_prediction'],
        'wrong_length': fold_dict['wrong_length'],
        'wrong_token': fold_dict['wrong_token'],
    }
    return new_dict


def compute_sequence_prediction_metrics(metric_dict):
    new_dict = {
        'accuracy': metric_dict['accuracy'].compute().cpu().item(),
        'precision': metric_dict['precision'].compute().cpu().item(),
        'recall': metric_dict['recall'].compute().cpu().item(),
        'f1': metric_dict['f1'].compute().cpu().item(),
        'correct_prediction': metric_dict['correct_prediction'],
        'wrong_length': metric_dict['wrong_length'],
        'wrong_token': metric_dict['wrong_token'],
    }
    return new_dict


def compute_gene_ontology_metrics(gene_ontology_dict):
    new_dict = {
        'accuracy': gene_ontology_dict['accuracy'].compute().cpu().item(),
        'f1': gene_ontology_dict['f1'].compute().cpu().item(),
        'correct_prediction': gene_ontology_dict['correct_prediction'],
        'wrong_token': gene_ontology_dict['wrong_token'],
    }
    return new_dict


def compute_enzyme_commission_metrics(enzyme_commission_dict):
    new_dict = {
        'accuracy': enzyme_commission_dict['accuracy'].compute().cpu().item(),
        'f1': enzyme_commission_dict['f1'].compute().cpu().item(),
        'correct_prediction': enzyme_commission_dict['correct_prediction'],
        'wrong_token': enzyme_commission_dict['wrong_token'],
    }
    return new_dict


def compute_enzyme_reaction_metrics(enzyme_reaction_dict):
    new_dict = {
        'accuracy': enzyme_reaction_dict['accuracy'].compute().cpu().item(),
        'f1': enzyme_reaction_dict['f1'].compute().cpu().item(),
        'correct_prediction': enzyme_reaction_dict['correct_prediction']
    }
    return new_dict


def compute_human_ppi_metrics(human_ppi_dict):
    new_dict = {
        'accuracy': human_ppi_dict['accuracy'].compute().cpu().item(),
        'f1': human_ppi_dict['f1'].compute().cpu().item(),
        'correct_prediction': human_ppi_dict['correct_prediction']
    }
    return new_dict


def compute_structure_similarity_metrics(structure_similarity_dict):
    correct_pred = structure_similarity_dict['correct_prediction']
    new_dict = {
        'spearman': structure_similarity_dict['spearman'].compute().cpu().item() if correct_pred > 0 else 0.0,
        'mae': structure_similarity_dict['mae'].compute().cpu().item(),
        'correct_prediction': structure_similarity_dict['correct_prediction']
    }
    return new_dict


def compute_protein_protein_interface_metrics(protein_protein_interface_dict):
    correct_pred = protein_protein_interface_dict['correct_prediction']
    new_dict = {
        'accuracy': protein_protein_interface_dict['accuracy'].compute().cpu().item(),
        'f1': protein_protein_interface_dict['f1'].compute().cpu().item(),
        'auc': protein_protein_interface_dict['auc'].compute().cpu().item() if correct_pred > 0 else 0.0,
        'mae': protein_protein_interface_dict['mae'].compute().cpu().item(),
        'correct_prediction': protein_protein_interface_dict['correct_prediction'],
        'wrong_token': protein_protein_interface_dict['wrong_token'],
    }
    return new_dict


def compute_fluorescence_metrics(fluorescence_dict):
    correct_pred = fluorescence_dict['correct_prediction']
    new_dict = {
        'spearman': fluorescence_dict['spearman'].compute().cpu().item() if correct_pred > 0 else 0.0,
        'mae': fluorescence_dict['mae'].compute().cpu().item(),
        'correct_prediction': fluorescence_dict['correct_prediction']
    }
    return new_dict


def compute_stability_metrics(stability_dict):
    correct_pred = stability_dict['correct_prediction']
    new_dict = {
        'spearman': stability_dict['spearman'].compute().cpu().item() if correct_pred > 0 else 0.0,
        'mae': stability_dict['mae'].compute().cpu().item(),
        'correct_prediction': stability_dict['correct_prediction']
    }
    return new_dict


def compute_protein_ligand_affinity_metrics(protein_ligand_affinity_dict):
    new_dict = {
        'rmse': protein_ligand_affinity_dict['rmse'].compute().cpu().item(),
        'correct_prediction': protein_ligand_affinity_dict['correct_prediction']
    }
    return new_dict


def prepare_metrics_dict(accelerator):
    metrics_dict = {}
    ptms_list = ['phosphorylation']
    for ptm in ptms_list:
        metrics_dict[ptm] = {
            'accuracy': torchmetrics.Accuracy(task="binary").to(accelerator.device),
            'length': [],
            "f1": torchmetrics.F1Score(task="binary").to(accelerator.device),
            'correct_prediction': 0
        }

    metrics_dict['auxiliary'] = {}
    for amino_acid_name in "ACDEFGHIKLMNPQRSTVWY":
        metrics_dict['auxiliary'][f'{amino_acid_name.lower()}'] = {
            'accuracy': torchmetrics.Accuracy(task="binary").to(accelerator.device),
            'length': [],
            "f1": torchmetrics.F1Score(task="binary").to(accelerator.device)
        }

    metrics_dict['localization'] = {
        'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=5).to(accelerator.device),
        'f1': torchmetrics.F1Score(task="multiclass", num_classes=5, average='macro').to(accelerator.device),
        'regression': [],
        'correct_prediction': 0
    }

    metrics_dict['localization_deeploc'] = {
        'accuracy': torchmetrics.Accuracy(task="multilabel", num_labels=10).to(accelerator.device),
        'macro_f1': torchmetrics.F1Score(task="multilabel", num_labels=10, average='macro').to(accelerator.device),
        'f1': torchmetrics.F1Score(task="multilabel", num_labels=10, average=None).to(accelerator.device),
        'correct_prediction': 0,
        'wrong_token': 0,
    }

    metrics_dict['fold'] = {
        'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=1195).to(accelerator.device),
        'f1': torchmetrics.F1Score(task="multiclass", num_classes=1195, average='macro').to(accelerator.device),
        'correct_prediction': 0
    }
    metrics_dict['enzyme_reaction'] = {
        'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=384).to(accelerator.device),
        'f1': torchmetrics.F1Score(task="multiclass", num_classes=384, average='macro').to(accelerator.device),
        'correct_prediction': 0
    }

    metrics_dict['human_ppi'] = {
        'accuracy': torchmetrics.Accuracy(task="binary").to(accelerator.device),
        'f1': torchmetrics.F1Score(task="binary").to(accelerator.device),
        'correct_prediction': 0
    }
    metrics_dict['structure_similarity'] = {
        'spearman': torchmetrics.SpearmanCorrCoef(num_outputs=1).to(accelerator.device),
        'mae': torchmetrics.MeanAbsoluteError().to(accelerator.device),
        'correct_prediction': 0
    }
    metrics_dict['protein_protein_interface'] = {
        'accuracy': torchmetrics.Accuracy(task="binary").to(accelerator.device),
        'f1': torchmetrics.F1Score(task="binary").to(accelerator.device),
        'auc': torchmetrics.AUROC(task="binary").to(accelerator.device),
        'mae': torchmetrics.MeanAbsoluteError().to(accelerator.device),
        'wrong_token': 0,
        'correct_prediction': 0
    }
    metrics_dict['fluorescence'] = {
        'spearman': torchmetrics.SpearmanCorrCoef(num_outputs=1).to(accelerator.device),
        'mae': torchmetrics.MeanAbsoluteError().to(accelerator.device),
        'correct_prediction': 0
    }
    metrics_dict['stability'] = {
        'spearman': torchmetrics.SpearmanCorrCoef(num_outputs=1).to(accelerator.device),
        'mae': torchmetrics.MeanAbsoluteError().to(accelerator.device),
        'correct_prediction': 0
    }
    metrics_dict['protein_ligand_affinity'] = {
        'rmse': torchmetrics.MeanSquaredError(squared=False).to(accelerator.device),
        'correct_prediction': 0
    }

    metrics_dict['amino_to_fold_seek'] = {
        'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=20).to(accelerator.device),
        'recall': torchmetrics.Recall(task="multiclass", num_classes=20, average='macro').to(accelerator.device),
        'precision': torchmetrics.Precision(task="multiclass", num_classes=20, average='macro').to(accelerator.device),
        'f1': torchmetrics.F1Score(task="multiclass", num_classes=20, average='macro').to(accelerator.device),
        'correct_prediction': 0,
        'wrong_length': 0,
        'wrong_token': 0,
    }

    metrics_dict['secondary_structure'] = {
        'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=3).to(accelerator.device),
        'recall': torchmetrics.Recall(task="multiclass", num_classes=3, average='macro').to(accelerator.device),
        'precision': torchmetrics.Precision(task="multiclass", num_classes=3, average='macro').to(accelerator.device),
        'f1': torchmetrics.F1Score(task="multiclass", num_classes=3, average='macro').to(accelerator.device),
        'correct_prediction': 0,
        'wrong_length': 0,
        'wrong_token': 0,
    }

    metrics_dict['gene_ontology_bp'] = {
        'accuracy': torchmetrics.Accuracy(task="multilabel", num_labels=1940).to(accelerator.device),
        'f1': torchmetrics.F1Score(task="multilabel", num_labels=1940, average='macro').to(accelerator.device),
        'correct_prediction': 0,
        'wrong_token': 0,
    }

    metrics_dict['gene_ontology_mf'] = {
        'accuracy': torchmetrics.Accuracy(task="multilabel", num_labels=488).to(accelerator.device),
        'f1': torchmetrics.F1Score(task="multilabel", num_labels=488, average='macro').to(accelerator.device),
        'correct_prediction': 0,
        'wrong_token': 0,
    }

    metrics_dict['gene_ontology_cc'] = {
        'accuracy': torchmetrics.Accuracy(task="multilabel", num_labels=320).to(accelerator.device),
        'f1': torchmetrics.F1Score(task="multilabel", num_labels=320, average='macro').to(accelerator.device),
        'correct_prediction': 0,
        'wrong_token': 0,
    }

    metrics_dict['enzyme_commission'] = {
        'accuracy': torchmetrics.Accuracy(task="multilabel", num_labels=538).to(accelerator.device),
        'f1': torchmetrics.F1Score(task="multilabel", num_labels=538, average='macro').to(accelerator.device),
        'correct_prediction': 0,
        'wrong_token': 0,
    }

    return metrics_dict


def find_sites(seq):
    try:
        # Find the index of the first occurrence of 2
        idx = seq.tolist().index(2)
    except ValueError:
        idx = 1
    return seq.tolist()[:idx]


def split_list_and_remove_duplicates_corrected(input_list, sep_index):
    """
    Splits a list into two parts based on the position of the first occurrence of the number 3,
    and removes duplicates from the left part that also appear in the right part.

    Parameters:
    - input_list (list): The list to be split.

    Returns:
    - tuple: A tuple containing two lists (list, list).

    Examples:
    - split_list_and_remove_duplicates_corrected([1, 4, 5, 3, 7, 4, 6, 9]) returns ([1, 5], [7, 4, 6, 9]).
    - split_list_and_remove_duplicates_corrected([1, 4, 3, 7, 3, 8, 9]) returns ([1], [7, 8, 9]).
    """
    try:
        # Find the index of the first occurrence of 3
        index = input_list.index(sep_index)
    except ValueError:
        # If 3 is not in the list, return the whole list as the left part and an empty list as the right part
        return input_list, []

    # Split the list at the index of 3
    first_part = input_list[:index]
    second_part = [x for x in input_list[index + 1:] if x != 3]

    # Remove elements from the first part that are also in the second part
    first_part = [x for x in first_part if x not in second_part]

    return first_part, second_part


def ptms_metric(pred_seq, trues, task_name, ptm, tokenizer):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    negative_pred, positive_pred = split_list_and_remove_duplicates_corrected(prediction_list,
                                                                              tokenizer.tokens_dict['<sep>'])
    negative_true, positive_true = split_list_and_remove_duplicates_corrected(true_list,
                                                                              tokenizer.tokens_dict['<sep>'])

    negative_pred, positive_pred = sorted(list(set(negative_pred))), sorted(list(set(positive_pred)))
    negative_true, positive_true = sorted(list(set(negative_true))), sorted(list(set(positive_true)))

    neg_true = torch.zeros_like(torch.tensor(negative_true))
    neg_pred = [0 if true in negative_pred else 1 for true in negative_true]

    pos_true = torch.ones_like(torch.tensor(positive_true))
    pos_pred = [1 if true in positive_pred else 0 for true in positive_true]

    # precision, recall, f1_score, accuracy = calculate_metrics(true_list, prediction_list)

    if len(neg_true) != 0:
        ptm['accuracy'].update(torch.tensor(neg_pred), neg_true)
        ptm['f1'].update(torch.tensor(neg_pred), neg_true)

    if len(pos_true) != 0:
        ptm['accuracy'].update(torch.tensor(pos_pred), pos_true)
        ptm['f1'].update(torch.tensor(pos_pred), pos_true)

    if len(pos_true) != 0 or len(neg_true) != 0:
        ptm['correct_prediction'] += 1

    ptm['length'].append(abs(len(prediction_list) - len(true_list)))

    return ptm


def auxiliary_metric(pred_seq, trues, tokenizer, auxiliary, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    pos_true = torch.ones_like(torch.tensor(true_list))
    pos_pred = [1 if true in prediction_list else 0 for true in true_list]

    if len(pos_true) != 0:
        auxiliary['accuracy'].update(torch.tensor(pos_pred), pos_true)
        auxiliary['f1'].update(torch.tensor(pos_pred), pos_true)

    auxiliary['length'].append(abs(len(prediction_list) - len(true_list)))

    return auxiliary


def localization_metric(pred_seq, trues, tokenizer, localization, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    label_to_index = {"other": 0, "sp": 1, "mt": 2, "ch": 3, "th": 4}
    try:
        pred_label = prediction_list[0]
        true_label = true_list[0]

        # if len(prediction_list) > 0:
        if tokenizer.index_token_dict[pred_label] in list(label_to_index.keys()):
            pred = torch.tensor([label_to_index[tokenizer.index_token_dict[pred_label]]]).to(accelerator.device)
            target = torch.tensor([label_to_index[tokenizer.index_token_dict[true_label]]]).to(accelerator.device)
            localization['accuracy'].update(pred, target)
            localization['f1'].update(pred, target)

        localization['correct_prediction'] += 1
        if true_list[0] != tokenizer.tokens_dict['other']:
            try:
                if prediction_list[0] != tokenizer.tokens_dict['other']:
                    regression = abs(int(tokenizer.index_token_dict[prediction_list[1]]) - int(
                        tokenizer.index_token_dict[true_list[1]]))
                else:
                    regression = int(tokenizer.index_token_dict[true_list[1]])
            except (IndexError, ValueError):
                regression = int(tokenizer.index_token_dict[true_list[1]])

            localization['regression'].append(regression)

    except IndexError:
        pass

    return localization


def localization_deeploc_metric(pred_seq, trues, tokenizer, localization_deeploc, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    label_to_index = tokenizer.localization_deeploc_label_index_mapping

    try:
        if all(element in tokenizer.index_token_dict for element in prediction_list):
            if len(prediction_list) == 0:
                pred = [torch.zeros((1, len(label_to_index)))]
            else:
                pred = [F.one_hot(torch.tensor([label_to_index[tokenizer.index_token_dict[element]]]),
                                  num_classes=len(label_to_index)) for element in prediction_list]

            if len(true_list) == 0:
                true = [torch.zeros((1, len(label_to_index)))]
            else:
                true = [F.one_hot(torch.tensor([label_to_index[tokenizer.index_token_dict[element]]]),
                                  num_classes=len(label_to_index)) for element in true_list]

            pred = torch.sum(torch.stack(pred), dim=0).to(accelerator.device)
            target = torch.sum(torch.stack(true), dim=0).to(accelerator.device)

            # pred = torch.tensor([pred]).to(accelerator.device)
            # target = torch.tensor([true]).to(accelerator.device)
            pred[pred != 0] = 1.0
            pred = pred.to(dtype=torch.float32)
            target = target.to(dtype=torch.int64)

            localization_deeploc['accuracy'].update(pred, target)
            localization_deeploc['macro_f1'].update(pred, target)
            localization_deeploc['f1'].update(pred, target)

            localization_deeploc['correct_prediction'] += 1

    except (IndexError, KeyError):
        localization_deeploc['wrong_token'] += 1

    return localization_deeploc


def enzyme_reaction_metric(pred_seq, trues, tokenizer, enzyme_reaction, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    label_to_index = tokenizer.er_label_index_mapping

    try:
        pred_label = tokenizer.index_token_dict[prediction_list[0]]
        true_label = tokenizer.index_token_dict[true_list[0]]

        if pred_label in list(label_to_index.keys()):
            pred = torch.tensor([label_to_index[pred_label]]).to(accelerator.device)
            target = torch.tensor([label_to_index[true_label]]).to(accelerator.device)
            enzyme_reaction['accuracy'].update(pred, target)
            enzyme_reaction['f1'].update(pred, target)

            enzyme_reaction['correct_prediction'] += 1

    except IndexError:
        pass

    return enzyme_reaction


def fold_metric(pred_seq, trues, tokenizer, fold, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    label_to_index = tokenizer.fold_label_index_mapping

    try:
        pred_label = prediction_list[0]
        true_label = true_list[0]

        if tokenizer.index_token_dict[pred_label] in list(label_to_index.keys()):
            pred = torch.tensor([label_to_index[tokenizer.index_token_dict[pred_label]]]).to(accelerator.device)
            target = torch.tensor([label_to_index[tokenizer.index_token_dict[true_label]]]).to(accelerator.device)
            fold['accuracy'].update(pred, target)
            fold['f1'].update(pred, target)

            fold['correct_prediction'] += 1

    except IndexError:
        pass

    return fold


def gene_ontology_metric(pred_seq, trues, tokenizer, gene_ontology, accelerator, mode):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    label_to_index = tokenizer.gene_ontology_label_index_mapping[mode]

    try:
        if all(element in tokenizer.index_token_dict for element in prediction_list):
            if len(prediction_list) == 0:
                pred = [torch.zeros((1, len(label_to_index)))]
            else:
                pred = [F.one_hot(torch.tensor([label_to_index[tokenizer.index_token_dict[element]]]),
                                  num_classes=len(label_to_index)) for element in prediction_list]

            if len(true_list) == 0:
                true = [torch.zeros((1, len(label_to_index)))]
            else:
                true = [F.one_hot(torch.tensor([label_to_index[tokenizer.index_token_dict[element]]]),
                                  num_classes=len(label_to_index)) for element in true_list]

            pred = torch.sum(torch.stack(pred), dim=0).to(accelerator.device)
            target = torch.sum(torch.stack(true), dim=0).to(accelerator.device)

            # pred = torch.tensor([pred]).to(accelerator.device)
            # target = torch.tensor([true]).to(accelerator.device)
            pred[pred != 0] = 1.0
            pred = pred.to(dtype=torch.float32)
            target = target.to(dtype=torch.int64)

            gene_ontology['accuracy'].update(pred, target)
            gene_ontology['f1'].update(pred, target)

            gene_ontology['correct_prediction'] += 1

    except (IndexError, KeyError):
        gene_ontology['wrong_token'] += 1
    # except RuntimeError:
    #     print('error')

    return gene_ontology


def reassemble_labels(label_parts):
    """
    Reassemble split label parts into their original EC label format.

    Parameters:
    label_parts (list): A list of strings representing parts of EC labels.

    Returns:
    list: A list of reconstructed EC labels.
    """
    return ['.'.join(label_parts[i:i + 4]) for i in range(0, len(label_parts), 4)]


def enzyme_commission_metric(pred_seq, trues, tokenizer, enzyme_commission, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    label_to_index = tokenizer.ec_label_index_mapping

    try:
        if len(prediction_list) % 4 == 0:
            prediction_list = reassemble_labels([tokenizer.index_token_dict[i] for i in prediction_list])
            true_list = reassemble_labels([tokenizer.index_token_dict[i] for i in true_list])
        else:
            return enzyme_commission

        if all(element in list(label_to_index.keys()) for element in prediction_list):
            if len(prediction_list) == 0:
                pred = [torch.zeros((1, len(label_to_index)))]
            else:
                pred = [F.one_hot(torch.tensor([label_to_index[element]]),
                                  num_classes=len(label_to_index)) for element in prediction_list]

            if len(true_list) == 0:
                true = [torch.zeros((1, len(label_to_index)))]
            else:
                true = [F.one_hot(torch.tensor([label_to_index[element]]),
                                  num_classes=len(label_to_index)) for element in true_list]

            pred = torch.sum(torch.stack(pred), dim=0).to(accelerator.device)
            target = torch.sum(torch.stack(true), dim=0).to(accelerator.device)

            # pred = torch.tensor([pred]).to(accelerator.device)
            # target = torch.tensor([true]).to(accelerator.device)
            pred[pred != 0] = 1.0
            pred = pred.to(dtype=torch.float32)
            target = target.to(dtype=torch.int64)

            enzyme_commission['accuracy'].update(pred, target)
            enzyme_commission['f1'].update(pred, target)

            enzyme_commission['correct_prediction'] += 1

    except (IndexError, KeyError):
        enzyme_commission['wrong_token'] += 1
    # except RuntimeError:
    #     print('error')

    return enzyme_commission


def amino_to_fold_seek_metric(pred_seq, trues, tokenizer, amino_to_fold_seek, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    if len(prediction_list) != len(true_list):
        if len(prediction_list) > len(true_list):
            prediction_list = prediction_list[:len(true_list)]
        else:
            amino_to_fold_seek['wrong_length'] += 1
            return amino_to_fold_seek

    label_to_index = tokenizer.amino_to_fold_seek_label_index_mapping

    try:
        if all(element in tokenizer.index_token_dict for element in prediction_list):
            pred = [label_to_index[tokenizer.index_token_dict[element]] for element in prediction_list]
            true = [label_to_index[tokenizer.index_token_dict[element]] for element in true_list]

            pred = torch.tensor([pred]).to(accelerator.device)
            target = torch.tensor([true]).to(accelerator.device)

            amino_to_fold_seek['accuracy'].update(pred, target)
            amino_to_fold_seek['precision'].update(pred, target)
            amino_to_fold_seek['recall'].update(pred, target)
            amino_to_fold_seek['f1'].update(pred, target)

            amino_to_fold_seek['correct_prediction'] += 1

    except (IndexError, KeyError):
        amino_to_fold_seek['wrong_token'] += 1
        pass

    return amino_to_fold_seek


def fold_seek_to_amino_metric(pred_seq, trues, tokenizer, fold_seek_to_amino, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    if len(prediction_list) != len(true_list):
        if len(prediction_list) > len(true_list):
            prediction_list = prediction_list[:len(true_list)]
        else:
            fold_seek_to_amino['wrong_length'] += 1
            return fold_seek_to_amino

    label_to_index = tokenizer.fold_seek_to_amino_label_index_mapping

    try:

        if all(element in tokenizer.index_token_dict for element in prediction_list):
            pred = [label_to_index[tokenizer.index_token_dict[element]] for element in prediction_list]
            true = [label_to_index[tokenizer.index_token_dict[element]] for element in true_list]

            pred = torch.tensor([pred]).to(accelerator.device)
            target = torch.tensor([true]).to(accelerator.device)

            fold_seek_to_amino['accuracy'].update(pred, target)
            fold_seek_to_amino['precision'].update(pred, target)
            fold_seek_to_amino['recall'].update(pred, target)
            fold_seek_to_amino['f1'].update(pred, target)

            fold_seek_to_amino['correct_prediction'] += 1

    except (IndexError, KeyError):
        fold_seek_to_amino['wrong_token'] += 1
        pass

    return fold_seek_to_amino


def secondary_structure_metric(pred_seq, trues, tokenizer, secondary_structure, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    if len(prediction_list) != len(true_list):
        if len(prediction_list) > len(true_list):
            prediction_list = prediction_list[:len(true_list)]
        else:
            secondary_structure['wrong_length'] += 1
            return secondary_structure

    label_to_index = tokenizer.secondary_structure_label_index_mapping

    try:

        if all(element in tokenizer.index_token_dict for element in prediction_list):
            pred = [label_to_index[tokenizer.index_token_dict[element]] for element in prediction_list]
            true = [label_to_index[tokenizer.index_token_dict[element]] for element in true_list]

            pred = torch.tensor([pred]).to(accelerator.device)
            target = torch.tensor([true]).to(accelerator.device)

            secondary_structure['accuracy'].update(pred, target)
            secondary_structure['precision'].update(pred, target)
            secondary_structure['recall'].update(pred, target)
            secondary_structure['f1'].update(pred, target)

            secondary_structure['correct_prediction'] += 1

    except (IndexError, KeyError):
        secondary_structure['wrong_token'] += 1
        pass

    return secondary_structure


def sequence_prediction_metric(pred_seq, trues, tokenizer, metric_dict, label_index_name, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    if len(prediction_list) != len(true_list):
        if len(prediction_list) > len(true_list):
            prediction_list = prediction_list[:len(true_list)]
        else:
            metric_dict['wrong_length'] += 1
            return metric_dict

    # label_to_index = tokenizer.secondary_structure_label_index_mapping
    # select the right label index mapping from the tokenizer based on the label_index_name
    label_to_index = getattr(tokenizer, label_index_name)

    try:

        if all(element in tokenizer.index_token_dict for element in prediction_list):
            pred = [label_to_index[tokenizer.index_token_dict[element]] for element in prediction_list]
            true = [label_to_index[tokenizer.index_token_dict[element]] for element in true_list]

            pred = torch.tensor([pred]).to(accelerator.device)
            target = torch.tensor([true]).to(accelerator.device)

            metric_dict['accuracy'].update(pred, target)
            metric_dict['precision'].update(pred, target)
            metric_dict['recall'].update(pred, target)
            metric_dict['f1'].update(pred, target)

            metric_dict['correct_prediction'] += 1

    except (IndexError, KeyError):
        metric_dict['wrong_token'] += 1
        pass

    return metric_dict


def human_ppi_metric(pred_seq, trues, tokenizer, label_to_index, human_ppi, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    # label_to_index = tokenizer.human_ppi_label_index_mapping

    try:
        pred_label = prediction_list[0]
        true_label = true_list[0]

        if tokenizer.index_token_dict[pred_label] in list(label_to_index.keys()):
            pred = torch.tensor([label_to_index[tokenizer.index_token_dict[pred_label]]]).to(accelerator.device)
            target = torch.tensor([label_to_index[tokenizer.index_token_dict[true_label]]]).to(accelerator.device)
            human_ppi['accuracy'].update(pred, target)
            human_ppi['f1'].update(pred, target)

            human_ppi['correct_prediction'] += 1

    except IndexError:
        pass

    return human_ppi


def ppi_metric(pred_seq, trues, tokenizer, ppi, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    label_to_index = tokenizer.ppi_label

    try:
        for token in prediction_list:
            if not tokenizer.index_token_dict[token] in list(label_to_index):
                break
        else:
            true_label = reconstruct_float([tokenizer.index_token_dict[token] for token in true_list])
            pred_label = reconstruct_float([tokenizer.index_token_dict[token] for token in prediction_list])

            pred = torch.tensor([pred_label]).to(accelerator.device)
            target = torch.tensor([true_label]).to(accelerator.device)
            ppi['spearman'].update(pred, target)
            ppi['mae'].update(pred, target)

            ppi['correct_prediction'] += 1

    except (IndexError, ValueError):
        pass

    return ppi


def structure_similarity_metric(pred_seq, trues, tokenizer, structure_similarity, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    label_to_index = tokenizer.structure_similarity_label

    try:
        for token in prediction_list:
            if not tokenizer.index_token_dict[token] in list(label_to_index):
                break
        else:
            true_label = reconstruct_float([tokenizer.index_token_dict[token] for token in true_list])
            pred_label = reconstruct_float([tokenizer.index_token_dict[token] for token in prediction_list])

            pred = torch.tensor([pred_label]).to(accelerator.device)
            target = torch.tensor([true_label]).to(accelerator.device)
            structure_similarity['spearman'].update(pred, target)
            structure_similarity['mae'].update(pred, target)

            structure_similarity['correct_prediction'] += 1

    except (IndexError, ValueError):
        pass

    return structure_similarity


def string_list_to_tuple_list(string_list):
    """
    Convert a list of strings, where numbers are separated by commas,
    into a list of tuples.

    Args:
    string_list (list of str): A list of strings representing numbers and
    commas.

    Returns:
    list of tuples: A list where each pair of numbers (as strings) before a
    comma is converted to a tuple of integers.
    """
    # Split the list by commas and convert to tuples
    return [tuple(map(int, string_list[i:i + 2])) for i in range(0, len(string_list), 3)]


def split_on_sep(input_list, sep_token):
    if sep_token in input_list:
        sep_index = input_list.index(sep_token)
        before_sep = input_list[:sep_index]
        after_sep = [item for item in input_list[sep_index + 1:] if item != sep_token]  # Remove additional <sep> tokens
    else:
        before_sep = []
        after_sep = input_list[:]

    return before_sep, after_sep


def protein_protein_interface_metric(pred_seq, trues, tokenizer, protein_protein_interface, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    true_list = [tokenizer.index_token_dict[token] for token in true_list]
    try:
        prediction_list = [tokenizer.index_token_dict[token] for token in prediction_list]
    except KeyError:
        protein_protein_interface['wrong_token'] += 1
        return protein_protein_interface

    try:
        tail_pred, preds = split_on_sep(prediction_list, '<sep>')

        # check if the tail is a pair
        if len(tail_pred) != 2:
            return protein_protein_interface
        elif len(preds) % 2 != 0:
            return protein_protein_interface
        else:
            pred_max_seq_1, pred_max_seq_2 = tail_pred
            pred_max_seq_1, pred_max_seq_2 = int(pred_max_seq_1), int(pred_max_seq_2)
            # pred_tuple_list = string_list_to_tuple_list(preds)
            pred_tuple_list = list(zip(preds[::2], preds[1::2]))

        labels = tokenizer.protein_protein_interface_label_list

        tail_true, true_list = split_on_sep(true_list, '<sep>')

        # extract the max sequence length of both sequences from the true list
        max_seq_1, max_seq_2 = int(tail_true[0]), int(tail_true[1])

        if not (all(int(value[0]) < max_seq_1 and int(value[1]) < max_seq_2 for value in pred_tuple_list)):
            return protein_protein_interface

        # true_tuple_list = string_list_to_tuple_list(true_list[:-3])
        true_tuple_list = list(zip(true_list[::2], true_list[1::2]))

        pred = torch.zeros((max_seq_1, max_seq_2)).to(accelerator.device)
        target = torch.zeros((max_seq_1, max_seq_2)).to(accelerator.device)

        for pair in true_tuple_list:
            target[int(pair[0]) - 1, int(pair[1]) - 1] = 1

        for pair in pred_tuple_list:
            if len(pair) != 2:
                return protein_protein_interface

            if str(pair[0]) in labels and str(pair[1]) in labels:
                pred[int(pair[0]) - 1, int(pair[1]) - 1] = 1
            else:
                return protein_protein_interface

        pred = pred.flatten()
        target = target.flatten()

        if str(pred_max_seq_1) in labels and str(pred_max_seq_2) in labels:
            regression_pred = torch.tensor([pred_max_seq_1]).to(accelerator.device)
            regression_target = torch.tensor([max_seq_1]).to(accelerator.device)

            protein_protein_interface['mae'].update(regression_pred, regression_target)

            regression_pred = torch.tensor([pred_max_seq_2]).to(accelerator.device)
            regression_target = torch.tensor([max_seq_2]).to(accelerator.device)

            protein_protein_interface['mae'].update(regression_pred, regression_target)

            protein_protein_interface['accuracy'].update(pred, target)
            protein_protein_interface['f1'].update(pred, target)
            protein_protein_interface['auc'].update(pred, target)

            protein_protein_interface['correct_prediction'] += 1

    except (IndexError, ValueError):
        pass

    return protein_protein_interface


def reconstruct_float(char_list):
    """ Reconstructs a float number from a list of its individual characters. """
    return float(''.join(char_list))


def fluorescence_metric(pred_seq, trues, tokenizer, fluorescence, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    label_to_index = tokenizer.fluorescence_label

    try:
        for token in prediction_list:
            if not tokenizer.index_token_dict[token] in list(label_to_index):
                break
        else:
            true_label = reconstruct_float([tokenizer.index_token_dict[token] for token in true_list])
            pred_label = reconstruct_float([tokenizer.index_token_dict[token] for token in prediction_list])

            pred = torch.tensor([pred_label]).to(accelerator.device)
            target = torch.tensor([true_label]).to(accelerator.device)
            fluorescence['spearman'].update(pred, target)
            fluorescence['mae'].update(pred, target)

            fluorescence['correct_prediction'] += 1

    except (IndexError, ValueError):
        pass

    return fluorescence


def stability_metric(pred_seq, trues, tokenizer, stability, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    label_to_index = tokenizer.stability_label

    try:
        for token in prediction_list:
            if not tokenizer.index_token_dict[token] in list(label_to_index):
                break
        else:
            true_label = reconstruct_float([tokenizer.index_token_dict[token] for token in true_list])
            pred_label = reconstruct_float([tokenizer.index_token_dict[token] for token in prediction_list])

            pred = torch.tensor([pred_label]).to(accelerator.device)
            target = torch.tensor([true_label]).to(accelerator.device)
            stability['spearman'].update(pred, target)
            stability['mae'].update(pred, target)

            stability['correct_prediction'] += 1

    except (IndexError, ValueError):
        pass

    return stability


def protein_ligand_affinity_metric(pred_seq, trues, tokenizer, structure_similarity, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    label_to_index = tokenizer.protein_ligand_affinity_label

    try:
        for token in prediction_list:
            if not tokenizer.index_token_dict[token] in list(label_to_index):
                break
        else:
            true_label = reconstruct_float([tokenizer.index_token_dict[token] for token in true_list])
            pred_label = reconstruct_float([tokenizer.index_token_dict[token] for token in prediction_list])

            true_label = unmap_labels(true_label)
            pred_label = unmap_labels(pred_label)

            pred = torch.tensor([pred_label]).to(accelerator.device)
            target = torch.tensor([true_label]).to(accelerator.device)
            structure_similarity['rmse'].update(pred, target)

            structure_similarity['correct_prediction'] += 1

    except (IndexError, ValueError):
        pass

    return structure_similarity


def solubility_metric(pred_seq, trues, tokenizer, solubility, accelerator):
    prediction_list = find_sites(pred_seq)[1:]
    true_list = find_sites(trues)[1:]

    label_to_index = tokenizer.solubility_label_index_mapping

    try:
        pred_label = prediction_list[0]
        true_label = true_list[0]

        if tokenizer.index_token_dict[pred_label] in list(label_to_index.keys()):
            pred = torch.tensor([label_to_index[tokenizer.index_token_dict[pred_label]]]).to(accelerator.device)
            target = torch.tensor([label_to_index[tokenizer.index_token_dict[true_label]]]).to(accelerator.device)
            solubility['accuracy'].update(pred, target)
            solubility['f1'].update(pred, target)

            solubility['correct_prediction'] += 1

    except IndexError:
        pass

    return solubility


def merge_items(items_list):
    """
    Merge list of strings into a single string if there are two or more items,
    or return the string representation of the single item if there is only one item.

    Parameters:
    items_list (list of str): List of strings to be merged.

    Returns:
    str: Merged string or string representation of the single item.
    """
    if len(items_list) >= 2:
        return '_'.join(items_list)
    elif len(items_list) == 1:
        return items_list[0]
    else:
        return ''


def compute_metrics(predictions, trues, tokenizer, metrics_dict, accelerator, mode):
    if mode != 'test':
        predictions = predictions.float().argmax(dim=-1).cpu().numpy()
    trues = trues.cpu().numpy()

    # Get the indices where '2' appears
    indices_of_2 = np.argwhere(predictions == 2)

    # Find the minimum index of '2' for each row, with default value as array's length
    first_2_indices = np.array([indices_of_2[indices_of_2[:, 0] == i][:, 1].min(initial=predictions.shape[1]) for i in
                                range(predictions.shape[0])])

    # Create a mask where each element is True if its index is greater than the first index of '2' in its row
    mask = np.arange(predictions.shape[1]) > first_2_indices[:, None]

    # Set elements to zero where mask is True
    predictions[mask] = 0
    for sample_number, pred_seq in enumerate(predictions):
        task_token_name = tokenizer.index_token_dict[find_sites(trues[sample_number])[0]]
        task_name = merge_items(task_token_name[1:-1].split('_')[1:])
        if task_name in list(ptms_list):
            metrics_dict[task_name] = ptms_metric(pred_seq, trues[sample_number], task_name, metrics_dict[task_name],
                                                  tokenizer)
        elif task_name == 'localization':
            metrics_dict['localization'] = localization_metric(pred_seq, trues[sample_number], tokenizer,
                                                               metrics_dict['localization'], accelerator)
        elif task_name == 'localization_deeploc':
            metrics_dict['localization_deeploc'] = localization_deeploc_metric(pred_seq, trues[sample_number],
                                                                               tokenizer,
                                                                               metrics_dict['localization_deeploc'],
                                                                               accelerator)
        elif task_name == 'fold':
            metrics_dict['fold'] = fold_metric(pred_seq, trues[sample_number], tokenizer, metrics_dict['fold'],
                                               accelerator)
        elif task_name == 'enzyme_reaction':
            metrics_dict['enzyme_reaction'] = enzyme_reaction_metric(pred_seq, trues[sample_number], tokenizer,
                                                                     metrics_dict['enzyme_reaction'], accelerator)
        elif task_name == 'human_ppi':
            metrics_dict['human_ppi'] = human_ppi_metric(pred_seq, trues[sample_number], tokenizer,
                                                         tokenizer.human_ppi_label_index_mapping,
                                                         metrics_dict['human_ppi'], accelerator)

        elif task_name == 'structure_similarity':
            metrics_dict['structure_similarity'] = structure_similarity_metric(
                pred_seq, trues[sample_number], tokenizer,
                metrics_dict['structure_similarity'], accelerator
            )
        elif task_name == 'protein_protein_interface':
            metrics_dict['protein_protein_interface'] = protein_protein_interface_metric(
                pred_seq, trues[sample_number],
                tokenizer, metrics_dict['protein_protein_interface'], accelerator
            )
        elif task_name == 'fluorescence':
            metrics_dict['fluorescence'] = fluorescence_metric(pred_seq, trues[sample_number], tokenizer,
                                                               metrics_dict['fluorescence'], accelerator)
        elif task_name == 'stability':
            metrics_dict['stability'] = stability_metric(pred_seq, trues[sample_number], tokenizer,
                                                         metrics_dict['stability'], accelerator)
        elif task_name == 'protein_ligand_affinity':
            metrics_dict['protein_ligand_affinity'] = protein_ligand_affinity_metric(
                pred_seq,
                trues[sample_number],
                tokenizer,
                metrics_dict[
                    'protein_ligand_affinity'],
                accelerator
            )

        elif task_name == 'amino_to_fold_seek':
            metrics_dict['amino_to_fold_seek'] = amino_to_fold_seek_metric(pred_seq, trues[sample_number], tokenizer,
                                                                           metrics_dict['amino_to_fold_seek'],
                                                                           accelerator)

        elif task_name == 'secondary_structure':
            metrics_dict['secondary_structure'] = sequence_prediction_metric(
                pred_seq, trues[sample_number],
                tokenizer,
                metrics_dict['secondary_structure'],
                'secondary_structure_label_index_mapping',
                accelerator)

        elif task_name == 'gene_ontology_bp':
            metrics_dict['gene_ontology_bp'] = gene_ontology_metric(pred_seq, trues[sample_number],
                                                                    tokenizer,
                                                                    metrics_dict['gene_ontology_bp'],
                                                                    accelerator, mode='bp')

        elif task_name == 'gene_ontology_mf':
            metrics_dict['gene_ontology_mf'] = gene_ontology_metric(pred_seq, trues[sample_number],
                                                                    tokenizer,
                                                                    metrics_dict['gene_ontology_mf'],
                                                                    accelerator, mode='mf')
        elif task_name == 'gene_ontology_cc':
            metrics_dict['gene_ontology_cc'] = gene_ontology_metric(pred_seq, trues[sample_number],
                                                                    tokenizer,
                                                                    metrics_dict['gene_ontology_cc'],
                                                                    accelerator, mode='cc')
        elif task_name == 'enzyme_commission':
            metrics_dict['enzyme_commission'] = enzyme_commission_metric(pred_seq, trues[sample_number], tokenizer,
                                                                         metrics_dict['enzyme_commission'], accelerator)
        elif task_name in list("ACDEFGHIKLMNPQRSTVWY".lower()):
            metrics_dict['auxiliary'][task_name] = auxiliary_metric(
                pred_seq, trues[sample_number], tokenizer,
                metrics_dict['auxiliary'][task_name], accelerator)
    return metrics_dict
