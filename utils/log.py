import numpy as np
from metrics import compute_all_metrics

ptms_list = ['phosphorylation']


def training_tensorboard_log(epoch, train_loss, metrics_dict, tools, configs):
    if tools['tensorboard_log']:
        if epoch > configs.train_settings.start_metric_epoch:
            metrics_dict = compute_all_metrics(metrics_dict, configs)
            for name in ptms_list:
                if len(metrics_dict[name]['length']) > 0:
                    tools['train_writer'].add_scalar(f'{name}_accuracy',
                                                     np.round(metrics_dict[name]['accuracy'], 4),
                                                     tools['epoch'])
                    tools['train_writer'].add_scalar(f'{name}_f1',
                                                     np.round(metrics_dict[name]['f1'], 4),
                                                     tools['epoch'])
            if configs.tasks.auxiliary:
                for name, auxiliary_metric_dict in metrics_dict['auxiliary'].items():
                    if len(auxiliary_metric_dict['length']) > 0:
                        tools['train_writer'].add_scalar(f'{name}_accuracy',
                                                         np.round(auxiliary_metric_dict['accuracy'], 4),
                                                         tools['epoch'])
                        tools['train_writer'].add_scalar(f'{name}_f1',
                                                         np.round(auxiliary_metric_dict['f1'], 4),
                                                         tools['epoch'])

            if configs.tasks.localization:
                tools['train_writer'].add_scalar(f'localization_accuracy',
                                                 np.round(metrics_dict['localization']['accuracy'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'localization_f1', np.round(metrics_dict['localization']['f1'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'localization_mae',
                                                 np.round(np.mean(metrics_dict['localization']['regression']), 4),
                                                 tools['epoch'])

            if configs.tasks.localization_deeploc:
                tools['train_writer'].add_scalar(f'localization_deeploc_accuracy',
                                                 np.round(metrics_dict['localization_deeploc']['accuracy'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'localization_deeploc_f1',
                                                 np.round(metrics_dict['localization_deeploc']['macro_f1'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'localization_deeploc_correct_prediction',
                                                 metrics_dict['localization_deeploc']['correct_prediction'],
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'localization_deeploc_wrong_token',
                                                 metrics_dict['localization_deeploc']['wrong_token'],
                                                 tools['epoch'])

            if configs.tasks.fold:
                tools['train_writer'].add_scalar(f'fold_accuracy', np.round(metrics_dict['fold']['accuracy'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'fold_f1', np.round(metrics_dict['fold']['f1'], 4),
                                                 tools['epoch'])

            if configs.tasks.gene_ontology:
                tools['train_writer'].add_scalar(f'gene_ontology_bp_accuracy',
                                                 np.round(metrics_dict['gene_ontology_bp']['accuracy'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'gene_ontology_bp_f1',
                                                 np.round(metrics_dict['gene_ontology_bp']['f1'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'gene_ontology_bp_correct_prediction',
                                                 metrics_dict['gene_ontology_bp']['correct_prediction'],
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'gene_ontology_bp_wrong_token',
                                                 metrics_dict['gene_ontology_bp']['wrong_token'],
                                                 tools['epoch'])

                tools['train_writer'].add_scalar(f'gene_ontology_mf_accuracy',
                                                 np.round(metrics_dict['gene_ontology_mf']['accuracy'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'gene_ontology_mf_f1',
                                                 np.round(metrics_dict['gene_ontology_mf']['f1'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'gene_ontology_mf_correct_prediction',
                                                 metrics_dict['gene_ontology_mf']['correct_prediction'],
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'gene_ontology_mf_wrong_token',
                                                 metrics_dict['gene_ontology_mf']['wrong_token'],
                                                 tools['epoch'])

                tools['train_writer'].add_scalar(f'gene_ontology_cc_accuracy',
                                                 np.round(metrics_dict['gene_ontology_cc']['accuracy'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'gene_ontology_cc_f1',
                                                 np.round(metrics_dict['gene_ontology_cc']['f1'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'gene_ontology_cc_correct_prediction',
                                                 metrics_dict['gene_ontology_cc']['correct_prediction'],
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'gene_ontology_cc_wrong_token',
                                                 metrics_dict['gene_ontology_cc']['wrong_token'],
                                                 tools['epoch'])

            if configs.tasks.enzyme_commission:
                tools['train_writer'].add_scalar(f'enzyme_commission_accuracy',
                                                 np.round(metrics_dict['enzyme_commission']['accuracy'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'enzyme_commission_f1',
                                                 np.round(metrics_dict['enzyme_commission']['f1'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'enzyme_commission_correct_prediction',
                                                 metrics_dict['enzyme_commission']['correct_prediction'],
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'enzyme_commission_wrong_token',
                                                 metrics_dict['enzyme_commission']['wrong_token'],
                                                 tools['epoch'])

            if configs.tasks.amino_to_fold_seek:
                tools['train_writer'].add_scalar(f'amino_to_fold_seek_accuracy',
                                                 np.round(metrics_dict['amino_to_fold_seek']['accuracy'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'amino_to_fold_seek_precision',
                                                 np.round(metrics_dict['amino_to_fold_seek']['precision'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'amino_to_fold_seek_recall',
                                                 np.round(metrics_dict['amino_to_fold_seek']['recall'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'amino_to_fold_seek_f1',
                                                 np.round(metrics_dict['amino_to_fold_seek']['f1'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'amino_to_fold_seek_correct_prediction',
                                                 metrics_dict['amino_to_fold_seek']['correct_prediction'],
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'amino_to_fold_seek_wrong_length',
                                                 metrics_dict['amino_to_fold_seek']['wrong_length'],
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'amino_to_fold_seek_wrong_token',
                                                 metrics_dict['amino_to_fold_seek']['wrong_token'],
                                                 tools['epoch'])

            if configs.tasks.secondary_structure:
                tools['train_writer'].add_scalar(f'secondary_structure_accuracy',
                                                 np.round(metrics_dict['secondary_structure']['accuracy'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'secondary_structure_precision',
                                                 np.round(metrics_dict['secondary_structure']['precision'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'secondary_structure_recall',
                                                 np.round(metrics_dict['secondary_structure']['recall'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'secondary_structure_f1',
                                                 np.round(metrics_dict['secondary_structure']['f1'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'secondary_structure_correct_prediction',
                                                 metrics_dict['secondary_structure']['correct_prediction'],
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'secondary_structure_wrong_length',
                                                 metrics_dict['secondary_structure']['wrong_length'],
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'secondary_structure_wrong_token',
                                                 metrics_dict['secondary_structure']['wrong_token'],
                                                 tools['epoch'])

            if configs.tasks.enzyme_reaction:
                tools['train_writer'].add_scalar(f'enzyme_reaction_accuracy',
                                                 np.round(metrics_dict['enzyme_reaction']['accuracy'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'enzyme_reaction_f1',
                                                 np.round(metrics_dict['enzyme_reaction']['f1'], 4),
                                                 tools['epoch'])

            if configs.tasks.human_ppi:
                tools['train_writer'].add_scalar(f'human_ppi_accuracy',
                                                 np.round(metrics_dict['human_ppi']['accuracy'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'human_ppi_f1', np.round(metrics_dict['human_ppi']['f1'], 4),
                                                 tools['epoch'])

            if configs.tasks.structure_similarity:
                tools['train_writer'].add_scalar(f'structure_similarity_spearman',
                                                 np.round(metrics_dict['structure_similarity']['spearman'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'structure_similarity_mae',
                                                 np.round(metrics_dict['structure_similarity']['mae'], 4),
                                                 tools['epoch'])

            if configs.tasks.protein_protein_interface:
                tools['train_writer'].add_scalar(f'protein_protein_interface_accuracy',
                                                 np.round(metrics_dict['protein_protein_interface']['accuracy'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'protein_protein_interface_f1',
                                                 np.round(metrics_dict['protein_protein_interface']['f1'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'protein_protein_interface_auc',
                                                 np.round(metrics_dict['protein_protein_interface']['auc'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'protein_protein_interface_mae',
                                                 np.round(metrics_dict['protein_protein_interface']['mae'], 4),
                                                 tools['epoch'])

            if configs.tasks.fluorescence:
                tools['train_writer'].add_scalar(f'fluorescence_spearman',
                                                 np.round(metrics_dict['fluorescence']['spearman'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'fluorescence_mae', np.round(metrics_dict['fluorescence']['mae'], 4),
                                                 tools['epoch'])

            if configs.tasks.stability:
                tools['train_writer'].add_scalar(f'stability_spearman',
                                                 np.round(metrics_dict['stability']['spearman'], 4),
                                                 tools['epoch'])
                tools['train_writer'].add_scalar(f'stability_mae', np.round(metrics_dict['stability']['mae'], 4),
                                                 tools['epoch'])

            if configs.tasks.protein_ligand_affinity:
                tools['train_writer'].add_scalar(f'protein_ligand_affinity_rmse',
                                                 np.round(metrics_dict['protein_ligand_affinity']['rmse'], 4),
                                                 tools['epoch'])

        tools['train_writer'].add_scalar(f'loss', np.round(train_loss, 8),
                                         tools['epoch'])


def validation_tensorboard_log(name, epoch, valid_loss, metrics_dict, tools, configs):
    metrics_dict = compute_all_metrics(metrics_dict, configs)
    if tools['tensorboard_log']:
        if name == 'fold':
            tools['valid_writer'].add_scalar(f'fold_accuracy', np.round(metrics_dict['fold']['accuracy'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'fold_f1', np.round(metrics_dict['fold']['f1'], 4),
                                             epoch)
        elif name == "gene_ontology":
            tools['valid_writer'].add_scalar(f'gene_ontology_bp_accuracy',
                                             np.round(metrics_dict['gene_ontology_bp']['accuracy'], 4),
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'gene_ontology_bp_f1',
                                             np.round(metrics_dict['gene_ontology_bp']['f1'], 4),
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'gene_ontology_bp_correct_prediction',
                                             metrics_dict['gene_ontology_bp']['correct_prediction'],
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'gene_ontology_bp_wrong_token',
                                             metrics_dict['gene_ontology_bp']['wrong_token'],
                                             tools['epoch'])

            tools['valid_writer'].add_scalar(f'gene_ontology_mf_accuracy',
                                             np.round(metrics_dict['gene_ontology_mf']['accuracy'], 4),
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'gene_ontology_mf_f1',
                                             np.round(metrics_dict['gene_ontology_mf']['f1'], 4),
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'gene_ontology_mf_correct_prediction',
                                             metrics_dict['gene_ontology_mf']['correct_prediction'],
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'gene_ontology_mf_wrong_token',
                                             metrics_dict['gene_ontology_mf']['wrong_token'],
                                             tools['epoch'])

            tools['valid_writer'].add_scalar(f'gene_ontology_cc_accuracy',
                                             np.round(metrics_dict['gene_ontology_cc']['accuracy'], 4),
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'gene_ontology_cc_f1',
                                             np.round(metrics_dict['gene_ontology_cc']['f1'], 4),
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'gene_ontology_cc_correct_prediction',
                                             metrics_dict['gene_ontology_cc']['correct_prediction'],
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'gene_ontology_cc_wrong_token',
                                             metrics_dict['gene_ontology_cc']['wrong_token'],
                                             tools['epoch'])

        if configs.tasks.enzyme_commission:
            tools['valid_writer'].add_scalar(f'enzyme_commission_accuracy',
                                             np.round(metrics_dict['enzyme_commission']['accuracy'], 4),
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'enzyme_commission_f1',
                                             np.round(metrics_dict['enzyme_commission']['f1'], 4),
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'enzyme_commission_correct_prediction',
                                             metrics_dict['enzyme_commission']['correct_prediction'],
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'enzyme_commission_wrong_token',
                                             metrics_dict['enzyme_commission']['wrong_token'],
                                             tools['epoch'])

        elif name == 'amino_to_fold_seek':
            tools['valid_writer'].add_scalar(f'amino_to_fold_seek_accuracy',
                                             np.round(metrics_dict['amino_to_fold_seek']['accuracy'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'amino_to_fold_seek_precision',
                                             np.round(metrics_dict['amino_to_fold_seek']['precision'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'amino_to_fold_seek_recall',
                                             np.round(metrics_dict['amino_to_fold_seek']['recall'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'amino_to_fold_seek_f1',
                                             np.round(metrics_dict['amino_to_fold_seek']['f1'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar('amino_to_fold_seek_correct_prediction',
                                             metrics_dict['amino_to_fold_seek']['correct_prediction'],
                                             tools['epoch'])
            tools['valid_writer'].add_scalar('amino_to_fold_seek_wrong_length',
                                             metrics_dict['amino_to_fold_seek']['wrong_length'],
                                             tools['epoch'])
            tools['valid_writer'].add_scalar('amino_to_fold_seek_wrong_token',
                                             metrics_dict['amino_to_fold_seek']['wrong_token'],
                                             tools['epoch'])

        elif name == 'secondary_structure':
            tools['valid_writer'].add_scalar('secondary_structure_accuracy',
                                             np.round(metrics_dict['secondary_structure']['accuracy'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar('secondary_structure_precision',
                                             np.round(metrics_dict['secondary_structure']['precision'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar('secondary_structure_recall',
                                             np.round(metrics_dict['secondary_structure']['recall'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar('secondary_structure_f1',
                                             np.round(metrics_dict['secondary_structure']['f1'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar('secondary_structure_correct_prediction',
                                             metrics_dict['secondary_structure']['correct_prediction'],
                                             tools['epoch'])
            tools['valid_writer'].add_scalar('secondary_structure_wrong_length',
                                             metrics_dict['secondary_structure']['wrong_length'],
                                             tools['epoch'])
            tools['valid_writer'].add_scalar('secondary_structure_wrong_token',
                                             metrics_dict['secondary_structure']['wrong_token'],
                                             tools['epoch'])

        elif name == 'enzyme_reaction':
            tools['valid_writer'].add_scalar(f'enzyme_reaction_accuracy',
                                             np.round(metrics_dict['enzyme_reaction']['accuracy'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'enzyme_reaction_f1', np.round(metrics_dict['enzyme_reaction']['f1'], 4),
                                             epoch)

        elif name == 'localization':
            tools['valid_writer'].add_scalar(f'localization_accuracy',
                                             np.round(metrics_dict['localization']['accuracy'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'localization_f1', np.round(metrics_dict['localization']['f1'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'localization_mae',
                                             np.round(np.mean(metrics_dict['localization']['regression']), 4),
                                             epoch)

        elif name == "loacalization_deeploc":
            tools['valid_writer'].add_scalar(f'loacalization_deeploc_accuracy',
                                             np.round(metrics_dict['loacalization_deeploc']['accuracy'], 4),
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'loacalization_deeploc_f1',
                                             np.round(metrics_dict['loacalization_deeploc']['macro_f1'], 4),
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'loacalization_deeploc_correct_prediction',
                                             metrics_dict['loacalization_deeploc']['correct_prediction'],
                                             tools['epoch'])
            tools['valid_writer'].add_scalar(f'loacalization_deeploc_wrong_token',
                                             metrics_dict['loacalization_deeploc']['wrong_token'],
                                             tools['epoch'])

        elif name == 'human_ppi':
            tools['valid_writer'].add_scalar(f'human_ppi_accuracy', np.round(metrics_dict['human_ppi']['accuracy'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'human_ppi_f1', np.round(metrics_dict['human_ppi']['f1'], 4),
                                             epoch)

        elif name == 'protein_protein_interface':
            tools['valid_writer'].add_scalar(f'protein_protein_interface_accuracy',
                                             np.round(metrics_dict['protein_protein_interface']['accuracy'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'protein_protein_interface_f1',
                                             np.round(metrics_dict['protein_protein_interface']['f1'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'protein_protein_interface_auc',
                                             np.round(metrics_dict['protein_protein_interface']['auc'], 4),
                                             epoch)
        elif name == 'structure_similarity':
            tools['valid_writer'].add_scalar(f'structure_similarity_spearman',
                                             np.round(metrics_dict['structure_similarity']['spearman'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'structure_similarity_mae',
                                             np.round(metrics_dict['structure_similarity']['mae'], 4),
                                             epoch)

        elif name == 'fluorescence':
            tools['valid_writer'].add_scalar(f'fluorescence_spearman',
                                             np.round(metrics_dict['fluorescence']['spearman'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'fluorescence_mae', np.round(metrics_dict['fluorescence']['mae'], 4),
                                             epoch)

        elif name == 'stability':
            tools['valid_writer'].add_scalar(f'stability_spearman', np.round(metrics_dict['stability']['spearman'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'stability_mae', np.round(metrics_dict['stability']['mae'], 4),
                                             epoch)

        elif name == 'protein_ligand_affinity':
            tools['valid_writer'].add_scalar(f'protein_ligand_affinity_rmse',
                                             np.round(metrics_dict['protein_ligand_affinity']['rmse'], 4),
                                             epoch)

        elif name in ['ubiquitylation', 'phosphorylation', 'acetylation', 'methylation',
                      'n_glycosylation', 'o_glycosylation', 'succinylation', 'duolin_ptm']:
            tools['valid_writer'].add_scalar(f'{name}_accuracy',
                                             np.round(metrics_dict[name]['accuracy'], 4),
                                             epoch)
            tools['valid_writer'].add_scalar(f'{name}_f1',
                                             np.round(metrics_dict[name]['f1'], 4),
                                             epoch)
        tools['valid_writer'].add_scalar(f'loss_{name}', np.round(valid_loss, 8),
                                         tools['epoch'])

        if configs.tasks.auxiliary:
            for name, metric_dict in metrics_dict['auxiliary'].items():
                if len(metric_dict['length']) > 0:
                    tools['valid_writer'].add_scalar(f'{name}_accuracy',
                                                     np.round(metric_dict['accuracy'], 4),
                                                     tools['epoch'])
                    tools['valid_writer'].add_scalar(f'{name}_f1',
                                                     np.round(metric_dict['f1'], 4),
                                                     tools['epoch'])


def training_log(epoch, train_loss, accelerator, tools, configs, metrics_dict, training_time, logging):
    if accelerator.is_main_process:
        logging.info(f'epoch {epoch}/{configs.train_settings.num_epochs} '
                     f'- steps {int(np.ceil(len(tools["train_loader"]) / tools["accum_iter"]))}, '
                     f'- time {np.round(training_time, 2)}s, '
                     f'train loss {np.round(train_loss, 8)}')
        if epoch > configs.train_settings.start_metric_epoch:
            for ptm_name in ptms_list:
                if len(metrics_dict[ptm_name]['length']) > 0:
                    logging.info(f'\ttrain {ptm_name} acc {np.round(metrics_dict[ptm_name]["accuracy"] * 100, 2)}, '
                                 f'f1 {np.round(metrics_dict[ptm_name]["f1"], 4)}, '
                                 f'mae number of prediction sites {np.round(np.mean(metrics_dict[ptm_name]["length"]), 4)}')

            if configs.tasks.auxiliary:
                list_accuracy = []
                list_f1 = []
                list_length = []
                for name, metric_dict in metrics_dict['auxiliary'].items():
                    if len(metric_dict['length']) > 0:
                        list_accuracy.append(np.round(metric_dict['accuracy'], 4))
                        list_f1.append(np.round(metric_dict['f1'], 4))
                        list_length.append(np.round(np.mean(metric_dict['length']), 4))
                if len(list_length) > 0:
                    logging.info(f'\ttrain average auxiliary acc {np.round(np.average(list_accuracy * 100), 2)}, '
                                 f'f1 {np.round(np.average(list_f1), 4)}, '
                                 f'mae number of prediction tokens {np.round(np.average(list_length), 4)}')

            if metrics_dict['localization']['correct_prediction'] > 0:
                logging.info(f'\ttrain localization acc {np.round(metrics_dict["localization"]["accuracy"] * 100, 4)}, '
                             f'macro f1 {np.round(metrics_dict["localization"]["f1"], 4)}, '
                             f'mae {np.round(np.mean(metrics_dict["localization"]["regression"]), 4)}, '
                             f'number of correct prediction {metrics_dict["localization"]["correct_prediction"]}')

            if metrics_dict['localization_deeploc']['correct_prediction'] > 0:
                logging.info(
                    f'\ttrain localization deeploc acc {np.round(metrics_dict["localization_deeploc"]["accuracy"] * 100, 4)}, '
                    f'macro f1 {np.round(metrics_dict["localization_deeploc"]["macro_f1"], 4)}, '
                    f'f1 {[np.round(v, 4) for v in metrics_dict["localization_deeploc"]["f1"].tolist()]}, '
                    f'number of wrong token {metrics_dict["localization_deeploc"]["wrong_token"]}, '
                    f'number of correct prediction {metrics_dict["localization_deeploc"]["correct_prediction"]}')

            if metrics_dict['fold']['correct_prediction'] > 0:
                logging.info(f'\ttrain fold acc {np.round(metrics_dict["fold"]["accuracy"] * 100, 4)}, '
                             f'macro f1 {np.round(metrics_dict["fold"]["f1"], 4)}, '
                             f'number of correct prediction {metrics_dict["fold"]["correct_prediction"]}')

            if metrics_dict['gene_ontology_bp']['correct_prediction'] > 0:
                logging.info(
                    f'\ttrain gene ontology bp acc {np.round(metrics_dict["gene_ontology_bp"]["accuracy"] * 100, 4)}, '
                    f'f1 {np.round(metrics_dict["gene_ontology_bp"]["f1"], 4)}, '
                    f'number of wrong token {metrics_dict["gene_ontology_bp"]["wrong_token"]}, '
                    f'number of correct prediction {metrics_dict["gene_ontology_bp"]["correct_prediction"]}')

            if metrics_dict['gene_ontology_mf']['correct_prediction'] > 0:
                logging.info(
                    f'\ttrain gene ontology mf acc {np.round(metrics_dict["gene_ontology_mf"]["accuracy"] * 100, 4)}, '
                    f'f1 {np.round(metrics_dict["gene_ontology_mf"]["f1"], 4)}, '
                    f'number of wrong token {metrics_dict["gene_ontology_mf"]["wrong_token"]}, '
                    f'number of correct prediction {metrics_dict["gene_ontology_mf"]["correct_prediction"]}')

            if metrics_dict['gene_ontology_cc']['correct_prediction'] > 0:
                logging.info(
                    f'\ttrain gene ontology cc acc {np.round(metrics_dict["gene_ontology_cc"]["accuracy"] * 100, 4)}, '
                    f'f1 {np.round(metrics_dict["gene_ontology_cc"]["f1"], 4)}, '
                    f'number of wrong token {metrics_dict["gene_ontology_cc"]["wrong_token"]}, '
                    f'number of correct prediction {metrics_dict["gene_ontology_cc"]["correct_prediction"]}')

            if metrics_dict['enzyme_commission']['correct_prediction'] > 0:
                logging.info(
                    f'\ttrain enzyme commission acc {np.round(metrics_dict["enzyme_commission"]["accuracy"] * 100, 4)}, '
                    f'f1 {np.round(metrics_dict["enzyme_commission"]["f1"], 4)}, '
                    f'number of wrong token {metrics_dict["enzyme_commission"]["wrong_token"]}, '
                    f'number of correct prediction {metrics_dict["enzyme_commission"]["correct_prediction"]}')

            if metrics_dict['amino_to_fold_seek']['correct_prediction'] > 0:
                logging.info(
                    f'\ttrain fold seek acc {np.round(metrics_dict["amino_to_fold_seek"]["accuracy"] * 100, 4)}, '
                    f'macro precision {np.round(metrics_dict["amino_to_fold_seek"]["precision"], 4)}, '
                    f'macro recall {np.round(metrics_dict["amino_to_fold_seek"]["recall"], 4)}, '
                    f'macro f1 {np.round(metrics_dict["amino_to_fold_seek"]["f1"], 4)}, '
                    f'wrong length {metrics_dict["amino_to_fold_seek"]["wrong_length"]}, '
                    f'wrong token {metrics_dict["amino_to_fold_seek"]["wrong_token"]}, '
                    f'number of correct prediction {metrics_dict["amino_to_fold_seek"]["correct_prediction"]}')

            if metrics_dict['secondary_structure']['correct_prediction'] > 0:
                logging.info(
                    f'\ttrain secondary structure acc {np.round(metrics_dict["secondary_structure"]["accuracy"] * 100, 4)}, '
                    f'macro precision {np.round(metrics_dict["secondary_structure"]["precision"], 4)}, '
                    f'macro recall {np.round(metrics_dict["secondary_structure"]["recall"], 4)}, '
                    f'macro f1 {np.round(metrics_dict["secondary_structure"]["f1"], 4)}, '
                    f'wrong length {metrics_dict["secondary_structure"]["wrong_length"]}, '
                    f'wrong token {metrics_dict["secondary_structure"]["wrong_token"]}, '
                    f'number of correct prediction {metrics_dict["secondary_structure"]["correct_prediction"]}')

            if metrics_dict['enzyme_reaction']['correct_prediction'] > 0:
                logging.info(
                    f'\ttrain enzyme reaction acc {np.round(metrics_dict["enzyme_reaction"]["accuracy"] * 100, 4)}, '
                    f'macro f1 {np.round(metrics_dict["enzyme_reaction"]["f1"], 4)}, '
                    f'number of correct prediction {metrics_dict["enzyme_reaction"]["correct_prediction"]}')

            if metrics_dict['human_ppi']['correct_prediction'] > 0:
                logging.info(f'\ttrain human ppi acc {np.round(metrics_dict["human_ppi"]["accuracy"] * 100, 4)}, '
                             f'macro f1 {np.round(metrics_dict["human_ppi"]["f1"], 4)}, '
                             f'number of correct prediction {metrics_dict["human_ppi"]["correct_prediction"]}')

            if metrics_dict['structure_similarity']['correct_prediction'] > 0:
                logging.info(
                    f'\ttrain structure similarity spearman p {np.round(metrics_dict["structure_similarity"]["spearman"], 4)}, '
                    f'mae {np.round(metrics_dict["structure_similarity"]["mae"], 4)}, '
                    f'number of correct prediction {metrics_dict["structure_similarity"]["correct_prediction"]}')

            if metrics_dict['protein_protein_interface']['correct_prediction'] > 0:
                logging.info(
                    f'\ttrain protein protein interface acc {np.round(metrics_dict["protein_protein_interface"]["accuracy"] * 100, 4)}, '
                    f'macro f1 {np.round(metrics_dict["protein_protein_interface"]["f1"], 4)}, '
                    f'auc {np.round(metrics_dict["protein_protein_interface"]["auc"], 4)}, '
                    f'wrong token {metrics_dict["protein_protein_interface"]["wrong_token"]}, '
                    f'number of correct prediction {metrics_dict["protein_protein_interface"]["correct_prediction"]}')

            if metrics_dict['fluorescence']['correct_prediction'] > 0:
                logging.info(
                    f'\ttrain fluorescence spearman p {np.round(metrics_dict["fluorescence"]["spearman"], 4)}, '
                    f'mae {np.round(metrics_dict["fluorescence"]["mae"], 4)}, '
                    f'number of correct prediction {metrics_dict["fluorescence"]["correct_prediction"]}')

            if metrics_dict['stability']['correct_prediction'] > 0:
                logging.info(f'\ttrain stability spearman p {np.round(metrics_dict["stability"]["spearman"], 4)}, '
                             f'mae {np.round(metrics_dict["stability"]["mae"], 4)}, '
                             f'number of correct prediction {metrics_dict["stability"]["correct_prediction"]}')

            if metrics_dict['protein_ligand_affinity']['correct_prediction'] > 0:
                logging.info(
                    f'\ttrain protein_ligand_affinity rmse {np.round(metrics_dict["protein_ligand_affinity"]["rmse"], 4)}, '
                    f'number of correct prediction {metrics_dict["protein_ligand_affinity"]["correct_prediction"]}')


def evaluation_log(i, name, valid_loss, dataloader, accelerator, metrics_dict, evaluation_time, logging, mode='valid'):
    if accelerator.is_main_process:
        logging.info(
            f'evaluation dataset {i + 1} - steps {len(dataloader)} - time {np.round(evaluation_time, 2)}s, '
            f'valid loss {np.round(valid_loss, 8)}')
        if name in ptms_list:
            if len(metrics_dict[name]['length']) > 0:
                logging.info(f'\t{mode} {name} acc {np.round(metrics_dict[name]["accuracy"] * 100, 2)}, '
                             f'f1 {np.round(metrics_dict[name]["f1"], 4)}, '
                             f'mae # of prediction sites {np.round(np.mean(metrics_dict[name]["length"]), 4)}')

        if name == 'auxiliary':
            list_accuracy = []
            list_f1 = []
            list_length = []
            for name, metric_dict in metrics_dict['auxiliary'].items():
                if len(metric_dict['length']) > 0:
                    list_accuracy.append(np.round(metric_dict['accuracy'], 4))
                    list_f1.append(np.round(metric_dict['f1'], 4))
                    list_length.append(np.round(np.mean(metric_dict['length']), 4))
            if len(list_length) > 0:
                logging.info(f'\t{mode} average auxiliary acc {np.round(np.average(list_accuracy * 100), 2)}, '
                             f'f1 {np.round(np.average(list_f1), 4)}, '
                             f'mae number of prediction tokens {np.round(np.average(list_length), 4)}')

        if metrics_dict['localization']["correct_prediction"] > 0:
            logging.info(f'\t{mode} localization acc {np.round(metrics_dict["localization"]["accuracy"] * 100, 4)}, '
                         f'macro f1 {np.round(metrics_dict["localization"]["f1"], 4)}, '
                         f'mae {np.round(np.mean(metrics_dict["localization"]["regression"]), 4)}, '
                         f'number of correct prediction {metrics_dict["localization"]["correct_prediction"]}')

        if metrics_dict['localization_deeploc']['correct_prediction'] > 0:
            logging.info(
                f'\t{mode} localization deeploc acc {np.round(metrics_dict["localization_deeploc"]["accuracy"] * 100, 4)}, '
                f'macro f1 {np.round(metrics_dict["localization_deeploc"]["macro_f1"], 4)}, '
                f'f1 {[np.round(v, 4) for v in metrics_dict["localization_deeploc"]["f1"].tolist()]}, '
                f'number of wrong token {metrics_dict["localization_deeploc"]["wrong_token"]}, '
                f'number of correct prediction {metrics_dict["localization_deeploc"]["correct_prediction"]}')

        if metrics_dict['fold']["correct_prediction"] > 0:
            logging.info(f'\t{mode} fold acc {np.round(metrics_dict["fold"]["accuracy"] * 100, 4)}, '
                         f'macro f1 {np.round(metrics_dict["fold"]["f1"], 4)}, '
                         f'number of correct prediction {metrics_dict["fold"]["correct_prediction"]}')

        if metrics_dict['gene_ontology_bp']['correct_prediction'] > 0:
            logging.info(
                f'\t{mode} gene ontology bp acc {np.round(metrics_dict["gene_ontology_bp"]["accuracy"] * 100, 4)}, '
                f'f1 {np.round(metrics_dict["gene_ontology_bp"]["f1"], 4)}, '
                f'number of wrong token {metrics_dict["gene_ontology_bp"]["wrong_token"]}, '
                f'number of correct prediction {metrics_dict["gene_ontology_bp"]["correct_prediction"]}')

        if metrics_dict['gene_ontology_mf']['correct_prediction'] > 0:
            logging.info(
                f'\t{mode} gene ontology mf acc {np.round(metrics_dict["gene_ontology_mf"]["accuracy"] * 100, 4)}, '
                f'f1 {np.round(metrics_dict["gene_ontology_mf"]["f1"], 4)}, '
                f'number of wrong token {metrics_dict["gene_ontology_mf"]["wrong_token"]}, '
                f'number of correct prediction {metrics_dict["gene_ontology_mf"]["correct_prediction"]}')

        if metrics_dict['gene_ontology_cc']['correct_prediction'] > 0:
            logging.info(
                f'\t{mode} gene ontology cc acc {np.round(metrics_dict["gene_ontology_cc"]["accuracy"] * 100, 4)}, '
                f'f1 {np.round(metrics_dict["gene_ontology_cc"]["f1"], 4)}, '
                f'number of wrong token {metrics_dict["gene_ontology_cc"]["wrong_token"]}, '
                f'number of correct prediction {metrics_dict["gene_ontology_cc"]["correct_prediction"]}')

        if metrics_dict['enzyme_commission']['correct_prediction'] > 0:
            logging.info(
                f'\t{mode} enzyme commission acc {np.round(metrics_dict["enzyme_commission"]["accuracy"] * 100, 4)}, '
                f'f1 {np.round(metrics_dict["enzyme_commission"]["f1"], 4)}, '
                f'number of wrong token {metrics_dict["enzyme_commission"]["wrong_token"]}, '
                f'number of correct prediction {metrics_dict["enzyme_commission"]["correct_prediction"]}')

        if metrics_dict['amino_to_fold_seek']["correct_prediction"] > 0:
            logging.info(f'\t{mode} fold seek acc {np.round(metrics_dict["amino_to_fold_seek"]["accuracy"] * 100, 4)}, '
                         f'macro precision {np.round(metrics_dict["amino_to_fold_seek"]["precision"], 4)}, '
                         f'macro recall {np.round(metrics_dict["amino_to_fold_seek"]["recall"], 4)}, '
                         f'macro f1 {np.round(metrics_dict["amino_to_fold_seek"]["f1"], 4)}, '
                         f'wrong length {metrics_dict["amino_to_fold_seek"]["wrong_length"]}, '
                         f'wrong token {metrics_dict["amino_to_fold_seek"]["wrong_token"]}, '
                         f'number of correct prediction {metrics_dict["amino_to_fold_seek"]["correct_prediction"]}')

        if metrics_dict['secondary_structure']["correct_prediction"] > 0:
            logging.info(
                f'\t{mode} secondary structure acc {np.round(metrics_dict["secondary_structure"]["accuracy"] * 100, 4)}, '
                f'macro precision {np.round(metrics_dict["secondary_structure"]["precision"], 4)}, '
                f'macro recall {np.round(metrics_dict["secondary_structure"]["recall"], 4)}, '
                f'macro f1 {np.round(metrics_dict["secondary_structure"]["f1"], 4)}, '
                f'wrong length {metrics_dict["secondary_structure"]["wrong_length"]}, '
                f'wrong token {metrics_dict["secondary_structure"]["wrong_token"]}, '
                f'number of correct prediction {metrics_dict["secondary_structure"]["correct_prediction"]}')

        if metrics_dict["enzyme_reaction"]["correct_prediction"] > 0:
            logging.info(
                f'\t{mode} enzyme reaction acc {np.round(metrics_dict["enzyme_reaction"]["accuracy"] * 100, 4)}, '
                f'macro f1 {np.round(metrics_dict["enzyme_reaction"]["f1"], 4)}, '
                f'number of correct prediction {metrics_dict["enzyme_reaction"]["correct_prediction"]}')

        if metrics_dict["human_ppi"]["correct_prediction"] > 0:
            logging.info(f'\t{mode} human ppi acc {np.round(metrics_dict["human_ppi"]["accuracy"] * 100, 4)}, '
                         f'macro f1 {np.round(metrics_dict["human_ppi"]["f1"], 4)}, '
                         f'number of correct prediction {metrics_dict["human_ppi"]["correct_prediction"]}')

        if metrics_dict['structure_similarity']['correct_prediction'] > 0:
            logging.info(
                f'\t{mode} structure_similarity spearman p {np.round(metrics_dict["structure_similarity"]["spearman"], 4)}, '
                f'mae {np.round(metrics_dict["structure_similarity"]["mae"], 4)}, '
                f'number of correct prediction {metrics_dict["structure_similarity"]["correct_prediction"]}')

        if metrics_dict['protein_protein_interface']["correct_prediction"] > 0:
            logging.info(
                f'\t{mode} protein protein interface acc {np.round(metrics_dict["protein_protein_interface"]["accuracy"] * 100, 4)}, '
                f'macro f1 {np.round(metrics_dict["protein_protein_interface"]["f1"], 4)}, '
                f'auc {np.round(metrics_dict["protein_protein_interface"]["auc"], 4)}, '
                f'wrong token {metrics_dict["protein_protein_interface"]["wrong_token"]}, '
                f'number of correct prediction {metrics_dict["protein_protein_interface"]["correct_prediction"]}')

        if metrics_dict['fluorescence']['correct_prediction'] > 0:
            logging.info(f'\t{mode} fluorescence spearman p {np.round(metrics_dict["fluorescence"]["spearman"], 4)}, '
                         f'mae {np.round(metrics_dict["fluorescence"]["mae"], 4)}, '
                         f'number of correct prediction {metrics_dict["fluorescence"]["correct_prediction"]}')

        if metrics_dict['stability']['correct_prediction'] > 0:
            logging.info(f'\t{mode} stability spearman p {np.round(metrics_dict["stability"]["spearman"], 4)}, '
                         f'mae {np.round(metrics_dict["stability"]["mae"], 4)}, '
                         f'number of correct prediction {metrics_dict["stability"]["correct_prediction"]}')

        if metrics_dict['protein_ligand_affinity']['correct_prediction'] > 0:
            logging.info(
                f'\t{mode} protein_ligand_affinity rmse {np.round(metrics_dict["protein_ligand_affinity"]["rmse"], 4)}, '
                f'number of correct prediction {metrics_dict["protein_ligand_affinity"]["correct_prediction"]}')
