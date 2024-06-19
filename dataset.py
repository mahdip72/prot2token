import os
import random
import torch
import yaml
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.utils import Tokenizer, calculate_class_weights, independent_exponential_smoothing
from transformers import AutoTokenizer
from tasks.localization import prepare_localization_samples
from tasks.fold import prepare_fold_samples
from tasks.enzyme_reaction import prepare_er_samples
from tasks.human_ppi import prepare_human_ppi_samples
from tasks.stability import prepare_stability_samples
from tasks.phosphorylation import prepare_phosphorylation_samples
from tasks.auxiliary_tasks import prepare_auxiliary_samples
from tasks.amino_to_fold_seek import prepare_amino_to_fold_seek_samples
from tasks.secondary_structure import prepare_secondary_structure_samples
from tasks.gene_ontology import prepare_gene_ontology_samples
from tasks.fluorescence import prepare_fluorescence_samples
from tasks.protein_protein_interface import prepare_protein_protein_interface_samples
from tasks.structure_similarity import prepare_structure_similarity_samples
from tasks.localization_deeploc import prepare_localization_deeploc_samples
from tasks.protein_ligand_affinity import prepare_protein_ligand_affinity_samples
from tasks.enzyme_commission import prepare_enzyme_commission_samples


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        raw_input_sequence, label = self.samples[idx]
        encoded_target = self.tokenizer(label, raw_input_sequence)
        encoded_target = torch.LongTensor(encoded_target)
        return raw_input_sequence, encoded_target

    def __len__(self):
        return len(self.samples)


class JointDataset(torch.utils.data.Dataset):
    def __init__(self, protein_encoder_tokenizer, decoder_tokenizer, task_weight, configs,
                 dataset_type='train', datasets_dict: dict = False, upsampling=False, upsampling_factor=1, **kwargs):

        for value, key in datasets_dict.items():
            setattr(self, f"{value}_list", key.samples)

        self.configs = configs
        self.dataset_type = dataset_type
        self.molecule_encoder_tokenizer = kwargs["molecule_encoder_tokenizer"]
        self.protein_encoder_tokenizer = protein_encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_protein_encoder_length = configs.prot2token_model.protein_encoder.max_len
        self.max_molecule_encoder_length = configs.prot2token_model.molecule_encoder.max_len
        self.max_decoder_length = configs.prot2token_model.decoder.max_len
        self.upsampling = upsampling
        self.upsampling_factor = upsampling_factor
        self.task_weight = task_weight
        self.items = []
        if upsampling:
            self.items += self.upscale_samples(
                [key.samples for key in datasets_dict.values()]
            )
        else:
            self.items += sum([key.samples for key in datasets_dict.values()], [])

    @staticmethod
    def upscale_samples(list_of_datasets):
        # Determine the maximum length among all lists
        max_len = max(len(dataset) for dataset in list_of_datasets)

        # Create a new list to store upscaled datasets
        upscaled_datasets = []

        for dataset in list_of_datasets:
            upscaled = dataset.copy()

            # Calculate how many samples to add
            diff = max_len - len(upscaled)

            # Upsample the current dataset by adding random duplicates
            for i in range(diff):
                rand_idx = random.randint(0, len(upscaled) - 1)
                upscaled.append(upscaled[rand_idx])

            # Append the upscaled dataset to the result list
            upscaled_datasets.extend(upscaled)

        return upscaled_datasets

    def __len__(self):
        return len(self.items)

    @staticmethod
    def random_masking(sequence, mask_token, prob=0.10) -> str:
        """
        Randomly replaces approximately mask_percent% of amino acids in the sequence with <mask>.

        :param sequence: The amino acid sequence as a string.
        :param prob: The percentage of amino acids to be replaced.
        :param mask_token: The token to replace the amino acids with.
        :return: The sequence with masked amino acids.
        """
        sequence_length = len(sequence)
        num_to_replace = int(sequence_length * prob)
        positions = random.sample(range(sequence_length), num_to_replace)

        masked_sequence = list(sequence)
        for pos in positions:
            masked_sequence[pos] = mask_token

        return ''.join(masked_sequence)

    @staticmethod
    def random_masking_ids(sequence_ids: torch.Tensor, mask_id: int, pad_token: int, exclude_ids: list,
                           prob: float = 0.10) -> torch.Tensor:
        """
        Randomly replaces approximately mask_percent% of amino acids in the sequence input_ids with <mask>.
        """
        if pad_token in list(sequence_ids.numpy()):
            sequence_length = list(sequence_ids.numpy()).index(1) - 1
        else:
            sequence_length = len(sequence_ids)
        num_to_replace = int(sequence_length * prob)
        positions = random.sample(range(sequence_length), num_to_replace)

        masked_sequence = sequence_ids.clone()
        for pos in positions:
            if masked_sequence[pos] not in exclude_ids:
                masked_sequence[pos] = mask_id

        return masked_sequence

    @staticmethod
    def extend_sample_weights_with_ones(sample_weight, encoded_target):
        # Calculate the difference in size
        diff = len(encoded_target) - len(sample_weight) - 1  # for bos token

        # Create a tensor filled with the value 1 of the required size
        extension = torch.full((diff,), 1)

        # Concatenate A with the extension tensor
        sample_weight = torch.cat((sample_weight, extension))

        return sample_weight

    def __getitem__(self, idx):
        task_name, sequence, target, prot_id, sample_weight = self.items[idx]
        sample_weight = torch.tensor(sample_weight)
        task_weight = 1
        if self.dataset_type == 'train':
            if self.configs.train_settings.task_weight:
                task_weight = self.task_weight[task_name]

        encoded_target = self.decoder_tokenizer(target, task_name=task_name, max_target_len=self.max_decoder_length)

        if len(sequence) == 3:
            smiles_sequence = sequence[2]
            sequence = sequence[0]
        else:
            smiles_sequence = ""

        if self.dataset_type == 'train':
            if self.configs.train_settings.random_masking > 0.0:
                if len(sequence) == 1:
                    pass
                else:
                    # This is for the case when we don't have two protein sequences
                    # sequence = self.random_masking(sequence, mask_token=self.protein_encoder_tokenizer.mask_token,
                    #                                prob=self.configs.train_settings.random_masking)

                    smiles_sequence = self.random_masking(
                        smiles_sequence, mask_token=self.molecule_encoder_tokenizer.mask_token,
                        prob=self.configs.train_settings.random_masking
                    )

        if self.protein_encoder_tokenizer:
            encoded_protein_sequence = self.protein_encoder_tokenizer(
                sequence, max_length=self.max_protein_encoder_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )  # todo

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

        if self.dataset_type == 'train':
            if self.configs.train_settings.random_masking > 0.0:
                encoded_protein_sequence['input_ids'] = self.random_masking_ids(
                    encoded_protein_sequence['input_ids'],
                    mask_id=self.protein_encoder_tokenizer.mask_token_id,
                    pad_token=self.protein_encoder_tokenizer.pad_token_id,
                    exclude_ids=list(self.protein_encoder_tokenizer.get_vocab().values())[-4:-1] + list(
                        self.protein_encoder_tokenizer.added_tokens_encoder.values()),
                    prob=self.configs.train_settings.random_masking
                )
        encoded_target = torch.LongTensor(encoded_target)

        if self.dataset_type == 'train':
            sample_weight = self.extend_sample_weights_with_ones(sample_weight, encoded_target)

            return encoded_protein_sequence, encoded_target, task_weight * sample_weight, encoded_molecule_sequence
        else:
            return encoded_protein_sequence, encoded_target, 1, encoded_molecule_sequence


def prepare_dataloaders(configs, logging):
    train_samples_list = []
    sum_of_target_samples = []
    tasks_tokens_list = []

    if configs.tasks.phosphorylation:
        train_samples_list.append(prepare_phosphorylation_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, r"phosphorylation/train.npz"),
            task_token=f"<task_phosphorylation>",
            positive_amino_acids=["S", "T", "Y"],
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            logging=logging,
            random_seed=configs.fix_seed
        ))
        tasks_tokens_list.append(f"<task_phosphorylation>")

    if configs.tasks.localization:
        train_localization_samples, valid_localization_samples = prepare_localization_samples(
            data_path=configs.train_settings.data_path,
            task_token=f"<task_localization>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        train_samples_list.append(train_localization_samples)
        sum_of_target_samples += ["other", "sp", "mt", "ch", "th"]
        tasks_tokens_list.append(f"<task_localization>")

    if configs.tasks.localization_deeploc:
        train_samples, localization_deeploc_label_index_mapping = prepare_localization_deeploc_samples(
            dataset_path=os.path.join(configs.train_settings.data_path,
                                      "localization_deeploc/Swissprot_Train_Validation_dataset.csv"),
            task_token=f"<task_localization_deeploc>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging,
            mode='train',
        )
        train_samples_list.append(train_samples)
        sum_of_target_samples += list(set([sample[2][0] for sample in train_samples_list[-1]]))
        tasks_tokens_list.append(f"<task_localization_deeploc>")
    else:
        localization_deeploc_label_index_mapping = {}

    if configs.tasks.fold:
        train_samples, fold_label_index_mapping = prepare_fold_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "fold_classification/train.csv"),
            task_token=f"<task_fold>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        train_samples_list.append(train_samples)
        sum_of_target_samples += list(set([sample[2][0] for sample in train_samples_list[-1]]))
        tasks_tokens_list.append(f"<task_fold>")
    else:
        fold_label_index_mapping = {}

    if configs.tasks.enzyme_reaction:
        train_samples, er_label_index_mapping = prepare_er_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "enzyme_reaction/train.csv"),
            task_token=f"<task_enzyme_reaction>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        train_samples_list.append(train_samples)
        sum_of_target_samples += list(er_label_index_mapping.keys())
        tasks_tokens_list.append(f"<task_enzyme_reaction>")
    else:
        er_label_index_mapping = {}

    if configs.tasks.human_ppi:
        train_samples, human_ppi_label_index_mapping = prepare_human_ppi_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "human_ppi/train.csv"),
            task_token=f"<task_human_ppi>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        train_samples_list.append(train_samples)
        sum_of_target_samples += list(human_ppi_label_index_mapping.keys())
        tasks_tokens_list.append("<task_human_ppi>")
    else:
        human_ppi_label_index_mapping = {}

    if configs.tasks.structure_similarity:
        train_samples, structure_similarity_label = prepare_structure_similarity_samples(
            dataset_path=configs.train_settings.data_path,
            task_token=f"<task_structure_similarity>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging,
            mode='train'
        )
        train_samples_list.append(train_samples)
        sum_of_target_samples += structure_similarity_label
        tasks_tokens_list.append(f"<task_structure_similarity>")
    else:
        structure_similarity_label = []

    if configs.tasks.protein_protein_interface:
        train_samples, protein_protein_interface_label_list = prepare_protein_protein_interface_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "protein_protein_interface/train.pkl"),
            task_token=f"<task_protein_protein_interface>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        train_samples_list.append(train_samples)
        sum_of_target_samples += protein_protein_interface_label_list
        tasks_tokens_list.append("<task_protein_protein_interface>")
    else:
        protein_protein_interface_label_list = []

    if configs.tasks.fluorescence:
        train_samples, fluorescence_label = prepare_fluorescence_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "fluorescence/train.csv"),
            task_token=f"<task_fluorescence>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        train_samples_list.append(train_samples)
        sum_of_target_samples += fluorescence_label
        tasks_tokens_list.append(f"<task_fluorescence>")
    else:
        fluorescence_label = []

    if configs.tasks.stability:
        train_samples, stability_label = prepare_stability_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "stability/train.csv"),
            task_token=f"<task_stability>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        train_samples_list.append(train_samples)
        sum_of_target_samples += stability_label
        tasks_tokens_list.append(f"<task_stability>")
    else:
        stability_label = []

    if configs.tasks.protein_ligand_affinity:
        if configs.prot2token_model.molecule_encoder.enable:
            train_samples, protein_ligand_affinity_label = prepare_protein_ligand_affinity_samples(
                dataset_path=os.path.join(configs.train_settings.data_path, "protein_ligand_affinity/train.csv"),
                task_token=f"<task_protein_ligand_affinity>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=configs.train_settings.max_task_samples,
                random_seed=configs.fix_seed,
                logging=logging
            )
            train_samples_list.append(train_samples)
            sum_of_target_samples += protein_ligand_affinity_label
            tasks_tokens_list.append(f"<task_protein_ligand_affinity>")
        else:
            raise ValueError("Molecule encoder must be enabled to train protein-ligand affinity task.")
    else:
        protein_ligand_affinity_label = []

    if configs.tasks.auxiliary:
        train_samples = prepare_auxiliary_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "swissprot/train_set.csv"),
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        train_samples_list.append(train_samples)
        tasks_list = list(set([sample[0] for sample in train_samples]))
        tasks_list.sort()
        tasks_tokens_list.extend(tasks_list)

    if configs.tasks.amino_to_fold_seek:
        train_samples, amino_to_fold_seek_label_index_mapping = prepare_amino_to_fold_seek_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "fold_seek/train_set.csv"),
            task_token=f"<task_amino_to_fold_seek>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        train_samples_list.append(train_samples)
        sum_of_target_samples += list(amino_to_fold_seek_label_index_mapping.keys())
        tasks_tokens_list.append("<task_amino_to_fold_seek>")
    else:
        amino_to_fold_seek_label_index_mapping = {}

    if configs.tasks.secondary_structure:
        train_samples, secondary_structure_label_index_mapping = prepare_secondary_structure_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "secondary_structure/train.csv"),
            task_token=f"<task_secondary_structure>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        train_samples_list.append(train_samples)
        sum_of_target_samples += list(secondary_structure_label_index_mapping.keys())
        tasks_tokens_list.append("<task_secondary_structure>")
    else:
        secondary_structure_label_index_mapping = {}

    if configs.tasks.gene_ontology:
        train_samples, gene_ontology_label_index_mapping = prepare_gene_ontology_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "gene_ontology/train.csv"),
            label_path=os.path.join(configs.train_settings.data_path, "gene_ontology/nrPDB-GO_annot.tsv"),
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            configs=configs,
            logging=logging,
        )
        train_samples_list.append(train_samples)
        target_samples = list(gene_ontology_label_index_mapping['mf'].keys())
        target_samples += list(gene_ontology_label_index_mapping['bp'].keys())
        target_samples += list(gene_ontology_label_index_mapping['cc'].keys())

        sum_of_target_samples += list(set(target_samples))

        tasks_tokens_list.append("<task_gene_ontology_mf>")
        tasks_tokens_list.append("<task_gene_ontology_bp>")
        tasks_tokens_list.append("<task_gene_ontology_cc>")
    else:
        gene_ontology_label_index_mapping = {}

    if configs.tasks.enzyme_commission:
        train_samples, ec_label_index_mapping, broken_ec_label_index_mapping = prepare_enzyme_commission_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "enzyme_commission/EC_train.csv"),
            label_path=os.path.join(configs.train_settings.data_path, "enzyme_commission/nrPDB-EC_annot.tsv"),
            task_token=f"<task_enzyme_commission>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        train_samples_list.append(train_samples)
        sum_of_target_samples += list(broken_ec_label_index_mapping.keys())
        tasks_tokens_list.append(f"<task_enzyme_commission>")
    else:
        ec_label_index_mapping = {}
        broken_ec_label_index_mapping = {}

    class_samples = {}
    for sub_dataset in train_samples_list:
        for sample in sub_dataset:
            if sample[0] not in class_samples.keys():
                # separate auxiliary task number of samples for the future
                class_samples[sample[0]] = len(sub_dataset)

    task_weight = calculate_class_weights(class_samples)
    task_weight = independent_exponential_smoothing(task_weight)

    decoder_tokenizer = Tokenizer(tasks_tokens_list=tasks_tokens_list,
                                  amino_to_fold_seek_label_index_mapping=amino_to_fold_seek_label_index_mapping,
                                  er_label_index_mapping=er_label_index_mapping,
                                  ec_label_index_mapping=ec_label_index_mapping,
                                  broken_ec_label_index_mapping=broken_ec_label_index_mapping,
                                  fold_label_index_mapping=fold_label_index_mapping,
                                  localization_deeploc_label_index_mapping=localization_deeploc_label_index_mapping,
                                  human_ppi_label_index_mapping=human_ppi_label_index_mapping,
                                  protein_protein_interface_label_list=protein_protein_interface_label_list,
                                  stability_label=stability_label,
                                  protein_ligand_affinity_label=protein_ligand_affinity_label,
                                  structure_similarity_label=structure_similarity_label,
                                  fluorescence_label=fluorescence_label,
                                  secondary_structure_label_index_mapping=secondary_structure_label_index_mapping,
                                  gene_ontology_label_index_mapping=gene_ontology_label_index_mapping,
                                  label_tokens=sum_of_target_samples,
                                  max_label_index=configs.prot2token_model.protein_encoder.max_len,
                                  configs=configs)

    train_datasets_dict = {}
    for i, train_samples_item in enumerate(train_samples_list):
        dataset_name = f"dataset_{i}"
        dataset = BaseDataset(
            train_samples_item,
            decoder_tokenizer,
        )
        train_datasets_dict[dataset_name] = dataset

    encoder_tokenizer = AutoTokenizer.from_pretrained(configs.prot2token_model.protein_encoder.model_name)

    encoder_molecule_tokenizer = AutoTokenizer.from_pretrained("gayane/BARTSmiles",
                                                               add_prefix_space=True)
    encoder_molecule_tokenizer.pad_token = '<pad>'
    encoder_molecule_tokenizer.bos_token = '<s>'
    encoder_molecule_tokenizer.eos_token = '</s>'
    encoder_molecule_tokenizer.mask_token = '<unk>'

    dataloaders_dict = {}
    train_joint_dataset = JointDataset(configs=configs,
                                       datasets_dict=train_datasets_dict,
                                       protein_encoder_tokenizer=encoder_tokenizer,
                                       molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                       decoder_tokenizer=decoder_tokenizer,
                                       dataset_type='train', task_weight=task_weight,
                                       upsampling=False, upsampling_factor=1
                                       )

    train_dataloader = DataLoader(
        train_joint_dataset,
        batch_size=configs.train_settings.batch_size,
        shuffle=configs.train_settings.shuffle,
        num_workers=configs.train_settings.num_workers,
        pin_memory=False,
    )
    dataloaders_dict["train"] = train_dataloader

    dataloaders_dict["valids"] = {}

    if configs.tasks.phosphorylation:
        valid_phosphorylation_samples = prepare_phosphorylation_samples(
            dataset_path=os.path.join(configs.valid_settings.data_path, r"phosphorylation/valid.npz"),
            task_token=f"<task_phosphorylation>",
            positive_amino_acids=["S", "T", "Y"],
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=10000,
            logging=logging,
            random_seed=configs.fix_seed,
        )
        valid_phosphorylation_dataset = BaseDataset(valid_phosphorylation_samples, decoder_tokenizer)
        valid_phosphorylation_dataset_final = JointDataset(configs=configs,
                                                           datasets_dict={
                                                               'dataset_phosphorylation': valid_phosphorylation_dataset},
                                                           protein_encoder_tokenizer=encoder_tokenizer,
                                                           molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                           decoder_tokenizer=decoder_tokenizer,
                                                           dataset_type='valid', task_weight=task_weight,
                                                           upsampling=False, upsampling_factor=1
                                                           )

        valid_phosphorylation_dataloader = DataLoader(
            valid_phosphorylation_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["phosphorylation"] = valid_phosphorylation_dataloader

    if configs.tasks.localization:
        valid_localization_dataset = BaseDataset(valid_localization_samples, decoder_tokenizer)
        valid_localization_dataset_final = JointDataset(configs=configs,
                                                        datasets_dict={
                                                            'dataset_localization': valid_localization_dataset},
                                                        protein_encoder_tokenizer=encoder_tokenizer,
                                                        molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                        decoder_tokenizer=decoder_tokenizer,
                                                        dataset_type='valid', task_weight=task_weight,
                                                        upsampling=False, upsampling_factor=1
                                                        )

        valid_localization_dataloader = DataLoader(
            valid_localization_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["localization"] = valid_localization_dataloader

    if configs.tasks.localization_deeploc:
        valid_localization_deeploc_samples, _ = prepare_localization_deeploc_samples(
            dataset_path=os.path.join(configs.valid_settings.data_path,
                                      "localization_deeploc/Swissprot_Train_Validation_dataset.csv"),
            task_token=f"<task_localization_deeploc>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging,
            mode='valid',
        )
        valid_localization_deeploc_dataset = BaseDataset(valid_localization_deeploc_samples, decoder_tokenizer)
        valid_localization_deeploc_dataset_final = JointDataset(configs=configs,
                                                                datasets_dict={
                                                                    'dataset_localization_deeploc': valid_localization_deeploc_dataset},
                                                                protein_encoder_tokenizer=encoder_tokenizer,
                                                                molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                                decoder_tokenizer=decoder_tokenizer,
                                                                dataset_type='valid', task_weight=task_weight,
                                                                upsampling=False, upsampling_factor=1
                                                                )

        valid_localization_deeploc_dataloader = DataLoader(
            valid_localization_deeploc_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["localization_deeploc"] = valid_localization_deeploc_dataloader

    if configs.tasks.fold:
        valid_fold_samples, _ = prepare_fold_samples(
            dataset_path=os.path.join(configs.valid_settings.data_path, "fold_classification/valid.csv"),
            task_token=f"<task_fold>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        valid_fold_dataset = BaseDataset(valid_fold_samples, decoder_tokenizer)
        valid_fold_dataset_final = JointDataset(configs=configs,
                                                datasets_dict={'dataset_fold': valid_fold_dataset},
                                                protein_encoder_tokenizer=encoder_tokenizer,
                                                molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                decoder_tokenizer=decoder_tokenizer,
                                                dataset_type='valid', task_weight=task_weight,
                                                upsampling=False, upsampling_factor=1
                                                )

        valid_fold_dataloader = DataLoader(
            valid_fold_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["fold"] = valid_fold_dataloader

    if configs.tasks.enzyme_reaction:
        valid_enzyme_reaction_samples, _ = prepare_er_samples(
            dataset_path=os.path.join(configs.valid_settings.data_path, "enzyme_reaction/validation.csv"),
            task_token=f"<task_enzyme_reaction>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=2562,
            random_seed=configs.fix_seed,
            logging=logging
        )
        valid_enzyme_reaction_dataset = BaseDataset(valid_enzyme_reaction_samples, decoder_tokenizer)
        valid_enzyme_reaction_dataset_final = JointDataset(configs=configs,
                                                           datasets_dict={
                                                               'dataset_enzyme_reaction': valid_enzyme_reaction_dataset},
                                                           protein_encoder_tokenizer=encoder_tokenizer,
                                                           molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                           decoder_tokenizer=decoder_tokenizer,
                                                           dataset_type='valid', task_weight=task_weight,
                                                           upsampling=False, upsampling_factor=1
                                                           )

        valid_enzyme_reaction_dataloader = DataLoader(
            valid_enzyme_reaction_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["enzyme_reaction"] = valid_enzyme_reaction_dataloader

    if configs.tasks.human_ppi:
        valid_human_ppi_samples, _ = prepare_human_ppi_samples(
            dataset_path=os.path.join(configs.valid_settings.data_path, "human_ppi/valid.csv"),
            task_token=f"<task_human_ppi>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        valid_human_ppi_dataset = BaseDataset(valid_human_ppi_samples, decoder_tokenizer)
        valid_human_ppi_dataset_final = JointDataset(configs=configs,
                                                     datasets_dict={'dataset_human_ppi': valid_human_ppi_dataset},
                                                     protein_encoder_tokenizer=encoder_tokenizer,
                                                     molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                     decoder_tokenizer=decoder_tokenizer,
                                                     dataset_type='valid', task_weight=task_weight,
                                                     upsampling=False, upsampling_factor=1
                                                     )

        valid_human_ppi_dataloader = DataLoader(
            valid_human_ppi_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["human_ppi"] = valid_human_ppi_dataloader

    if configs.tasks.structure_similarity:
        valid_structure_similarity_samples, _ = prepare_structure_similarity_samples(
            dataset_path=configs.valid_settings.data_path,
            task_token="<task_structure_similarity>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging,
            mode='valid'
        )
        valid_structure_similarity_dataset = BaseDataset(valid_structure_similarity_samples, decoder_tokenizer)
        valid_structure_similarity_dataset_final = JointDataset(
            configs=configs,
            datasets_dict={'dataset_structure_similarity': valid_structure_similarity_dataset},
            protein_encoder_tokenizer=encoder_tokenizer,
            molecule_encoder_tokenizer=encoder_molecule_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            dataset_type='valid', task_weight=task_weight,
            upsampling=False, upsampling_factor=1
        )

        valid_structure_similarity_dataloader = DataLoader(
            valid_structure_similarity_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["structure_similarity"] = valid_structure_similarity_dataloader

    if configs.tasks.protein_protein_interface:
        valid_protein_protein_interface_samples, _ = prepare_protein_protein_interface_samples(
            dataset_path=os.path.join(configs.valid_settings.data_path, "protein_protein_interface/valid.pkl"),
            task_token=f"<task_protein_protein_interface>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        valid_protein_protein_interface_dataset = BaseDataset(valid_protein_protein_interface_samples,
                                                              decoder_tokenizer)
        valid_protein_protein_interface_dataset_final = JointDataset(configs=configs,
                                                                     datasets_dict={
                                                                         'dataset_protein_protein_interface': valid_protein_protein_interface_dataset},
                                                                     protein_encoder_tokenizer=encoder_tokenizer,
                                                                     molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                                     decoder_tokenizer=decoder_tokenizer,
                                                                     dataset_type='valid', task_weight=task_weight,
                                                                     upsampling=False, upsampling_factor=1
                                                                     )

        valid_protein_protein_interface_dataloader = DataLoader(
            valid_protein_protein_interface_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["protein_protein_interface"] = valid_protein_protein_interface_dataloader

    if configs.tasks.fluorescence:
        valid_fluorescence_samples, _ = prepare_fluorescence_samples(
            dataset_path=os.path.join(configs.valid_settings.data_path, "fluorescence/valid.csv"),
            task_token=f"<task_fluorescence>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        valid_fluorescence_dataset = BaseDataset(valid_fluorescence_samples, decoder_tokenizer)
        valid_fluorescence_dataset_final = JointDataset(
            configs=configs,
            datasets_dict={'dataset_fluorescence': valid_fluorescence_dataset},
            protein_encoder_tokenizer=encoder_tokenizer,
            molecule_encoder_tokenizer=encoder_molecule_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            dataset_type='valid', task_weight=task_weight,
            upsampling=False, upsampling_factor=1
        )

        valid_fluorescence_dataloader = DataLoader(
            valid_fluorescence_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["fluorescence"] = valid_fluorescence_dataloader

    if configs.tasks.stability:
        valid_stability_samples, _ = prepare_stability_samples(
            dataset_path=os.path.join(configs.valid_settings.data_path, "stability/valid.csv"),
            task_token=f"<task_stability>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        valid_stability_dataset = BaseDataset(valid_stability_samples, decoder_tokenizer)
        valid_stability_dataset_final = JointDataset(
            configs=configs,
            datasets_dict={'dataset_stability': valid_stability_dataset},
            protein_encoder_tokenizer=encoder_tokenizer,
            molecule_encoder_tokenizer=encoder_molecule_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            dataset_type='valid', task_weight=task_weight,
            upsampling=False, upsampling_factor=1
        )

        valid_stability_dataloader = DataLoader(
            valid_stability_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["stability"] = valid_stability_dataloader

    if configs.tasks.protein_ligand_affinity:
        valid_protein_ligand_affinity_samples, _ = prepare_protein_ligand_affinity_samples(
            dataset_path=os.path.join(configs.valid_settings.data_path, "protein_ligand_affinity/valid.csv"),
            task_token=f"<task_protein_ligand_affinity>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=937,
            random_seed=configs.fix_seed,
            logging=logging
        )
        valid_protein_ligand_affinity_dataset = BaseDataset(valid_protein_ligand_affinity_samples, decoder_tokenizer)
        valid_protein_ligand_affinity_dataset_final = JointDataset(
            configs=configs,
            datasets_dict={'dataset_protein_ligand_affinity': valid_protein_ligand_affinity_dataset},
            protein_encoder_tokenizer=encoder_tokenizer,
            molecule_encoder_tokenizer=encoder_molecule_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            dataset_type='valid', task_weight=task_weight,
            upsampling=False, upsampling_factor=1
        )

        valid_protein_ligand_affinity_dataloader = DataLoader(
            valid_protein_ligand_affinity_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["protein_ligand_affinity"] = valid_protein_ligand_affinity_dataloader

    if configs.tasks.auxiliary:
        valid_auxiliary_samples = prepare_auxiliary_samples(
            dataset_path=os.path.join(configs.valid_settings.data_path, "swissprot/valid_set.csv"),
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=200,
            random_seed=configs.fix_seed,
            logging=logging
        )
        valid_auxiliary_dataset = BaseDataset(valid_auxiliary_samples, decoder_tokenizer)
        valid_auxiliary_dataset_final = JointDataset(configs=configs,
                                                     datasets_dict={'dataset_auxiliary': valid_auxiliary_dataset},
                                                     protein_encoder_tokenizer=encoder_tokenizer,
                                                     molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                     decoder_tokenizer=decoder_tokenizer,
                                                     dataset_type='valid', task_weight=task_weight,
                                                     upsampling=False, upsampling_factor=1
                                                     )

        valid_auxiliary_dataloader = DataLoader(
            valid_auxiliary_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["auxiliary"] = valid_auxiliary_dataloader

    if configs.tasks.amino_to_fold_seek:
        valid_amino_to_fold_seek_samples, _ = prepare_amino_to_fold_seek_samples(
            dataset_path=os.path.join(configs.valid_settings.data_path, "fold_seek/valid_set.csv"),
            task_token=f"<task_amino_to_fold_seek>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging
        )
        valid_amino_to_fold_seek_dataset = BaseDataset(valid_amino_to_fold_seek_samples, decoder_tokenizer)
        valid_amino_to_fold_seek_dataset_final = JointDataset(
            configs=configs,
            datasets_dict={'dataset_amino_to_fold_seek': valid_amino_to_fold_seek_dataset},
            protein_encoder_tokenizer=encoder_tokenizer,
            molecule_encoder_tokenizer=encoder_molecule_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            dataset_type='valid', task_weight=task_weight,
            upsampling=False, upsampling_factor=1
        )

        valid_amino_to_fold_seek_dataloader = DataLoader(
            valid_amino_to_fold_seek_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["amino_to_fold_seek"] = valid_amino_to_fold_seek_dataloader

    if configs.tasks.secondary_structure:
        valid_secondary_structure_samples, _ = prepare_secondary_structure_samples(
            dataset_path=os.path.join(configs.valid_settings.data_path, "secondary_structure/valid.csv"),
            task_token=f"<task_secondary_structure>",
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=2170,
            random_seed=configs.fix_seed,
            logging=logging
        )
        valid_secondary_structure_dataset = BaseDataset(valid_secondary_structure_samples, decoder_tokenizer)
        valid_secondary_structure_dataset_final = JointDataset(
            configs=configs,
            datasets_dict={'dataset_secondary_structure': valid_secondary_structure_dataset},
            protein_encoder_tokenizer=encoder_tokenizer,
            molecule_encoder_tokenizer=encoder_molecule_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            dataset_type='valid', task_weight=task_weight,
            upsampling=False, upsampling_factor=1
        )

        valid_secondary_structure_dataloader = DataLoader(
            valid_secondary_structure_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["secondary_structure"] = valid_secondary_structure_dataloader

    if configs.tasks.gene_ontology:
        valid_gene_ontology_samples, _ = prepare_gene_ontology_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "gene_ontology/valid.csv"),
            label_path=os.path.join(configs.train_settings.data_path, "gene_ontology/nrPDB-GO_annot.tsv"),
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=3323,
            random_seed=configs.fix_seed,
            logging=logging,
            configs=configs,
            task_type=['bp']
        )
        valid_gene_ontology_dataset = BaseDataset(valid_gene_ontology_samples, decoder_tokenizer)
        valid_gene_ontology_dataset_final = JointDataset(
            configs=configs,
            datasets_dict={'dataset_gene_ontology': valid_gene_ontology_dataset},
            protein_encoder_tokenizer=encoder_tokenizer,
            molecule_encoder_tokenizer=encoder_molecule_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            dataset_type='valid', task_weight=task_weight,
            upsampling=False, upsampling_factor=1
        )

        valid_gene_ontology_dataloader = DataLoader(
            valid_gene_ontology_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["gene_ontology_bp"] = valid_gene_ontology_dataloader

        valid_gene_ontology_samples, _ = prepare_gene_ontology_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "gene_ontology/valid.csv"),
            label_path=os.path.join(configs.train_settings.data_path, "gene_ontology/nrPDB-GO_annot.tsv"),
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=3323,
            random_seed=configs.fix_seed,
            logging=logging,
            configs=configs,
            task_type=['mf']
        )
        valid_gene_ontology_dataset = BaseDataset(valid_gene_ontology_samples, decoder_tokenizer)
        valid_gene_ontology_dataset_final = JointDataset(
            configs=configs,
            datasets_dict={'dataset_gene_ontology': valid_gene_ontology_dataset},
            protein_encoder_tokenizer=encoder_tokenizer,
            molecule_encoder_tokenizer=encoder_molecule_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            dataset_type='valid', task_weight=task_weight,
            upsampling=False, upsampling_factor=1
        )

        valid_gene_ontology_dataloader = DataLoader(
            valid_gene_ontology_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["gene_ontology_mf"] = valid_gene_ontology_dataloader

        valid_gene_ontology_samples, _ = prepare_gene_ontology_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "gene_ontology/valid.csv"),
            label_path=os.path.join(configs.train_settings.data_path, "gene_ontology/nrPDB-GO_annot.tsv"),
            max_length=configs.prot2token_model.protein_encoder.max_len,
            max_samples=3323,
            random_seed=configs.fix_seed,
            logging=logging,
            configs=configs,
            task_type=['cc']
        )
        valid_gene_ontology_dataset = BaseDataset(valid_gene_ontology_samples, decoder_tokenizer)
        valid_gene_ontology_dataset_final = JointDataset(
            configs=configs,
            datasets_dict={'dataset_gene_ontology': valid_gene_ontology_dataset},
            protein_encoder_tokenizer=encoder_tokenizer,
            molecule_encoder_tokenizer=encoder_molecule_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            dataset_type='valid', task_weight=task_weight,
            upsampling=False, upsampling_factor=1
        )

        valid_gene_ontology_dataloader = DataLoader(
            valid_gene_ontology_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["gene_ontology_cc"] = valid_gene_ontology_dataloader

    if configs.tasks.enzyme_commission:
        valid_enzyme_commission_samples, _, _ = prepare_enzyme_commission_samples(
            dataset_path=os.path.join(configs.train_settings.data_path, "enzyme_commission/EC_valid.csv"),
            label_path=os.path.join(configs.train_settings.data_path, "enzyme_commission/nrPDB-EC_annot.tsv"),
            task_token="<task_enzyme_commission>",
            max_length=1729,
            max_samples=configs.train_settings.max_task_samples,
            random_seed=configs.fix_seed,
            logging=logging,
        )
        valid_enzyme_commission_dataset = BaseDataset(valid_enzyme_commission_samples, decoder_tokenizer)
        valid_enzyme_commission_dataset_final = JointDataset(
            configs=configs,
            datasets_dict={'dataset_enzyme_commission': valid_enzyme_commission_dataset},
            protein_encoder_tokenizer=encoder_tokenizer,
            molecule_encoder_tokenizer=encoder_molecule_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            dataset_type='valid', task_weight=task_weight,
            upsampling=False, upsampling_factor=1
        )

        valid_enzyme_commission_dataloader = DataLoader(
            valid_enzyme_commission_dataset_final,
            batch_size=configs.valid_settings.batch_size,
            shuffle=False,
            num_workers=configs.valid_settings.num_workers,
            pin_memory=False,
        )
        dataloaders_dict['valids']["enzyme_commission"] = valid_enzyme_commission_dataloader

    if configs.test_settings.enable:
        dataloaders_dict["tests"] = {}

        if configs.tasks.phosphorylation:
            test_phosphorylation_samples = prepare_phosphorylation_samples(
                dataset_path=os.path.join(configs.test_settings.data_path, r"phosphorylation/test.npz"),
                task_token=f"<task_phosphorylation>",
                positive_amino_acids=["S", "T", "Y"],
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=426,  # at 2046 length
                logging=logging,
                random_seed=configs.fix_seed,
            )
            test_phosphorylation_dataset = BaseDataset(test_phosphorylation_samples, decoder_tokenizer)
            test_phosphorylation_dataset_final = JointDataset(configs=configs,
                                                              datasets_dict={
                                                                  'dataset_phosphorylation': test_phosphorylation_dataset},
                                                              protein_encoder_tokenizer=encoder_tokenizer,
                                                              molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                              decoder_tokenizer=decoder_tokenizer,
                                                              dataset_type='test', task_weight=task_weight,
                                                              upsampling=False, upsampling_factor=1
                                                              )

            test_phosphorylation_dataloader = DataLoader(
                test_phosphorylation_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["phosphorylation"] = test_phosphorylation_dataloader

        if configs.tasks.fluorescence:
            test_fluorescence_samples, _ = prepare_fluorescence_samples(
                dataset_path=os.path.join(configs.test_settings.data_path, "fluorescence/test.csv"),
                task_token=f"<task_fluorescence>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=27217,
                random_seed=configs.fix_seed,
                logging=logging
            )
            test_fluorescence_dataset = BaseDataset(test_fluorescence_samples, decoder_tokenizer)
            test_fluorescence_dataset_final = JointDataset(
                configs=configs,
                datasets_dict={'dataset_fluorescence': test_fluorescence_dataset},
                protein_encoder_tokenizer=encoder_tokenizer,
                molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                dataset_type='test', task_weight=task_weight,
                upsampling=False, upsampling_factor=1
            )

            test_fluorescence_dataloader = DataLoader(
                test_fluorescence_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["fluorescence"] = test_fluorescence_dataloader

        if configs.tasks.stability:
            test_stability_samples, _ = prepare_stability_samples(
                dataset_path=os.path.join(configs.test_settings.data_path, "stability/test.csv"),
                task_token=f"<task_stability>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=12851,
                random_seed=configs.fix_seed,
                logging=logging
            )
            test_stability_dataset = BaseDataset(test_stability_samples, decoder_tokenizer)
            test_stability_dataset_final = JointDataset(
                configs=configs,
                datasets_dict={'dataset_stability': test_stability_dataset},
                protein_encoder_tokenizer=encoder_tokenizer,
                molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                dataset_type='test', task_weight=task_weight,
                upsampling=False, upsampling_factor=1
            )

            test_stability_dataloader = DataLoader(
                test_stability_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["stability"] = test_stability_dataloader

        if configs.tasks.protein_ligand_affinity:
            test_protein_ligand_affinity_samples, _ = prepare_protein_ligand_affinity_samples(
                dataset_path=os.path.join(configs.test_settings.data_path, "protein_ligand_affinity/test.csv"),
                task_token=f"<task_protein_ligand_affinity>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=285,
                random_seed=configs.fix_seed,
                logging=logging
            )
            test_protein_ligand_affinity_dataset = BaseDataset(test_protein_ligand_affinity_samples, decoder_tokenizer)
            test_protein_ligand_affinity_dataset_final = JointDataset(
                configs=configs,
                datasets_dict={'dataset_protein_ligand_affinity': test_protein_ligand_affinity_dataset},
                protein_encoder_tokenizer=encoder_tokenizer,
                molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                dataset_type='test', task_weight=task_weight,
                upsampling=False, upsampling_factor=1
            )

            test_protein_ligand_affinity_dataloader = DataLoader(
                test_protein_ligand_affinity_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["protein_ligand_affinity"] = test_protein_ligand_affinity_dataloader

        if configs.tasks.enzyme_reaction:
            test_enzyme_reaction_samples, _ = prepare_er_samples(
                dataset_path=os.path.join(configs.test_settings.data_path, "enzyme_reaction/test.csv"),
                task_token=f"<task_enzyme_reaction>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=5651,
                random_seed=configs.fix_seed,
                logging=logging
            )
            test_enzyme_reaction_dataset = BaseDataset(test_enzyme_reaction_samples, decoder_tokenizer)
            test_enzyme_reaction_dataset_final = JointDataset(configs=configs,
                                                              datasets_dict={
                                                                  'dataset_enzyme_reaction': test_enzyme_reaction_dataset},
                                                              protein_encoder_tokenizer=encoder_tokenizer,
                                                              molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                              decoder_tokenizer=decoder_tokenizer,
                                                              dataset_type='test', task_weight=task_weight,
                                                              upsampling=False, upsampling_factor=1
                                                              )

            test_enzyme_reaction_dataloader = DataLoader(
                test_enzyme_reaction_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["enzyme_reaction"] = test_enzyme_reaction_dataloader

        if configs.tasks.localization_deeploc:
            test_localization_deeploc_samples, _ = prepare_localization_deeploc_samples(
                dataset_path=os.path.join(configs.test_settings.data_path, "localization_deeploc/hpa_testset.csv"),
                task_token=f"<task_localization_deeploc>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=1717,
                random_seed=configs.fix_seed,
                logging=logging,
                mode='test',
            )
            test_localization_deeploc_dataset = BaseDataset(test_localization_deeploc_samples, decoder_tokenizer)
            test_localization_deeploc_dataset_final = JointDataset(configs=configs,
                                                                   datasets_dict={
                                                                       'dataset_localization_deeploc': test_localization_deeploc_dataset},
                                                                   protein_encoder_tokenizer=encoder_tokenizer,
                                                                   molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                                   decoder_tokenizer=decoder_tokenizer,
                                                                   dataset_type='test', task_weight=task_weight,
                                                                   upsampling=False, upsampling_factor=1
                                                                   )

            test_localization_deeploc_dataloader = DataLoader(
                test_localization_deeploc_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["localization_deeploc"] = test_localization_deeploc_dataloader

        if configs.tasks.fold:
            test_fold_samples, _ = prepare_fold_samples(
                dataset_path=os.path.join(configs.test_settings.data_path, "fold_classification/test_fold_holdout.csv"),
                task_token=f"<task_fold>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=718,
                random_seed=configs.fix_seed,
                logging=logging
            )
            test_fold_dataset = BaseDataset(test_fold_samples, decoder_tokenizer)
            test_fold_dataset_final = JointDataset(configs=configs,
                                                   datasets_dict={'dataset_fold': test_fold_dataset},
                                                   protein_encoder_tokenizer=encoder_tokenizer,
                                                   molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                   decoder_tokenizer=decoder_tokenizer,
                                                   dataset_type='test', task_weight=task_weight,
                                                   upsampling=False, upsampling_factor=1
                                                   )

            test_fold_dataloader = DataLoader(
                test_fold_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["fold"] = test_fold_dataloader

            test_fold_samples, _ = prepare_fold_samples(
                dataset_path=os.path.join(configs.test_settings.data_path,
                                          "fold_classification/test_family_holdout.csv"),
                task_token=f"<task_fold>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=1272,
                random_seed=configs.fix_seed,
                logging=logging
            )
            test_fold_dataset = BaseDataset(test_fold_samples, decoder_tokenizer)
            test_fold_dataset_final = JointDataset(configs=configs,
                                                   datasets_dict={'dataset_fold': test_fold_dataset},
                                                   protein_encoder_tokenizer=encoder_tokenizer,
                                                   molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                   decoder_tokenizer=decoder_tokenizer,
                                                   dataset_type='test', task_weight=task_weight,
                                                   upsampling=False, upsampling_factor=1
                                                   )

            test_fold_dataloader = DataLoader(
                test_fold_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            # dataloaders_dict['tests']["fold"] = test_fold_dataloader

            test_fold_samples, _ = prepare_fold_samples(
                dataset_path=os.path.join(configs.test_settings.data_path,
                                          "fold_classification/test_superfamily_holdout.csv"),
                task_token=f"<task_fold>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=1254,
                random_seed=configs.fix_seed,
                logging=logging
            )
            test_fold_dataset = BaseDataset(test_fold_samples, decoder_tokenizer)
            test_fold_dataset_final = JointDataset(configs=configs,
                                                   datasets_dict={'dataset_fold': test_fold_dataset},
                                                   protein_encoder_tokenizer=encoder_tokenizer,
                                                   molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                   decoder_tokenizer=decoder_tokenizer,
                                                   dataset_type='test', task_weight=task_weight,
                                                   upsampling=False, upsampling_factor=1
                                                   )

            test_fold_dataloader = DataLoader(
                test_fold_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            # dataloaders_dict['tests']["fold"] = test_fold_dataloader

        if configs.tasks.secondary_structure:
            test_secondary_structure_samples, _ = prepare_secondary_structure_samples(
                dataset_path=os.path.join(configs.test_settings.data_path, "secondary_structure/test.csv"),
                task_token=f"<task_secondary_structure>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=513,
                random_seed=configs.fix_seed,
                logging=logging
            )
            test_secondary_structure_dataset = BaseDataset(test_secondary_structure_samples, decoder_tokenizer)
            test_secondary_structure_dataset_final = JointDataset(
                configs=configs,
                datasets_dict={'dataset_secondary_structure': test_secondary_structure_dataset},
                protein_encoder_tokenizer=encoder_tokenizer,
                molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                dataset_type='test', task_weight=task_weight,
                upsampling=False, upsampling_factor=1
            )

            test_secondary_structure_dataloader = DataLoader(
                test_secondary_structure_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["secondary_structure"] = test_secondary_structure_dataloader

        if configs.tasks.gene_ontology:
            test_gene_ontology_samples, _ = prepare_gene_ontology_samples(
                dataset_path=os.path.join(configs.train_settings.data_path, "gene_ontology/test.csv"),
                label_path=os.path.join(configs.train_settings.data_path, "gene_ontology/nrPDB-GO_annot.tsv"),
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=3416,
                random_seed=configs.fix_seed,
                logging=logging,
                configs=configs,
                task_type=['bp']
            )
            test_gene_ontology_dataset = BaseDataset(test_gene_ontology_samples, decoder_tokenizer)
            test_gene_ontology_dataset_final = JointDataset(configs=configs,
                                                            datasets_dict={
                                                                'dataset_gene_ontology': test_gene_ontology_dataset},
                                                            protein_encoder_tokenizer=encoder_tokenizer,
                                                            molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                            decoder_tokenizer=decoder_tokenizer,
                                                            dataset_type='test', task_weight=task_weight,
                                                            upsampling=False, upsampling_factor=1
                                                            )

            test_gene_ontology_dataloader = DataLoader(
                test_gene_ontology_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["gene_ontology_bp"] = test_gene_ontology_dataloader

            test_gene_ontology_samples, _ = prepare_gene_ontology_samples(
                dataset_path=os.path.join(configs.train_settings.data_path, "gene_ontology/test.csv"),
                label_path=os.path.join(configs.train_settings.data_path, "gene_ontology/nrPDB-GO_annot.tsv"),
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=3416,
                random_seed=configs.fix_seed,
                logging=logging,
                configs=configs,
                task_type=['mf']
            )
            test_gene_ontology_dataset = BaseDataset(test_gene_ontology_samples, decoder_tokenizer)
            test_gene_ontology_dataset_final = JointDataset(configs=configs,
                                                            datasets_dict={
                                                                'dataset_gene_ontology': test_gene_ontology_dataset},
                                                            protein_encoder_tokenizer=encoder_tokenizer,
                                                            molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                            decoder_tokenizer=decoder_tokenizer,
                                                            dataset_type='test', task_weight=task_weight,
                                                            upsampling=False, upsampling_factor=1
                                                            )

            test_gene_ontology_dataloader = DataLoader(
                test_gene_ontology_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["gene_ontology_mf"] = test_gene_ontology_dataloader

            test_gene_ontology_samples, _ = prepare_gene_ontology_samples(
                dataset_path=os.path.join(configs.train_settings.data_path, "gene_ontology/test.csv"),
                label_path=os.path.join(configs.train_settings.data_path, "gene_ontology/nrPDB-GO_annot.tsv"),
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=3416,
                random_seed=configs.fix_seed,
                logging=logging,
                configs=configs,
                task_type=['cc']
            )
            test_gene_ontology_dataset = BaseDataset(test_gene_ontology_samples, decoder_tokenizer)
            test_gene_ontology_dataset_final = JointDataset(configs=configs,
                                                            datasets_dict={
                                                                'dataset_gene_ontology': test_gene_ontology_dataset},
                                                            protein_encoder_tokenizer=encoder_tokenizer,
                                                            molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                            decoder_tokenizer=decoder_tokenizer,
                                                            dataset_type='test', task_weight=task_weight,
                                                            upsampling=False, upsampling_factor=1
                                                            )

            test_gene_ontology_dataloader = DataLoader(
                test_gene_ontology_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["gene_ontology_cc"] = test_gene_ontology_dataloader

        if configs.tasks.enzyme_commission:
            test_enzyme_commission_samples, _, _ = prepare_enzyme_commission_samples(
                dataset_path=os.path.join(configs.train_settings.data_path, "enzyme_commission/EC_test.csv"),
                label_path=os.path.join(configs.train_settings.data_path, "enzyme_commission/nrPDB-EC_annot.tsv"),
                task_token="<task_enzyme_commission>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=1919,
                random_seed=configs.fix_seed,
                logging=logging,
            )
            test_enzyme_commission_dataset = BaseDataset(test_enzyme_commission_samples, decoder_tokenizer)
            test_enzyme_commission_dataset_final = JointDataset(
                configs=configs,
                datasets_dict={'dataset_enzyme_commission': test_enzyme_commission_dataset},
                protein_encoder_tokenizer=encoder_tokenizer,
                molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                dataset_type='test', task_weight=task_weight,
                upsampling=False, upsampling_factor=1
            )

            test_enzyme_commission_dataloader = DataLoader(
                test_enzyme_commission_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["enzyme_commission"] = test_enzyme_commission_dataloader

        if configs.tasks.human_ppi:
            test_human_ppi_samples, _ = prepare_human_ppi_samples(
                dataset_path=os.path.join(configs.test_settings.data_path, "human_ppi/test.csv"),
                task_token=f"<task_human_ppi>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=237,
                random_seed=configs.fix_seed,
                logging=logging
            )
            test_human_ppi_dataset = BaseDataset(test_human_ppi_samples, decoder_tokenizer)
            test_human_ppi_dataset_final = JointDataset(configs=configs,
                                                        datasets_dict={'dataset_human_ppi': test_human_ppi_dataset},
                                                        protein_encoder_tokenizer=encoder_tokenizer,
                                                        molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                        decoder_tokenizer=decoder_tokenizer,
                                                        dataset_type='test', task_weight=task_weight,
                                                        upsampling=False, upsampling_factor=1
                                                        )

            test_human_ppi_dataloader = DataLoader(
                test_human_ppi_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["human_ppi"] = test_human_ppi_dataloader

        if configs.tasks.structure_similarity:
            test_structure_similarity_samples, _ = prepare_structure_similarity_samples(
                dataset_path=configs.test_settings.data_path,
                task_token="<task_structure_similarity>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=7000,
                random_seed=configs.fix_seed,
                logging=logging,
                mode='test'
            )
            test_structure_similarity_dataset = BaseDataset(test_structure_similarity_samples, decoder_tokenizer)
            test_structure_similarity_dataset_final = JointDataset(
                configs=configs,
                datasets_dict={'dataset_structure_similarity': test_structure_similarity_dataset},
                protein_encoder_tokenizer=encoder_tokenizer,
                molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                dataset_type='test', task_weight=task_weight,
                upsampling=False, upsampling_factor=1
            )

            test_structure_similarity_dataloader = DataLoader(
                test_structure_similarity_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["structure_similarity"] = test_structure_similarity_dataloader

        if configs.tasks.protein_protein_interface:
            test_protein_protein_interface_samples, _ = prepare_protein_protein_interface_samples(
                dataset_path=os.path.join(configs.test_settings.data_path, "protein_protein_interface/test.pkl"),
                task_token=f"<task_protein_protein_interface>",
                max_length=configs.prot2token_model.protein_encoder.max_len,
                max_samples=568,
                random_seed=configs.fix_seed,
                logging=logging
            )
            test_protein_protein_interface_dataset = BaseDataset(test_protein_protein_interface_samples,
                                                                 decoder_tokenizer)
            test_protein_protein_interface_dataset_final = JointDataset(configs=configs,
                                                                        datasets_dict={
                                                                            'dataset_protein_protein_interface': test_protein_protein_interface_dataset},
                                                                        protein_encoder_tokenizer=encoder_tokenizer,
                                                                        molecule_encoder_tokenizer=encoder_molecule_tokenizer,
                                                                        decoder_tokenizer=decoder_tokenizer,
                                                                        dataset_type='test', task_weight=task_weight,
                                                                        upsampling=False, upsampling_factor=1
                                                                        )

            test_protein_protein_interface_dataloader = DataLoader(
                test_protein_protein_interface_dataset_final,
                batch_size=configs.test_settings.batch_size,
                shuffle=False,
                num_workers=configs.test_settings.num_workers,
                pin_memory=False,
            )
            dataloaders_dict['tests']["protein_protein_interface"] = test_protein_protein_interface_dataloader

    return dataloaders_dict, encoder_tokenizer, decoder_tokenizer


if __name__ == '__main__':
    from utils.utils import get_dummy_logger
    from utils.utils import load_configs

    logger, buffer = get_dummy_logger()

    config_path = './configs/config.yaml'

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs(config_file)

    dataloaders_dict, encoder_tokenizer, decoder_tokenizer = prepare_dataloaders(test_configs, logger)
    # For test dataset modules
    print('done')
