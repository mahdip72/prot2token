import torch
import math
from torch import nn
from timm.models.layers import trunc_normal_
from transformers import EsmModel
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from peft import LoraConfig, PeftConfig, get_peft_model, prepare_model_for_kbit_training
from .utils import prepare_config_and_checkpoint, load_checkpoints_inference
from tqdm import tqdm


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Create a long tensor containing positions
        position = torch.arange(0, max_len).unsqueeze(1)
        # Create a long tensor containing dimension values
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # Calculate positional encodings
        pos_enc = torch.zeros((1, max_len, d_model))
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        return x + self.pos_enc[:, :x.size(1)]


def get_nb_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(model, logging, description=""):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)
    logging.info(
        f"{description} trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )


def verify_data_types(model, logging):
    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        logging.info(f"{k}, {v}, {v / total}")


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device))
            == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(tgt, pad_idx, device):
    """
    tgt: shape(N, L)
    """
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    tgt_padding_mask = (tgt == pad_idx)

    return tgt_mask, tgt_padding_mask


class ProteinEncoder(nn.Module):
    def __init__(self, configs, model_name='facebook/esm2_t33_650M_UR50D', out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        if configs.prot2token_model.protein_encoder.quantization_4_bit:
            from transformers import BitsAndBytesConfig
            # QLoRa fine-tuning:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = EsmModel.from_pretrained(model_name, quantization_config=quantization_config,
                                                  load_in_4bit=True)
            self.model = prepare_model_for_kbit_training(self.model,
                                                         use_gradient_checkpointing=True)
        else:
            self.model = EsmModel.from_pretrained(model_name)

        if configs.prot2token_model.protein_encoder.lora.enable:
            config = LoraConfig(
                r=configs.prot2token_model.protein_encoder.lora.r,
                lora_alpha=configs.prot2token_model.protein_encoder.lora.lora_alpha,
                target_modules=[
                    "query",
                    "key",
                    "value",
                    "dense"
                ],
                inference_mode=False,
                # modules_to_save=["pooler"],
                lora_dropout=configs.prot2token_model.protein_encoder.lora.lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(self.model, config)
            if configs.prot2token_model.protein_encoder.quantization_4_bit:
                for param in self.model.embeddings.word_embeddings.parameters():
                    param.requires_grad = True

        elif not configs.prot2token_model.protein_encoder.quantization_4_bit and not configs.prot2token_model.protein_encoder.lora.enable and configs.prot2token_model.protein_encoder.fine_tune.enable:
            # fine-tune the latest layer

            # Freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False

            # Allow the parameters of the last transformer block to be updated during fine-tuning
            for param in self.model.encoder.layer[
                         -configs.prot2token_model.protein_encoder.fine_tune.last_layers_trainable:].parameters():
                param.requires_grad = True

            for param in self.model.encoder.emb_layer_norm_after.parameters():
                param.requires_grad = True

            # self.model.gradient_checkpointing_enable()
        else:
            # Freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False

        if configs.prot2token_model.protein_encoder.tune_embedding:
            for param in self.model.embeddings.word_embeddings.parameters():
                param.requires_grad = True

        for param in self.model.pooler.parameters():
            param.requires_grad = False

        for param in self.model.contact_head.parameters():
            param.requires_grad = False

        self.bottleneck = nn.Conv1d(self.model.embeddings.position_embeddings.embedding_dim, out_dim, 1)

        # print_trainable_parameters(self.model, logging)
        # verify_data_types(self.model, logging)

    def forward(self, x):
        features = self.model(input_ids=x["protein_sequence"]['input_ids'],
                              attention_mask=x['protein_sequence']['attention_mask'])
        features.last_hidden_state = features.last_hidden_state.permute(0, 2, 1)
        return self.bottleneck(features.last_hidden_state).permute(0, 2, 1)


class MoleculeEncoder(nn.Module):
    def __init__(self, configs, model_name, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

        self.model = AutoModel.from_pretrained(model_name)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        if configs.prot2token_model.molecule_encoder.fine_tune.enable:
            # Allow the parameters of the last transformer block to be updated during fine-tuning
            for param in self.model.encoder.layers[
                         -configs.prot2token_model.molecule_encoder.fine_tune.last_layers_trainable:].parameters():
                param.requires_grad = True

        if configs.prot2token_model.molecule_encoder.tune_embedding:
            for param in self.model.encoder.embed_tokens.parameters():
                param.requires_grad = True

        self.bottleneck = nn.Conv1d(self.model.shared.embedding_dim, out_dim, 1)

    def forward(self, x):
        features = self.model(input_ids=x["molecule_sequence"]['input_ids'],
                              attention_mask=x['molecule_sequence']['attention_mask'])
        features.encoder_last_hidden_state = features.encoder_last_hidden_state.permute(0, 2, 1)
        return self.bottleneck(features.encoder_last_hidden_state).permute(0, 2, 1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, dim_feedforward, max_len, pad_idx, activation_function,
                 **kwargs):
        super().__init__()
        self.configs = kwargs['configs']
        self.decoder_tokenizer = kwargs['decoder_tokenizer']
        self.dim = dim
        self.positional_encoding_type = self.configs.prot2token_model.positional_encoding_type

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim, padding_idx=0)
        if self.positional_encoding_type == 'absolute':
            self.absolute = self.pos_enc = AbsolutePositionalEmbedding(
                dim, max(self.configs.prot2token_model.protein_encoder.max_len,
                         self.configs.prot2token_model.molecule_encoder.max_len, max_len))
        elif self.positional_encoding_type == 'learned':
            self.decoder_pos_embed = nn.Parameter(torch.randn(1, max_len - 1, dim) * .02)
            self.decoder_pos_drop = nn.Dropout(p=0.05)
        else:
            raise ValueError(f'Unknown positional encoding type: {self.positional_encoding_type}')

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim_feedforward,
                                                   activation=activation_function)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(dim, vocab_size)

        if self.configs.prot2token_model.protein_encoder.drop_positional_encoding and self.positional_encoding_type == 'learned':
            self.protein_encoder_pos_embed = nn.Parameter(
                torch.randn(1, self.configs.prot2token_model.protein_encoder.max_len, dim) * .02)
            self.protein_encoder_pos_drop = nn.Dropout(p=0.05)

        if self.configs.prot2token_model.molecule_encoder.drop_positional_encoding and self.positional_encoding_type == 'learned' and self.configs.prot2token_model.molecule_encoder.enable:
            self.molecule_encoder_pos_embed = nn.Parameter(
                torch.randn(1, self.configs.prot2token_model.molecule_encoder.max_len, dim) * .02)
            self.molecule_encoder_pos_drop = nn.Dropout(p=0.05)

        self.max_len = max_len
        self.pad_idx = pad_idx
        # self.device = device

        # self.init_weights(logging)

    def init_weights(self, logging):
        for name, p in self.named_parameters():
            if name in ['protein_encoder_pos_embed', 'decoder_pos_embed', 'molecule_encoder_pos_embed',
                        'embedding.weight'
                        ]:
                # logging.info(f"skipping randomly initializing {name}")
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if self.configs.prot2token_model.protein_encoder.drop_positional_encoding and self.positional_encoding_type == 'learned':
            trunc_normal_(self.protein_encoder_pos_embed, std=.02)
        if self.configs.prot2token_model.molecule_encoder.enable:
            if self.configs.prot2token_model.molecule_encoder.drop_positional_encoding and self.positional_encoding_type == 'learned':
                trunc_normal_(self.molecule_encoder_pos_embed, std=.02)
        if self.positional_encoding_type == 'learned':
            trunc_normal_(self.decoder_pos_embed, std=.02)

    def drop_positional_encoding(self, embedding, model_type):
        # Get the sequence length from the embedding
        _, seq_length, _ = embedding.shape
        if model_type == 'protein':
            if self.configs.prot2token_model.protein_encoder.drop_positional_encoding and self.positional_encoding_type == 'learned':
                embedding = self.protein_encoder_pos_drop(embedding + self.protein_encoder_pos_embed)
            elif self.configs.prot2token_model.protein_encoder.drop_positional_encoding and self.positional_encoding_type == 'absolute':
                embedding = self.absolute(embedding)
        elif model_type == 'molecule':
            if self.configs.prot2token_model.molecule_encoder.drop_positional_encoding and self.positional_encoding_type == 'learned':
                embedding = self.molecule_encoder_pos_drop(embedding + self.molecule_encoder_pos_embed)
            elif self.configs.prot2token_model.molecule_encoder.drop_positional_encoding and self.positional_encoding_type == 'absolute':
                embedding = self.absolute(embedding)
        elif model_type == 'decoder':
            if self.positional_encoding_type == 'learned':
                embedding = self.decoder_pos_drop(embedding + self.decoder_pos_embed[:, :seq_length, :])
            elif self.positional_encoding_type == 'absolute':
                embedding = self.absolute(embedding)
        return embedding

    def forward(self, protein_encoder_out, molecule_encoder_out, target_input):
        tgt_mask, tgt_padding_mask = create_mask(target_input, self.pad_idx, protein_encoder_out.device)
        tgt_embedding = self.embedding(target_input)

        # Drop positional encoding
        tgt_embedding = self.drop_positional_encoding(tgt_embedding, 'decoder')
        protein_encoder_out = self.drop_positional_encoding(protein_encoder_out, 'protein')
        if self.configs.prot2token_model.molecule_encoder.enable:
            molecule_encoder_out = self.drop_positional_encoding(molecule_encoder_out, 'molecule')

        protein_encoder_out = protein_encoder_out.transpose(0, 1)
        molecule_encoder_out = molecule_encoder_out.transpose(0, 1)

        if self.configs.prot2token_model.molecule_encoder.enable:
            # Concatenate the protein and molecule representations
            encoders_out = torch.cat([protein_encoder_out, molecule_encoder_out], dim=0)
        else:
            encoders_out = protein_encoder_out

        tgt_embedding = tgt_embedding.transpose(0, 1)

        preds = self.decoder(memory=encoders_out,
                             tgt=tgt_embedding,
                             tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_padding_mask)
        preds = preds.transpose(0, 1)
        return self.output(preds)

    def prediction(self, protein_encoder_out, molecule_encoder_out, tgt):
        # Initialize the generated sequence with the initial token
        predicted_sequence = tgt[..., :1]
        generated_tokens = tgt[..., :2]

        # Loop over the range of maximum sequence length
        for _ in range(self.max_len - 2):
            # Generate the next token using the `forward` method
            predicted_sequence = self(protein_encoder_out, molecule_encoder_out, generated_tokens)

            # Get the id of the token with the highest probability
            next_token_id = predicted_sequence.argmax(dim=-1)[..., -1:]

            # Concatenate the predicted token with the existing sequence
            generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)

            # If the predicted token is an `<eos>` token, break the loop
            if next_token_id.item() == self.decoder_tokenizer.tokens_dict['<eos>']:
                break

        padded_sequence = torch.zeros(predicted_sequence.shape[0],
                                      self.max_len - 1 - predicted_sequence.shape[1],
                                      predicted_sequence.shape[2]).to(predicted_sequence.device)
        padded_sequence[:, :, 0] = 1
        return torch.cat([predicted_sequence, padded_sequence], dim=1)

    def inference_greedy(self, protein_encoder_out, molecule_encoder_out, tgt):
        # Initialize the generated sequence with the initial token
        generated_tokens = tgt[..., :2]

        # Loop over the range of maximum sequence length
        for _ in range(self.max_len - 2):
            # Generate the next token using the `forward` method
            predicted_sequence = self(protein_encoder_out, molecule_encoder_out, generated_tokens)

            # Get the id of the token with the highest probability
            next_token_id = predicted_sequence.argmax(dim=-1)[..., -1:]

            # Concatenate the predicted token with the existing sequence
            generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)

            # If the predicted token is an `<eos>` token, break the loop
            if next_token_id.item() == self.decoder_tokenizer.tokens_dict['<eos>']:
                break

        return generated_tokens

    def inference_beam_search(self, protein_encoder_out, molecule_encoder_out, tgt, beam_width=1, temperature=1.0,
                              top_k=1):
        # Initialize the beam with the initial token
        beam = [(tgt[..., :2], 0)]  # (sequence, cumulative log probability)

        for _ in range(self.max_len - 2):
            candidates = []
            for seq, score in beam:
                # Generate the next token probabilities using the `forward` method
                predicted_sequence = self(protein_encoder_out, molecule_encoder_out, seq)
                next_token_probs = predicted_sequence[..., -1, :]

                # Apply temperature scaling
                next_token_probs = next_token_probs / temperature

                # Apply top-k sampling
                topk_probs, topk_indices = torch.topk(next_token_probs, top_k, dim=-1)
                topk_probs = F.log_softmax(topk_probs, dim=-1)

                # Sample from the top-k probabilities
                sampled_index = torch.multinomial(torch.exp(topk_probs), 1)
                next_token_id = topk_indices.gather(-1, sampled_index).squeeze(-1)

                for i in range(next_token_id.size(0)):
                    candidate_seq = torch.cat([seq[i:i + 1], next_token_id[i:i + 1].unsqueeze(0)], dim=-1)
                    candidate_score = score + topk_probs[i, sampled_index[i]].item()
                    candidates.append((candidate_seq, candidate_score))

            # Select the top `beam_width` candidates based on cumulative log probabilities
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # If any sequence ends with an `<eos>` token, break the loop
            if any(self.decoder_tokenizer.tokens_dict['<eos>'] in seq for seq, _ in beam):
                break

        # Return the best sequence from the beam
        best_sequence = max(beam, key=lambda x: x[1])[0]
        return best_sequence


class EncoderDecoder(nn.Module):
    def __init__(self, protein_encoder, molecule_encoder, decoder, encoder_tokenizer, molecule_encoder_tokenizer,
                 decoder_tokenizer, device, task_token, configs):
        super().__init__()
        self.protein_encoder = protein_encoder
        self.molecule_encoder = molecule_encoder
        self.decoder = decoder

        self.protein_encoder_tokenizer = encoder_tokenizer
        self.molecule_encoder_tokenizer = molecule_encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

        self.configs = configs
        self.inference_config = {
            "beam_width": 3,
            "temperature": 1.0,
            "top_k": 3
        }
        self.device = device
        self.task_token = task_token

        self.max_protein_encoder_length = configs.prot2token_model.protein_encoder.max_len
        self.max_molecule_encoder_length = configs.prot2token_model.molecule_encoder.max_len
        self.max_decoder_length = configs.prot2token_model.decoder.max_len

        self.dummy_representation = torch.zeros(1, self.configs.prot2token_model.molecule_encoder.max_len,
                                                self.decoder.dim)

    def forward(self, batch, mode=False, **kwargs):
        protein_encoder_out = self.protein_encoder(batch)
        if self.configs.prot2token_model.molecule_encoder.enable:
            molecule_encoder_out = self.molecule_encoder(batch)
        else:
            molecule_encoder_out = self.dummy_representation

        if mode == 'inference_greedy':
            preds = self.decoder.inference_greedy(protein_encoder_out, molecule_encoder_out, batch["target_input"])
        elif mode == 'inference_beam_search':
            preds = self.decoder.inference_beam_search(protein_encoder_out, molecule_encoder_out, batch["target_input"],
                                                       kwargs['inference_config']["beam_width"],
                                                       kwargs['inference_config']["temperature"],
                                                       kwargs['inference_config']["top_k"])
        else:
            preds = self.decoder(protein_encoder_out, molecule_encoder_out, batch["target_input"])

        return preds

    def prepare_sample(self, sequences, task_name):
        samples_list = []
        for sequence in sequences:
            sequence = [sequence]
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

                encoded_protein_sequence['input_ids'] = encoded_protein_sequence['input_ids']
                encoded_protein_sequence['attention_mask'] = encoded_protein_sequence['attention_mask']
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

                encoded_molecule_sequence['input_ids'] = encoded_molecule_sequence['input_ids']
                encoded_molecule_sequence['attention_mask'] = encoded_molecule_sequence['attention_mask']
            else:
                encoded_molecule_sequence = torch.zeros(1, 64, 320).long()

            encoded_target = torch.LongTensor(encoded_target).unsqueeze(0)
            samples_list.append(
                (encoded_protein_sequence, encoded_target, encoded_molecule_sequence, sequence, task_name))

        return samples_list

    def run(self, samples, progress_bar=True, inference_type='inference_greedy', merging_character=','):
        """
        Run the model on the given protein sequences.

        :param samples: A list of protein sequences to run the model on.
        :param progress_bar: Whether to show the progress bar or not.
        :param inference_type: The type of inference to use. It can be one of these: ['inference_greedy',
        'inference_beam_search'].
        :param merging_character: The character to use for merging the tokens at the end.
        :return: The results of the model on the given samples in the form of a list consist of the protein sequence,
        the task name, and the predicted sequence.
        """
        torch.cuda.empty_cache()
        data = self.prepare_sample(samples, task_name=self.task_token)

        counter = 0
        results = []
        self.eval()
        for i, sample in enumerate(tqdm(data, desc=f'inference', total=len(samples),
                                        leave=False, disable=not progress_bar)):
            protein_sequence, target, molecule_sequence, sequence, task_name = sample

            batch = {"protein_sequence": protein_sequence.to(self.device),
                     "molecule_sequence": molecule_sequence.to(self.device),
                     "target_input": target.to(self.device)}

            with torch.inference_mode():
                preds = self(batch, mode=inference_type, inference_config=self.inference_config)
                preds = preds.detach().cpu().numpy().tolist()[0]
                preds = [self.decoder_tokenizer.index_token_dict[pred] for pred in preds[2:-1]]
                preds = self.postprocessing(preds, merging_character)
                results.append([sequence[0], task_name, preds])
            counter += 1

        return results

    def postprocessing(self, preds, merging_character):
        if self.task_token == "<task_fluorescence>":
            # Convert the predicted tokens to the fluorescence value
            preds = self.denormalize_labels(
                float("".join(preds)),
                original_min=1.283, original_max=4.123, new_min=0.0001, new_max=0.9999
            )
            preds = list(str(preds))
        elif self.task_token == "<task_stability>":
            # Convert the predicted tokens to the stability value
            preds = self.denormalize_labels(
                float("".join(preds)),
                original_min=-1.97, original_max=3.4, new_min=0.0001, new_max=0.9999
            )
            preds = list(str(preds))
        preds = merging_character.join(preds)
        return preds

    @staticmethod
    def denormalize_labels(mapped_label, original_min=1.283, original_max=4.123, new_min=0.0001, new_max=0.9999):
        # Linearly map the mapped label back to the original range
        mapped_label = ((mapped_label - new_min) / (new_max - new_min)) * (original_max - original_min) + original_min
        mapped_label = round(mapped_label, 4)
        return mapped_label


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


def prepare_models(name, device, compile_model=False):
    """
    Prepare the encoder, decoder, and the encoder-decoder model.

    Args:
        name: The name of the prot2token task. It can be one of these: ['stability', 'fluorescence'].
        device: The device to use for prediction.
        compile_model: Whether to compile the model or not.

    Returns:
        The prot2token model.
    """

    encoder_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    molecule_encoder_tokenizer = AutoTokenizer.from_pretrained("gayane/BARTSmiles",
                                                               add_prefix_space=True)
    molecule_encoder_tokenizer.pad_token = '<pad>'
    molecule_encoder_tokenizer.bos_token = '<s>'
    molecule_encoder_tokenizer.eos_token = '</s>'
    molecule_encoder_tokenizer.mask_token = '<unk>'

    configs, checkpoint_path, decoder_tokenizer_dict = prepare_config_and_checkpoint(name)

    decoder_tokenizer = InferenceTokenizer(decoder_tokenizer_dict)

    # Prepare the protein encoder.
    protein_encoder = ProteinEncoder(
        model_name=configs.prot2token_model.protein_encoder.model_name,
        out_dim=configs.prot2token_model.decoder.dimension,
        configs=configs,
    )

    # freeze all parameters
    for param in protein_encoder.parameters():
        param.requires_grad = False

    if configs.prot2token_model.molecule_encoder.enable:
        # Prepare the molecule encoder.
        molecule_encoder = MoleculeEncoder(model_name=configs.prot2token_model.molecule_encoder.model_name,
                                           out_dim=configs.prot2token_model.decoder.dimension,
                                           configs=configs,
                                           )
        # freeze all parameters
        for param in molecule_encoder.parameters():
            param.requires_grad = False

    else:
        molecule_encoder = None

    # Prepare the decoder.
    decoder = Decoder(vocab_size=decoder_tokenizer.vocab_size,
                      dim=configs.prot2token_model.decoder.dimension,
                      num_heads=configs.prot2token_model.decoder.num_heads,
                      num_layers=configs.prot2token_model.decoder.num_layers,
                      max_len=configs.prot2token_model.decoder.max_len,
                      pad_idx=decoder_tokenizer.tokens_dict['<pad>'],
                      dim_feedforward=configs.prot2token_model.decoder.dim_feedforward,
                      activation_function=configs.prot2token_model.decoder.activation_function,
                      decoder_tokenizer=decoder_tokenizer,
                      configs=configs)

    # freeze all parameters
    for param in decoder.parameters():
        param.requires_grad = False

    task_token_dict = {'stability': '<task_stability>',
                       'fluorescence': '<task_fluorescence>'}
    # Prepare the encoder-decoder model.
    final_model = EncoderDecoder(protein_encoder, molecule_encoder, decoder,
                                 encoder_tokenizer,
                                 molecule_encoder_tokenizer,
                                 decoder_tokenizer,
                                 device,
                                 task_token=task_token_dict[name],
                                 configs=configs)

    # freeze all parameters
    for param in final_model.parameters():
        param.requires_grad = False

    final_model = load_checkpoints_inference(checkpoint_path, final_model)

    final_model = final_model.to(device)

    if compile_model:
        final_model = torch.compile(final_model)

    return final_model


if __name__ == '__main__':
    print('Testing the model')
