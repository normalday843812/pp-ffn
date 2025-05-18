import argparse
import os
import sys
import gc
import json
import random
import re
import yaml
import time
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_scheduler
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from tqdm import tqdm

from .config_loader import load_config, SimpleConfig
from .dataset_utils import (
    load_and_tokenize_raw_data,
    format_sample_for_training,
    DataCollatorForPadding,
    DEFAULT_LABEL_PAD_TOKEN_ID
)
from .model_ppffn import GPT2LMHeadModelWithPPFFN

if TYPE_CHECKING:
    import wandb

try:
    import wandb as actual_wandb
    WANDB_IMPORTED = True
except ImportError:
    WANDB_IMPORTED = False
    actual_wandb = None
    print("INFO: wandb not installed. Proceeding without W&B logging.")


def set_seed(seed_value: int):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def cleanup_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_dataset_length_estimate(file_path: Optional[str]) -> int:
    if not file_path or not os.path.exists(file_path):
        return 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

class LatentStepModel(PreTrainedModel, GenerationMixin):
    main_input_name = "input_ids"

    def __init__(self,
                 base_causallm: PreTrainedModel,
                 latent_token_id: int,
                 eos_token_id: int):
        super().__init__(base_causallm.config)
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id

        if hasattr(self.base_causallm, 'get_input_embeddings'):
            self.embedding_layer = self.base_causallm.get_input_embeddings()
        elif hasattr(self.base_causallm, 'transformer') and hasattr(self.base_causallm.transformer, 'wte'): # GPT-2 specific
            self.embedding_layer = self.base_causallm.transformer.wte
        else:
            raise ValueError("Cannot reliably get input embeddings from the base model for LatentStepModel.")

    @property
    def device(self) -> torch.device:
        return self.base_causallm.device

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding_layer

    def set_input_embeddings(self, value: nn.Embedding):
        if hasattr(self.base_causallm, 'set_input_embeddings'):
            self.base_causallm.set_input_embeddings(value)
        elif hasattr(self.base_causallm, 'transformer') and hasattr(self.base_causallm.transformer, 'wte'): # GPT-2 specific
             self.base_causallm.transformer.wte = value
        self.embedding_layer = value

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, **kwargs) -> dict:
        model_inputs = self.base_causallm.prepare_inputs_for_generation(input_ids, past_key_values, **kwargs)
        return model_inputs

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                use_cache: Optional[bool] = None,
                **kwargs) -> CausalLMOutputWithPast:

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if past_key_values is not None:
            return self.base_causallm(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, use_cache=use_cache, labels=labels
            )

        batch_size, seq_len = input_ids.shape
        current_inputs_embeds = self.embedding_layer(input_ids)
        
        latent_indices_flat = (input_ids == self.latent_token_id).nonzero(as_tuple=False)
        
        latent_lists_by_instance = [[] for _ in range(batch_size)]
        for i in range(latent_indices_flat.shape[0]):
            batch_idx = latent_indices_flat[i, 0].item()
            token_idx = latent_indices_flat[i, 1].item()
            latent_lists_by_instance[batch_idx].append(token_idx)
        
        for b_idx in range(batch_size):
            latent_lists_by_instance[b_idx].sort()

        max_latents_in_any_instance = max(len(l) for l in latent_lists_by_instance) if any(latent_lists_by_instance) else 0

        collected_logits_segments: List[torch.Tensor] = []
        current_kv_cache_for_passes: Optional[Tuple[Tuple[torch.Tensor]]] = None
        hidden_states_kv_offset = 0

        for pass_idx in range(max_latents_in_any_instance + 1):
            segment_start_original_idx = hidden_states_kv_offset
            segment_end_original_idx = seq_len

            if pass_idx < max_latents_in_any_instance:
                min_latent_pos_this_pass = seq_len
                has_latents_this_pass_globally = False
                for b_idx_instance in range(batch_size):
                    if pass_idx < len(latent_lists_by_instance[b_idx_instance]):
                        min_latent_pos_this_pass = min(min_latent_pos_this_pass, latent_lists_by_instance[b_idx_instance][pass_idx])
                        has_latents_this_pass_globally = True
                
                segment_end_original_idx = min_latent_pos_this_pass + 1 if has_latents_this_pass_globally else segment_start_original_idx


            if segment_start_original_idx >= segment_end_original_idx:
                if pass_idx >= max_latents_in_any_instance and segment_start_original_idx >= seq_len:
                    break 
                else:
                    continue 

            segment_inputs_embeds = current_inputs_embeds[:, segment_start_original_idx:segment_end_original_idx, :]
            
            current_segment_attention_mask = attention_mask[:, :segment_end_original_idx] if attention_mask is not None else None
            
            current_segment_position_ids = None
            if position_ids is not None:
                current_segment_position_ids = position_ids[:, segment_start_original_idx:segment_end_original_idx]

            outputs = self.base_causallm(
                inputs_embeds=segment_inputs_embeds,
                attention_mask=current_segment_attention_mask, 
                position_ids=current_segment_position_ids,
                past_key_values=current_kv_cache_for_passes,
                output_hidden_states=True,
                use_cache=True
            )

            collected_logits_segments.append(outputs.logits)
            if use_cache:
                current_kv_cache_for_passes = outputs.past_key_values
            else:
                current_kv_cache_for_passes = None

            if pass_idx < max_latents_in_any_instance:
                hidden_states_from_output = outputs.hidden_states[-1]
                
                for b_idx in range(batch_size):
                    if pass_idx < len(latent_lists_by_instance[b_idx]):
                        latent_token_original_pos = latent_lists_by_instance[b_idx][pass_idx]
                        producing_token_original_pos = latent_token_original_pos - 1 
                        
                        if producing_token_original_pos >= 0:
                            idx_in_segment = producing_token_original_pos - segment_start_original_idx
                            
                            if 0 <= idx_in_segment < hidden_states_from_output.shape[1]:
                                current_inputs_embeds[b_idx, latent_token_original_pos, :] = \
                                    hidden_states_from_output[b_idx, idx_in_segment, :]
            
            hidden_states_kv_offset = segment_end_original_idx
            if hidden_states_kv_offset >= seq_len:
                break
        
        if collected_logits_segments:
            final_logits = torch.cat(collected_logits_segments, dim=1)
            if final_logits.shape[1] < seq_len:
                 padding_needed = seq_len - final_logits.shape[1]
                 padding_tensor = torch.full((batch_size, padding_needed, final_logits.shape[2]), -1e9,
                                             device=final_logits.device, dtype=final_logits.dtype)
                 final_logits = torch.cat([final_logits, padding_tensor], dim=1)
        elif seq_len > 0 :
            vocab_size = self.config.vocab_size
            final_logits = torch.full((batch_size, seq_len, vocab_size), -1e9,
                                      device=input_ids.device, dtype=current_inputs_embeds.dtype)
        else: 
            final_logits = torch.empty((batch_size, 0, self.config.vocab_size), 
                                       device=input_ids.device, dtype=current_inputs_embeds.dtype)


        loss = None
        if labels is not None and final_logits.numel() > 0 and final_logits.shape[1] > 1:
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        elif labels is not None:
            loss = torch.tensor(0.0, device=final_logits.device, dtype=final_logits.dtype)


        if not use_cache:
            current_kv_cache_for_passes = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=final_logits,
            past_key_values=current_kv_cache_for_passes,
            hidden_states=None,
            attentions=None
        )

def parse_generated_answer(generated_text: str, dataset_name: Optional[str]) -> str:
    dataset_name_lower = dataset_name.lower() if dataset_name else "unknown"
    boxed_match = re.search(r"boxed\{(.+?)\}", generated_text)
    if boxed_match: return boxed_match.group(1).replace(",", "").strip()
    hash_match = re.search(r"####\s*([\-0-9\.\,]+)", generated_text)
    if hash_match: return hash_match.group(1).replace(",", "").strip()

    if "gsm8k" in dataset_name_lower:
        numbers = re.findall(r"([\-0-9\.\,]+)", generated_text)
        if numbers: return numbers[-1].replace(",", "").strip()

    if "prosqa" in dataset_name_lower:
        if "###" in generated_text:
            answer_part = generated_text.split("###")[-1].strip()
            return answer_part.splitlines()[0].strip() if answer_part else ""
        lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
        return lines[-1] if lines else generated_text.strip()

    return generated_text.strip()

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, tokenizer: PreTrainedTokenizerBase,
             device: torch.device, config: SimpleConfig, special_token_ids: Dict[str, int],
             epoch_num: int, wandb_run_obj: Optional[Any],
             use_amp: bool
             ) -> Tuple[float, float, float, float]:
    model.eval()
    total_eval_loss = 0.0
    all_parsed_predictions: List[str] = []
    all_ground_truths: List[str] = []
    eval_samples_logged_wandb = 0
    
    all_newly_generated_token_counts: List[int] = []
    total_inference_time_seconds: float = 0.0
    num_inference_samples: int = 0

    ground_truth_answers_map: Dict[int, str] = {}
    if config.val_path and os.path.exists(config.val_path):
        try:
            print(f"Loading ground truth answers for validation from: {config.val_path}")
            with open(config.val_path, 'r', encoding='utf-8') as f:
                val_data_list = json.load(f) 
                if not isinstance(val_data_list, list):
                    print(f"Warning: Validation data in {config.val_path} for ground truths is not a list. Accuracy calculation might be affected.")
                    val_data_list = [] 

                for i, sample in enumerate(val_data_list):
                    if not isinstance(sample, dict):
                        print(f"Warning: Item {i} in {config.val_path} is not a dictionary. Skipping.")
                        continue
                    original_idx = sample.get("idx", sample.get("original_idx", i))
                    answer = sample.get("answer")
                    if answer is not None:
                        ground_truth_answers_map[original_idx] = str(answer).strip().replace(",", "")
                    else:
                        print(f"Warning: Item {i} (idx: {original_idx}) in {config.val_path} is missing 'answer' field.")
        except json.JSONDecodeError as e:
            print(f"ERROR: Could not decode JSON from validation file {config.val_path} for ground truth answers. Accuracy will be 0. Details: {e}")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while loading ground truth answers from {config.val_path}. Accuracy will be 0. Details: {e}")
    if not ground_truth_answers_map:
        print("Warning: Ground truth answer map is empty. Validation accuracy will be 0.")

    val_progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num+1} Validation", dynamic_ncols=True, leave=False)
    for batch in val_progress_bar:
        original_indices_batch = batch.pop("original_idx", torch.tensor([-1], device=device))
        batch_on_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(**batch_on_device)
        if outputs.loss is not None: total_eval_loss += outputs.loss.item() * batch_on_device["input_ids"].shape[0]

        for i in range(batch_on_device["input_ids"].shape[0]):
            current_input_ids = batch_on_device["input_ids"][i:i+1]
            current_attention_mask = batch_on_device["attention_mask"][i:i+1]
            current_original_idx = original_indices_batch[i].item()

            if current_original_idx == -1 or current_original_idx not in ground_truth_answers_map:
                continue

            max_new_tokens_eval = int(config.get('eval_max_new_tokens', 128))
            gen_kwargs = {
                "temperature": float(config.get('generate_temperature', 0.1)),
                "top_p": float(config.get('generate_top_p', 0.95)),
                "do_sample": bool(config.get('generate_do_sample', False)),
                "num_beams": int(config.get('generate_num_beams', 1)),
            }
            if not gen_kwargs["do_sample"]:
                gen_kwargs.pop("temperature", None); gen_kwargs.pop("top_p", None)
            
            pad_token_id_for_generate = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else special_token_ids['eos_token_id']

            inference_start_time = time.time()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                generated_ids = model.generate(
                    input_ids=current_input_ids, 
                    attention_mask=current_attention_mask,
                    max_new_tokens=max_new_tokens_eval,
                    eos_token_id=special_token_ids['eos_token_id'],
                    pad_token_id=pad_token_id_for_generate,
                    **gen_kwargs
                )
            inference_end_time = time.time()
            total_inference_time_seconds += (inference_end_time - inference_start_time)
            num_inference_samples += 1

            prompt_len = current_input_ids.shape[1]
            generated_tokens_only = generated_ids[0, prompt_len:] 
            num_newly_generated = len(generated_tokens_only)
            all_newly_generated_token_counts.append(num_newly_generated)

            generated_text = tokenizer.decode(generated_tokens_only, skip_special_tokens=True)
            parsed_prediction = parse_generated_answer(generated_text, config.dataset_name)
            all_parsed_predictions.append(parsed_prediction)

            ground_truth_ans = ground_truth_answers_map.get(current_original_idx)
            all_ground_truths.append(ground_truth_ans)
            if WANDB_IMPORTED and wandb_run_obj and eval_samples_logged_wandb < int(config.get('wandb_log_eval_samples', 3)):
                log_entry = {
                    f"eval_epoch_{epoch_num+1}/sample_{current_original_idx}_prompt": tokenizer.decode(current_input_ids[0], skip_special_tokens=False),
                    f"eval_epoch_{epoch_num+1}/sample_{current_original_idx}_generated_full": tokenizer.decode(generated_ids[0], skip_special_tokens=True),
                    f"eval_epoch_{epoch_num+1}/sample_{current_original_idx}_generated_answer_part": generated_text,
                    f"eval_epoch_{epoch_num+1}/sample_{current_original_idx}_parsed_prediction": parsed_prediction,
                    f"eval_epoch_{epoch_num+1}/sample_{current_original_idx}_ground_truth": ground_truth_ans
                }
                wandb_run_obj.log(log_entry)
                eval_samples_logged_wandb += 1
    
    num_eval_batches = len(dataloader)
    avg_val_loss = total_eval_loss / num_eval_batches if num_eval_batches > 0 else float('inf')

    accuracy = 0.0
    if all_parsed_predictions and len(all_parsed_predictions) == len(all_ground_truths):
        correct_predictions = sum(1 for pred, gt in zip(all_parsed_predictions, all_ground_truths) if pred == gt)
        accuracy = correct_predictions / len(all_parsed_predictions) if len(all_parsed_predictions) > 0 else 0.0
    
    avg_newly_generated_tokens = sum(all_newly_generated_token_counts) / len(all_newly_generated_token_counts) if all_newly_generated_token_counts else 0.0
    avg_inference_time_per_sample = total_inference_time_seconds / num_inference_samples if num_inference_samples > 0 else 0.0

    print(f"Epoch {epoch_num+1} Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}")
    print(f"Epoch {epoch_num+1} Avg Newly Generated Tokens: {avg_newly_generated_tokens:.2f}")
    print(f"Epoch {epoch_num+1} Avg Inference Time per Sample: {avg_inference_time_per_sample:.4f} seconds")

    model.train()
    return avg_val_loss, accuracy, avg_newly_generated_tokens, avg_inference_time_per_sample


def main():
    parser = argparse.ArgumentParser(description="Latent Step Reasoning Model Training Script")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        config = load_config(args.config_file)
    except FileNotFoundError: print(f"ERROR: Configuration file not found: {args.config_file}"); sys.exit(1)
    except yaml.YAMLError: print(f"ERROR: Invalid YAML in configuration file: {args.config_file}"); sys.exit(1)

    set_seed(int(config.get('seed', 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if str(device) == "cuda": print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    use_mixed_precision = bool(config.get('use_mixed_precision', True)) and torch.cuda.is_available()
    if use_mixed_precision:
        print("INFO: Using mixed precision training")
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)

    experiment_save_dir = os.path.join(config.get('save_path', 'results'), config.experiment_name)
    os.makedirs(experiment_save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=bool(config.get('trust_remote_code', False)))
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    if tokenizer.pad_token_id is None: 
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Tokenizer '{config.model_id}' missing pad_token_id. Set to eos_token_id: {tokenizer.eos_token_id}")

    special_tokens_to_add = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
    if num_added_toks > 0: print(f"Added {num_added_toks} new special tokens to tokenizer.")

    special_token_ids = {
        'start_latent_id': tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        'latent_token_id': tokenizer.convert_tokens_to_ids("<|latent|>"),
        'end_latent_id': tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        'eos_token_id': tokenizer.eos_token_id
    }

    base_hf_model: PreTrainedModel
    ppffn_num_paths_config_val = config.get('ppffn_num_paths')
    num_paths_for_ppffn = 0 

    if ppffn_num_paths_config_val is not None:
        try:
            num_paths_for_ppffn = int(ppffn_num_paths_config_val)
            if num_paths_for_ppffn < 0: 
                print(f"Warning: 'ppffn_num_paths' value '{ppffn_num_paths_config_val}' is negative. Defaulting to 0 paths (no PP-FFN).")
                num_paths_for_ppffn = 0
        except ValueError:
            print(f"Warning: 'ppffn_num_paths' value '{ppffn_num_paths_config_val}' from config is not a valid integer. Defaulting to 0 paths (no PP-FFN).")
            num_paths_for_ppffn = 0 
    
    should_load_ppffn = False
    if num_paths_for_ppffn > 0:
        should_load_ppffn = True
        print(f"Info: 'ppffn_num_paths' is {num_paths_for_ppffn}. Attempting to load PP-FFN model.")
    else:
        print(f"Info: 'ppffn_num_paths' is {num_paths_for_ppffn} (or was None/invalid). Standard Hugging Face model will be loaded (PP-FFN not configured).")

    if should_load_ppffn:
        print(f"Initializing custom PP-FFN model based on {config.model_id} with {num_paths_for_ppffn} paths...")
        try:
            base_hf_model = GPT2LMHeadModelWithPPFFN.from_pretrained_with_ppffn(
                config.model_id, 
                ppffn_simple_config=config, 
                trust_remote_code=bool(config.get('trust_remote_code', False))
            )
            print("Successfully initialized GPT2LMHeadModelWithPPFFN.")
        except Exception as e:
            print(f"ERROR: Failed to initialize GPT2LMHeadModelWithPPFFN: {e}")
            print("Falling back to standard Hugging Face model.")
            base_hf_model = AutoModelForCausalLM.from_pretrained(
                config.model_id, 
                trust_remote_code=bool(config.get('trust_remote_code', False))
            )
    else:
        print(f"Initializing base Hugging Face model: {config.model_id}.")
        base_hf_model = AutoModelForCausalLM.from_pretrained(
            config.model_id, 
            trust_remote_code=bool(config.get('trust_remote_code', False))
        )

    base_hf_model.resize_token_embeddings(len(tokenizer))
    model_to_train: nn.Module = base_hf_model

    if bool(config.get('use_latent_loop', False)):
        print(f"Wrapping model with LatentStepModel for iterative reasoning.")
        model_to_train = LatentStepModel(
            base_causallm=base_hf_model, 
            latent_token_id=special_token_ids['latent_token_id'],
            eos_token_id=special_token_ids['eos_token_id']
        )
    model_to_train.to(device)

    total_params = sum(p.numel() for p in model_to_train.parameters() if p.requires_grad)
    active_params = sum(p.numel() for p in model_to_train.parameters() if p.requires_grad and p.is_floating_point())
    print(f"Experiment: {config.experiment_name}")
    print(f"  Total Trainable Parameters: {total_params:,}")
    print(f"  Active Floating Point Parameters: {active_params:,}")

    optimizer_lr = float(config.lr) 
    optimizer_weight_decay = float(config.get('weight_decay', 0.01)) 

    optimizer = optim.AdamW(model_to_train.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)
    lr_scheduler = None
    if bool(config.get('use_lr_scheduler', False)):
        estimated_train_len = get_dataset_length_estimate(config.train_path)
        if estimated_train_len > 0 :
            max_samples_cfg = config.get('max_train_samples') 
            num_effective_samples = min(estimated_train_len, int(max_samples_cfg) if max_samples_cfg is not None else estimated_train_len)
            batch_size_train = int(config.batch_size_training)
            grad_accum = int(config.get('gradient_accumulation_steps', 1))
            steps_per_epoch_raw = num_effective_samples / batch_size_train
            num_optimizer_steps_per_epoch = steps_per_epoch_raw / grad_accum
            num_optimizer_steps_per_epoch = int(num_optimizer_steps_per_epoch) + (1 if num_optimizer_steps_per_epoch % 1 != 0 else 0) 
            total_training_steps = int(config.num_epochs) * num_optimizer_steps_per_epoch
            if total_training_steps > 0:
                lr_scheduler = get_scheduler(
                    name=config.get('lr_scheduler_type', "linear"), optimizer=optimizer,
                    num_warmup_steps=int(config.get('num_warmup_steps', 0)), num_training_steps=total_training_steps
                )
                print(f"Initialized LR scheduler: type={config.get('lr_scheduler_type', 'linear')}, total_steps={total_training_steps} (based on ~{num_effective_samples} samples)")
            else: print("Warning: Total training steps for LR scheduler is 0. Not using scheduler.")
        else: print("Warning: Could not estimate train dataset length for LR scheduler. Not using scheduler.")

    base_train_dataset, base_val_dataset = None, None
    try:
        print("Loading and tokenizing datasets...")
        base_train_dataset = load_and_tokenize_raw_data(config.train_path, tokenizer, config, config.get('max_train_samples'))
        if config.val_path: 
            base_val_dataset = load_and_tokenize_raw_data(config.val_path, tokenizer, config, config.get('max_val_samples'))
    except json.JSONDecodeError as e:
        print(f"FATAL: Failed to decode JSON from a data file. Please check the format of your data files (e.g., {config.train_path}).")
        print(f"Error details: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"FATAL: Data file not found: {e.filename}. Please check paths in your config.")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: An unexpected error occurred during data loading: {e}")
        sys.exit(1)

    if base_train_dataset is None or not hasattr(base_train_dataset, 'num_rows') or base_train_dataset.num_rows == 0:
        print(f"FATAL: Training dataset from '{config.train_path}' is empty or could not be loaded correctly.")
        sys.exit(1)
    if config.val_path and (base_val_dataset is None or not hasattr(base_val_dataset, 'num_rows') or base_val_dataset.num_rows == 0):
        print(f"Warning: Validation dataset from '{config.val_path}' is empty or could not be loaded. Validation will be skipped.")
        base_val_dataset = None 

    data_collator = DataCollatorForPadding(tokenizer, label_pad_token_id=DEFAULT_LABEL_PAD_TOKEN_ID)

    wandb_run_obj: Optional['wandb.sdk.wandb_run.Run'] = None
    if WANDB_IMPORTED and not bool(config.get('debug', False)) and bool(config.get('use_wandb', True)):
        try:
            wandb_config_log = config.to_dict()
            wandb_config_log["trainable_parameters_total"] = total_params
            wandb_config_log["trainable_parameters_active_fp"] = active_params
            wandb_config_log["use_mixed_precision"] = use_mixed_precision
            wandb_run_obj = actual_wandb.init(
                project=config.get('wandb_project', 'default-project'), 
                name=config.experiment_name, 
                config=wandb_config_log
            )
        except Exception as e: print(f"Failed to initialize wandb: {e}. Proceeding without wandb logging.")

    print(f"\n--- Starting Training: {config.experiment_name} ---")
    metric_for_best = config.get('metric_for_best_model', 'accuracy')
    metric_mode = 'max' if metric_for_best == 'accuracy' else 'min'
    best_val_metric = -float('inf') if metric_mode == 'max' else float('inf')

    global_optimizer_step_count = 0
    num_epochs = int(config.num_epochs)
    grad_accum_steps = int(config.get('gradient_accumulation_steps', 1))

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model_to_train.train()
        current_epoch_stage = 0
        if bool(config.get('use_latent_loop', False)):
            current_epoch_stage = epoch // int(config.get('epochs_per_stage', 1))
            max_stage_cfg = config.get('max_latent_stage')
            if max_stage_cfg is not None:
                 current_epoch_stage = min(current_epoch_stage, int(max_stage_cfg))

        print(f"\nEpoch {epoch+1}/{num_epochs}, Training Stage (if applicable): {current_epoch_stage if bool(config.get('use_latent_loop')) else 'N/A (SFT)'}")
        fn_kwargs_for_formatting = {"config": config, "scheduled_stage": current_epoch_stage, "special_token_ids": special_token_ids}

        num_map_processors = os.cpu_count() if not bool(config.get('debug', False)) and os.cpu_count() and os.cpu_count() > 1 else 1
        formatted_train_dataset = base_train_dataset.map(
            format_sample_for_training, fn_kwargs=fn_kwargs_for_formatting,
            load_from_cache_file=False, num_proc=num_map_processors, desc=f"Formatting train data for epoch {epoch+1}"
        )

        train_dataloader = DataLoader(
            formatted_train_dataset, batch_size=int(config.batch_size_training),
            collate_fn=data_collator, shuffle=True, num_workers=int(config.get('num_workers', 0)), pin_memory=True
        )

        epoch_total_train_loss = 0.0
        optimizer.zero_grad()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train", dynamic_ncols=True, leave=False)
        for step_in_epoch, batch in enumerate(progress_bar):
            batch.pop("original_idx", None) 
            batch_on_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_mixed_precision):
                outputs = model_to_train(**batch_on_device)
                loss = outputs.loss

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Epoch {epoch+1}, Step {step_in_epoch+1}: Invalid loss ({loss}). Skipping batch.")
                cleanup_gpu_memory()
                scaler.update()
                optimizer.zero_grad()
                continue

            current_batch_loss_value = loss.item()
            loss_to_backward = loss
            if grad_accum_steps > 1: loss_to_backward = loss / grad_accum_steps
            
            scaler.scale(loss_to_backward).backward()
            epoch_total_train_loss += current_batch_loss_value

            if (step_in_epoch + 1) % grad_accum_steps == 0 or (step_in_epoch + 1) == len(train_dataloader):
                max_grad_norm_cfg = config.get('max_grad_norm')
                if max_grad_norm_cfg is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_to_train.parameters(), float(max_grad_norm_cfg))
                
                scaler.step(optimizer)
                scaler.update()
                
                if lr_scheduler: lr_scheduler.step()
                optimizer.zero_grad()
                global_optimizer_step_count +=1

                if WANDB_IMPORTED and wandb_run_obj:
                    wandb_run_obj.log({
                        "train/step_loss": current_batch_loss_value,
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "global_step": global_optimizer_step_count
                    })
            progress_bar.set_postfix({"loss": f"{current_batch_loss_value:.4f}"})

        avg_epoch_train_loss = epoch_total_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else float('inf')
        print(f"Epoch {epoch+1} Avg Train Loss (per batch): {avg_epoch_train_loss:.4f}")
        
        epoch_end_time = time.time()
        epoch_duration_seconds = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1} Training Duration: {epoch_duration_seconds:.2f} seconds")

        if WANDB_IMPORTED and wandb_run_obj: 
            wandb_log_data = {
                "train/epoch_loss": avg_epoch_train_loss, 
                "train/epoch_duration_seconds": epoch_duration_seconds,
                "epoch": epoch + 1
            }

        if base_val_dataset is not None and base_val_dataset.num_rows > 0:
            num_val_map_processors = min(4, os.cpu_count() if os.cpu_count() else 1) 
            formatted_val_dataset = base_val_dataset.map(
                format_sample_for_training, fn_kwargs=fn_kwargs_for_formatting,
                load_from_cache_file=False, num_proc=num_val_map_processors, desc=f"Formatting val data for epoch {epoch+1}"
            )
            val_dataloader = DataLoader(
                formatted_val_dataset, batch_size=int(config.get('batch_size_eval', 1)),
                collate_fn=data_collator, num_workers=int(config.get('num_workers', 0))
            )
            avg_val_loss, val_accuracy, avg_newly_generated_tokens, avg_inference_time_per_sample = evaluate(
                model_to_train, val_dataloader, tokenizer, device, config, special_token_ids, epoch, wandb_run_obj, use_mixed_precision # Pass use_amp
            )
            
            if WANDB_IMPORTED and wandb_run_obj:
                wandb_log_data.update({
                    "eval/epoch_loss": avg_val_loss, 
                    "eval/accuracy": val_accuracy, 
                    "eval/avg_newly_generated_tokens": avg_newly_generated_tokens,
                    "eval/avg_inference_time_sample_seconds": avg_inference_time_per_sample,
                    "epoch_completed": epoch + 1
                })

            current_metric_for_saving = val_accuracy if metric_mode == 'max' else avg_val_loss
            improved = (current_metric_for_saving > best_val_metric) if metric_mode == 'max' else (current_metric_for_saving < best_val_metric)

            if not bool(config.get('debug', False)):
                save_checkpoint_this_epoch = False
                if bool(config.get('save_only_improve', True)):
                    if improved:
                        print(f"Val metric ({metric_for_best}) improved from {best_val_metric:.4f} to {current_metric_for_saving:.4f}. Saving model.")
                        best_val_metric = current_metric_for_saving
                        save_checkpoint_this_epoch = True
                else: 
                    save_checkpoint_this_epoch = True
                    if improved: best_val_metric = current_metric_for_saving

                if save_checkpoint_this_epoch:
                    checkpoint_name = f"checkpoint_epoch_{epoch+1}_val_{metric_for_best}_{current_metric_for_saving:.4f}"
                    checkpoint_path = os.path.join(experiment_save_dir, checkpoint_name)
                    
                    model_to_save = model_to_train.base_causallm if isinstance(model_to_train, LatentStepModel) else model_to_train
                    
                    model_to_save.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    try:
                        config_to_save = config.to_dict()
                        config_to_save['use_mixed_precision_effective'] = use_mixed_precision
                        with open(os.path.join(checkpoint_path, 'training_config.yaml'), 'w') as f_cfg:
                            yaml.dump(config_to_save, f_cfg, sort_keys=False)
                    except Exception as e_cfg: print(f"Warning: Could not save training config to checkpoint: {e_cfg}")
                    print(f"Saved checkpoint to {checkpoint_path}")
        else: 
            print(f"Epoch {epoch+1}: Validation dataset is empty or not provided. Skipping evaluation and checkpointing.")
        
        if WANDB_IMPORTED and wandb_run_obj:
            if "eval/epoch_loss" not in wandb_log_data: 
                 wandb_log_data["epoch_completed"] = epoch + 1
            wandb_run_obj.log(wandb_log_data)


        cleanup_gpu_memory()

    if WANDB_IMPORTED and wandb_run_obj : wandb_run_obj.finish()
    print(f"--- Training Finished: {config.experiment_name} ---")

if __name__ == "__main__":
    main()