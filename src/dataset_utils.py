# src/dataset_utils.py
import json
import os
import random
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union

import torch
from datasets import Dataset # type: ignore
from transformers import PreTrainedTokenizerBase

DEFAULT_LABEL_PAD_TOKEN_ID = -100
IGNORE_INDEX = -100

EXPECTED_FORMATTED_COLUMNS = ["input_ids", "attention_mask", "labels", "position_ids", "original_idx"]


def load_and_tokenize_raw_data(
    file_path: Optional[str],
    tokenizer: PreTrainedTokenizerBase,
    config: Any,
    max_samples: Optional[int] = None
) -> Optional[Dataset]:
    if not file_path or not os.path.exists(file_path):
        if file_path:
            print(f"Info: Data file not found: {file_path}. Returning None.")
        return None

    raw_data_list = []
    print(f"Loading raw data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_json = json.load(f)
            if isinstance(loaded_json, list):
                raw_data_list = loaded_json
            else:
                print(f"Error: Unsupported JSON structure in {file_path}. Expected a list of objects.")
                # Return an empty dataset with at least 'original_idx' for downstream checks
                return Dataset.from_dict({"original_idx": []})


        if max_samples is not None and max_samples > 0:
            raw_data_list = raw_data_list[:max_samples]

        if not raw_data_list:
            print(f"Info: No data loaded from {file_path}. Returning empty dataset.")
            return Dataset.from_dict({"original_idx": []})

        for i, item in enumerate(raw_data_list):
            item['original_idx'] = item.get('idx', i)

    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from file {file_path}. Details: {e}")
        return Dataset.from_dict({"original_idx": []})
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading file {file_path}: {e}")
        return Dataset.from_dict({"original_idx": []})

    processed_samples = []
    for i, item in enumerate(raw_data_list):
        processed_samples.append({
            "question": str(item.get("question", "")),
            "steps": [str(s) for s in item.get("steps", [])],
            "answer": str(item.get("answer", "")),
            "original_idx": item.get("original_idx", i)
        })

    if not processed_samples: # Should be caught by raw_data_list check, but as a safeguard
        print(f"Info: No valid samples processed from {file_path}. Returning empty dataset.")
        return Dataset.from_dict({"original_idx": []})
        
    hf_dataset = Dataset.from_list(processed_samples)

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        question_tokens = tokenizer(
            [q + "\n" for q in examples["question"]], add_special_tokens=True
        ).input_ids

        steps_tokens_list = []
        for steps_for_sample in examples["steps"]:
            steps_tokens_list.append(
                [tokenizer.encode(s + "\n", add_special_tokens=False) for s in steps_for_sample]
            )

        answer_format = config.get("answer_format_prefix", "### ")
        answer_tokens = [
            tokenizer.encode(answer_format + ans, add_special_tokens=False) + [tokenizer.eos_token_id]
            for ans in examples["answer"]
        ]

        return {
            "question_tokens": question_tokens,
            "steps_tokens": steps_tokens_list,
            "answer_tokens": answer_tokens,
        }
    
    num_map_proc_config = config.get('num_map_processors', None)
    if num_map_proc_config is None: # Default if not in config
        num_map_proc = os.cpu_count() if os.cpu_count() and os.cpu_count() > 1 else 1
    else:
        num_map_proc = int(num_map_proc_config)


    tokenized_dataset = hf_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["question", "steps", "answer"],
        num_proc=num_map_proc,
        desc="Tokenizing raw data"
    )
    print(f"Tokenized dataset from {file_path} has {len(tokenized_dataset)} samples.")
    return tokenized_dataset

def format_sample_for_training(
    sample: Dict[str, Any],
    config: Any,
    scheduled_stage: int,
    special_token_ids: Dict[str, int]
) -> Dict[str, Any]:
    question_tokens = sample['question_tokens']
    steps_tokens_flat = list(itertools.chain.from_iterable(sample['steps_tokens']))
    answer_tokens = sample['answer_tokens']

    start_latent_id = special_token_ids['start_latent_id']
    latent_token_id = special_token_ids['latent_token_id']
    end_latent_id = special_token_ids['end_latent_id']

    input_ids = []
    labels = []

    if not bool(config.get('use_latent_loop', False)):
        if bool(config.get('no_cot', False)):
            input_ids = question_tokens + answer_tokens
            labels = [IGNORE_INDEX] * len(question_tokens) + answer_tokens
        else:
            input_ids = question_tokens + steps_tokens_flat + answer_tokens
            labels = ([IGNORE_INDEX] * len(question_tokens)) + steps_tokens_flat + answer_tokens
    else:
        max_steps_in_sample = len(sample['steps_tokens'])
        num_latent_tokens_this_stage = scheduled_stage * int(config.get('c_thought', 1))

        if bool(config.get('no_special_latent_markers', False)):
            latent_segment = [latent_token_id] * num_latent_tokens_this_stage
        else:
            latent_segment = [start_latent_id] + ([latent_token_id] * num_latent_tokens_this_stage) + [end_latent_id]

        n_skip_steps = scheduled_stage
        max_latent_stage_cfg = config.get('max_latent_stage')
        if max_latent_stage_cfg is not None and scheduled_stage > int(max_latent_stage_cfg):
            n_skip_steps = max_steps_in_sample
            if bool(config.get('pad_latent_to_max', False)):
                num_latent_tokens_fixed = int(config.get('max_latent_stage', 0)) * int(config.get('c_thought', 1))
                if bool(config.get('no_special_latent_markers', False)):
                    latent_segment = [latent_token_id] * num_latent_tokens_fixed
                else:
                    latent_segment = [start_latent_id] + ([latent_token_id] * num_latent_tokens_fixed) + [end_latent_id]

        steps_to_include_tokens = list(itertools.chain.from_iterable(sample['steps_tokens'][n_skip_steps:]))
        input_ids = question_tokens + latent_segment + steps_to_include_tokens + answer_tokens
        labels = ([IGNORE_INDEX] * len(question_tokens)) + \
                 ([IGNORE_INDEX] * len(latent_segment)) + \
                 steps_to_include_tokens + \
                 answer_tokens

    max_len = int(config.get('max_seq_length', 512))
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    
    attention_mask = [1] * len(input_ids)
    position_ids = list(range(len(input_ids)))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "position_ids": position_ids,
        "original_idx": sample['original_idx']
    }


@dataclass
class DataCollatorForPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = DEFAULT_LABEL_PAD_TOKEN_ID
    position_id_pad_value: int = 0 
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Explicitly extract the features we expect and will process
        # This makes the collator robust to extra columns being passed in the `features` list.
        
        batch_input_ids = [feature["input_ids"] for feature in features]
        batch_attention_mask = [feature["attention_mask"] for feature in features]
        batch_labels = [feature["labels"] for feature in features] if "labels" in features[0] else None
        batch_position_ids = [feature["position_ids"] for feature in features] if "position_ids" in features[0] else None
        batch_original_idx = [feature["original_idx"] for feature in features] if "original_idx" in features[0] else None

        # Use tokenizer.pad on a dictionary containing only the features it knows how to pad well.
        # tokenizer.pad expects a list of dicts, or a dict of lists.
        # We will prepare a dict of lists for input_ids and attention_mask.
        to_pad_by_tokenizer = {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}
        
        padded_by_tokenizer = self.tokenizer.pad(
            to_pad_by_tokenizer,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None, # Pad to Python lists first
        )

        sequence_length = len(padded_by_tokenizer["input_ids"][0])
        
        final_batch: Dict[str, Any] = {
            "input_ids": padded_by_tokenizer["input_ids"],
            "attention_mask": padded_by_tokenizer["attention_mask"]
        }

        if batch_labels is not None:
            padded_labels = []
            for label_sequence in batch_labels:
                difference = sequence_length - len(label_sequence)
                padded_labels.append(label_sequence + [self.label_pad_token_id] * difference)
            final_batch["labels"] = padded_labels
        
        if batch_position_ids is not None:
            padded_position_ids = []
            for pos_id_sequence in batch_position_ids:
                difference = sequence_length - len(pos_id_sequence)
                padded_position_ids.append(pos_id_sequence + [self.position_id_pad_value] * difference)
            final_batch["position_ids"] = padded_position_ids
        
        if self.return_tensors == "pt":
            for key, value in final_batch.items():
                if isinstance(value, list): # Ensure it's a list before tensor conversion
                    dtype = torch.long # Default to long for most ID-like fields
                    if key not in ["input_ids", "attention_mask", "labels", "position_ids"]:
                         # Heuristic for other potential fields if they were lists of numbers
                        if value and isinstance(value[0], float): dtype = torch.float
                    
                    try:
                        final_batch[key] = torch.tensor(value, dtype=dtype)
                    except ValueError as e:
                        # This should ideally not be reached if padding is correct for all extracted fields
                        print(f"Error converting key '{key}' to tensor during final batch construction. Padded length was {sequence_length}.")
                        print(f"Problematic value (first element example, type {type(value[0]) if value else 'N/A'}, length {len(value[0]) if value and isinstance(value[0], list) else 'N/A'}): {str(value)[:200]}...")
                        raise e
            
        if batch_original_idx is not None and self.return_tensors == "pt":
            final_batch["original_idx"] = torch.tensor(batch_original_idx, dtype=torch.long)

        return final_batch


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from types import SimpleNamespace

    mock_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if mock_tokenizer.pad_token_id is None:
        mock_tokenizer.pad_token_id = mock_tokenizer.eos_token_id
    
    special_tokens_to_add = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
    mock_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
    
    special_token_ids_test = {
        'start_latent_id': mock_tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        'latent_token_id': mock_tokenizer.convert_tokens_to_ids("<|latent|>"),
        'end_latent_id': mock_tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        'eos_token_id': mock_tokenizer.eos_token_id
    }

    mock_config = SimpleNamespace(
        model_id="gpt2",
        max_seq_length=70,
        use_latent_loop=True,
        c_thought=2,
        max_latent_stage=1,
        no_cot=False,
        no_special_latent_markers=False,
        pad_latent_to_max=False,
        answer_format_prefix = "### ",
        num_map_processors = 1,
        get=lambda key, default=None: getattr(mock_config, key, default)
    )

    dummy_data = [
        {
            "question": "Q1 short", "steps": ["S1.1", "S1.2"], "answer": "A1", "idx": 0
        },
        {
            "question": "Q2 much longer question", "steps": ["S2.1"], "answer": "A2 long", "idx": 1
        }
    ]
    dummy_file_path = "_test_dataset_utils_dummy_data.json"
    with open(dummy_file_path, 'w') as f:
        json.dump(dummy_data, f)

    print(f"--- Testing load_and_tokenize_raw_data (with standard JSON) ---")
    base_dataset = load_and_tokenize_raw_data(dummy_file_path, mock_tokenizer, mock_config, max_samples=2)

    if base_dataset and base_dataset.num_rows > 0: # Check with num_rows for HF Dataset
        print(f"Loaded base_dataset with {base_dataset.num_rows} samples.")
        # The output of load_and_tokenize_raw_data now has 'question_tokens', 'steps_tokens', 'answer_tokens', 'original_idx'
        # format_sample_for_training consumes these and outputs 'input_ids', 'attention_mask', 'labels', 'position_ids', 'original_idx'
        
        # Manually create the dataset that would be fed to the collator (output of .map(format_sample_for_training))
        # This requires that `base_dataset` itself is a HF Dataset, which it is.
        # Ensure that `format_sample_for_training` works on the output of `load_and_tokenize_raw_data`
        
        # In train.py, .map() would remove columns like "question_tokens" if specified.
        # For this test, we assume those columns are removed before collator.
        # The collator now expects ONLY the keys it will process.

        formatted_features_list = []
        for i in range(base_dataset.num_rows):
            # Simulate the output of base_dataset.map(format_sample_for_training, remove_columns=...)
            # format_sample_for_training takes one sample from base_dataset
            formatted_sample = format_sample_for_training(base_dataset[i], mock_config, i % 2, special_token_ids_test) # Alternate stage for variety
            formatted_features_list.append(formatted_sample)
        
        print(f"\n--- Sample 0 (Stage 0) going to collator ---")
        print(f"Input IDs len: {len(formatted_features_list[0]['input_ids'])}")
        print(f"Position IDs len: {len(formatted_features_list[0]['position_ids'])}")

        print(f"\n--- Sample 1 (Stage 1) going to collator ---")
        print(f"Input IDs len: {len(formatted_features_list[1]['input_ids'])}")
        print(f"Position IDs len: {len(formatted_features_list[1]['position_ids'])}")

        print(f"\n--- Testing DataCollatorForPadding ---")
        collator = DataCollatorForPadding(tokenizer=mock_tokenizer, label_pad_token_id=IGNORE_INDEX, position_id_pad_value=0)
        
        collated_batch = collator(formatted_features_list)
        print("\nCollated batch keys:", collated_batch.keys())
        for key in collated_batch.keys():
            if isinstance(collated_batch[key], torch.Tensor):
                print(f"Shape of '{key}': {collated_batch[key].shape}")
        print(f"Input IDs (sample 0 in batch, shape {collated_batch['input_ids'][0].shape}): {collated_batch['input_ids'][0]}")
        print(f"Position IDs (sample 0 in batch, shape {collated_batch['position_ids'][0].shape}): {collated_batch['position_ids'][0]}")
        print(f"Labels (sample 0 in batch, shape {collated_batch['labels'][0].shape}): {collated_batch['labels'][0]}")

    else:
        print("Failed to load dataset for testing.")

    if os.path.exists(dummy_file_path):
        os.remove(dummy_file_path)
    print(f"\n--- dataset_utils.py testing finished ---")