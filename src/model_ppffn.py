# src/model_ppffn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml # For testing block
import traceback # For testing block
from typing import Optional, Any, Dict

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2PreTrainedModel,
    GPT2Model,
    GPT2LMHeadModel,
    GPT2Block,
    GPT2Config
)
from transformers.activations import NewGELUActivation
from transformers import AutoTokenizer # For testing block

# Assuming config_loader.py is in the same package directory (src)
from .config_loader import SimpleConfig, load_config

class ParallelPathFFN(nn.Module):
    """A Feed-Forward Network block with N parallel paths, replacing a standard MLP."""
    def __init__(self, hf_config: GPT2Config, ppffn_custom_config: SimpleConfig):
        super().__init__()
        self.hidden_size = hf_config.n_embd
        self.num_paths = ppffn_custom_config.get('ppffn_num_paths', 2)
        
        default_total_inner_dim = hf_config.n_inner if hf_config.n_inner is not None else 4 * self.hidden_size
        
        path_intermediate_size_config = ppffn_custom_config.get('ppffn_path_intermediate_size')
        if path_intermediate_size_config is None:
            self.path_intermediate_size = max(1, default_total_inner_dim // self.num_paths)
        else:
            self.path_intermediate_size = path_intermediate_size_config

        self.paths = nn.ModuleList()
        for _ in range(self.num_paths):
            path = nn.Sequential(
                nn.Linear(self.hidden_size, self.path_intermediate_size),
                NewGELUActivation(),
                nn.Linear(self.path_intermediate_size, self.hidden_size)
            )
            self.paths.append(path)

        self.aggregation_method = ppffn_custom_config.get('ppffn_aggregation_method', 'sum').lower()

        if self.aggregation_method == 'concat_squeeze':
            self.squeeze_layer = nn.Linear(self.num_paths * self.hidden_size, self.hidden_size)
        elif self.aggregation_method == 'gate':
            self.gate_layer = nn.Linear(self.hidden_size, self.num_paths)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        path_outputs = [path(hidden_states) for path in self.paths]

        if self.aggregation_method == 'sum':
            aggregated_output = torch.sum(torch.stack(path_outputs, dim=0), dim=0)
        elif self.aggregation_method == 'mean':
            aggregated_output = torch.mean(torch.stack(path_outputs, dim=0), dim=0)
        elif self.aggregation_method == 'concat_squeeze':
            concatenated = torch.cat(path_outputs, dim=-1)
            aggregated_output = self.squeeze_layer(concatenated)
        elif self.aggregation_method == 'gate':
            gate_weights = F.softmax(self.gate_layer(hidden_states), dim=-1) # (B, S, N)
            stacked_path_outputs = torch.stack(path_outputs, dim=0).permute(1, 2, 0, 3) # (B, S, N, H)
            aggregated_output = torch.sum(stacked_path_outputs * gate_weights.unsqueeze(-1), dim=2) # Sum over N
        else:
            raise ValueError(f"Unknown aggregation method for PP-FFN: {self.aggregation_method}")
        return aggregated_output

class GPT2BlockWithPPFFN(GPT2Block):
    """A GPT-2 Transformer Block using the ParallelPathFFN instead of the standard MLP."""
    def __init__(self, config: GPT2Config, ppffn_custom_config: SimpleConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx=layer_idx)
        self.mlp = ParallelPathFFN(config, ppffn_custom_config)

class GPT2ModelWithPPFFN(GPT2Model):
    """The bare GPT-2 model outputting raw hidden-states, using PP-FFN blocks."""
    def __init__(self, config: GPT2Config, ppffn_custom_config: SimpleConfig):
        super().__init__(config) # Initializes standard GPT2Model components, including self.h with GPT2Blocks
        # Override the stack of blocks 'h' with our custom blocks
        self.h = nn.ModuleList(
            [GPT2BlockWithPPFFN(config, ppffn_custom_config, layer_idx=i) for i in range(config.n_layer)]
        )
        # self.post_init() is called by super().__init__(config) if inheriting from PreTrainedModel

class GPT2LMHeadModelWithPPFFN(GPT2LMHeadModel):
    """GPT-2 Model with a language modeling head on top, using PP-FFN blocks."""
    def __init__(self, config: GPT2Config, ppffn_custom_config: SimpleConfig):
        super().__init__(config) # Initializes self.transformer as GPT2Model and self.lm_head
        # Override self.transformer with our custom PP-FFN version
        self.transformer = GPT2ModelWithPPFFN(config, ppffn_custom_config)
        # self.lm_head and weight tying are handled by super().__init__ and its call to post_init()

    @classmethod
    def from_pretrained_with_ppffn(
        cls,
        pretrained_model_name_or_path: str,
        ppffn_simple_config: SimpleConfig,
        *model_args,
        **kwargs
    ):
        hf_base_config = GPT2Config.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(hf_base_config, ppffn_simple_config) # PP-FFN parts are randomly initialized here

        try:
            base_model_hf = GPT2LMHeadModel.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
            base_model_state_dict = base_model_hf.state_dict()
            del base_model_hf
        except Exception as e:
            print(f"Error loading base model '{pretrained_model_name_or_path}' for state_dict: {e}")
            raise

        current_model_state_dict = model.state_dict()
        copied_keys_count = 0
        
        # Only copy weights that are NOT part of the MLP (which is now PP-FFN)
        # and exist in both models with matching shapes.
        weights_to_load = {}
        for key, param_base in base_model_state_dict.items():
            if ".mlp." not in key: # Skip original MLP weights
                if key in current_model_state_dict and current_model_state_dict[key].shape == param_base.shape:
                    weights_to_load[key] = param_base
                    copied_keys_count +=1
        
        # Load the filtered state dict. Missing keys (PP-FFN parts) will keep their init.
        # Unexpected keys (original MLP parts not in our model) will be ignored by strict=False.
        model.load_state_dict(weights_to_load, strict=False)
        
        print(f"Initialized PP-FFN model. Copied {copied_keys_count} parameter sets "
              f"from pre-trained '{pretrained_model_name_or_path}'. MLP/FFN parts specific to PP-FFN are newly initialized.")
        
        return model

# --- Testing Block ---
if __name__ == "__main__":
    # To run this test block:
    # 1. Make sure you have a `src/__init__.py` (can be empty)
    # 2. From the project root (`ppffn_project/`), run:
    #    python -m src.model_ppffn
    print("--- Testing model_ppffn.py ---")

    DUMMY_PPFFN_CONFIG_FILE = "_test_model_ppffn_cfg.yaml"
    ppffn_yaml_params = {
        "ppffn_num_paths": 2,
        "ppffn_path_intermediate_size": 512, # For gpt2 (hidden 768), default inner is 3072. 512*2=1024.
        "ppffn_aggregation_method": "sum"
    }
    with open(DUMMY_PPFFN_CONFIG_FILE, 'w') as f:
        yaml.dump(ppffn_yaml_params, f)
    
    try:
        test_ppffn_config = load_config(DUMMY_PPFFN_CONFIG_FILE)
        model_name = "gpt2" 
        
        print(f"\nAttempting to create custom PP-FFN model based on '{model_name}' (sum aggregation)...")
        custom_model_sum = GPT2LMHeadModelWithPPFFN.from_pretrained_with_ppffn(
            model_name, 
            ppffn_simple_config=test_ppffn_config
        )
        print(f"Successfully created '{model_name}' with PP-FFN (sum).")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
        
        test_input_text = "Hello, this is a test."
        inputs = tokenizer(test_input_text, return_tensors="pt")
        
        custom_model_sum.eval()
        with torch.no_grad():
            outputs_sum = custom_model_sum(**inputs)
        
        print("Logits shape (sum):", outputs_sum.logits.shape) 
        assert outputs_sum.logits.shape == (1, inputs.input_ids.shape[1], custom_model_sum.config.vocab_size)
        print("Forward pass successful (sum aggregation).")

        # Test with concat_squeeze aggregation
        ppffn_yaml_params_concat = {
            "ppffn_num_paths": 3, "ppffn_path_intermediate_size": 256, 
            "ppffn_aggregation_method": "concat_squeeze"
        }
        test_ppffn_config_concat = SimpleConfig(ppffn_yaml_params_concat)
        
        print(f"\nAttempting to create custom PP-FFN model based on '{model_name}' (concat_squeeze aggregation)...")
        custom_model_concat = GPT2LMHeadModelWithPPFFN.from_pretrained_with_ppffn(
            model_name,
            ppffn_simple_config=test_ppffn_config_concat
        )
        print(f"Successfully created '{model_name}' with PP-FFN (concat_squeeze).")
        custom_model_concat.eval()
        with torch.no_grad():
            outputs_concat = custom_model_concat(**inputs)
        print("Logits shape (concat_squeeze):", outputs_concat.logits.shape)
        assert outputs_concat.logits.shape == (1, inputs.input_ids.shape[1], custom_model_concat.config.vocab_size)
        print("Forward pass successful (concat_squeeze aggregation).")

    except Exception as e:
        print(f"An error occurred during testing: {e}")
        traceback.print_exc()
    finally:
        if os.path.exists(DUMMY_PPFFN_CONFIG_FILE):
            os.remove(DUMMY_PPFFN_CONFIG_FILE)
    
    print("\n--- Testing of model_ppffn.py finished ---")