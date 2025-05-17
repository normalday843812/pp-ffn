# src/config_loader.py

import yaml
import argparse
import os
from typing import Any, Dict, Optional, List

class SimpleConfig:
    """
    A simple class to hold configuration parameters.
    Allows accessing config values as attributes (e.g., config.lr)
    and provides a get method for safe access with defaults.
    Nested dictionaries are also converted to SimpleConfig objects.
    """
    def __init__(self, config_dict: Dict[str, Any]):
        # Store the original dict for __str__ and to_dict methods
        self._config_dict = config_dict
        for key, value in config_dict.items():
            # Ensure keys are valid attribute names for direct attribute access
            # This is a basic check; more robust validation might be needed for all edge cases.
            processed_key = key.replace('-', '_') # Convert hyphens to underscores for attribute names
            if not processed_key.isidentifier():
                print(f"Warning: Config key '{key}' (processed as '{processed_key}') is not a valid Python identifier and will not be directly accessible as an attribute using dot notation. Use .get('{key}') instead.")
            setattr(self, processed_key, self._parse_value(value))

    def _parse_value(self, value: Any) -> Any:
        """Recursively parse values, converting dicts to SimpleConfig and lists of dicts."""
        if isinstance(value, dict):
            return SimpleConfig(value)
        elif isinstance(value, list):
            return [self._parse_value(item) for item in value]
        return value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Gets a configuration value.
        Tries to access as an attribute (handling processed keys like hyphens to underscores).
        If direct attribute access fails (e.g., original key had special chars not convertible
        to valid identifiers or key simply doesn't exist), it falls back to checking the
        original dictionary.
        """
        processed_key = key.replace('-', '_')
        if hasattr(self, processed_key):
            return getattr(self, processed_key)
        # Fallback for keys that couldn't be set as attributes or if user uses original key
        if self._config_dict is not None and key in self._config_dict:
             # Reparse here to ensure nested dicts become SimpleConfig objects even via this path
            return self._parse_value(self._config_dict[key])
        return default

    def __repr__(self) -> str:
        # Represent the config by showing its attributes (excluding internal _config_dict)
        attrs = {k: v for k, v in self.__dict__.items() if k != '_config_dict'}
        return f"SimpleConfig({attrs})"

    def __str__(self) -> str:
        # Pretty print the original dictionary structure using YAML
        # This is often more readable for nested structures
        try:
            return yaml.dump(self._config_dict, indent=2, default_flow_style=False, sort_keys=False)
        except Exception: # Fallback if _config_dict is not yaml-dumpable for some reason
            return repr(self)


    def to_dict(self) -> Dict[str, Any]:
        """Returns the original dictionary that was loaded."""
        return self._config_dict


def load_config(config_path: str) -> SimpleConfig:
    """
    Loads a YAML configuration file from the given path.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A SimpleConfig object holding the configuration.

    Raises:
        FileNotFoundError: If the config_path does not exist.
        yaml.YAMLError: If the YAML file is not valid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        try:
            config_dict = yaml.safe_load(f)
            if config_dict is None: # Handle empty YAML file case
                print(f"Warning: Configuration file at {config_path} is empty or contains only null values.")
                config_dict = {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {config_path}")
            raise e

    return SimpleConfig(config_dict)

# This block allows testing the script directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and test a YAML configuration file.")
    parser.add_argument(
        "config_file_path",
        type=str,
        help="Path to the YAML configuration file (e.g., ../configs/gsm8k_base_sft.yaml)"
    )
    args = parser.parse_args()

    print(f"Attempting to load configuration from: {args.config_file_path}")

    try:
        config = load_config(args.config_file_path)

        print("\n--- Configuration Loaded Successfully ---")
        print("String representation (YAML dump of original dict):")
        print(config) # This will use __str__

        print("\n--- Accessing some specific values: ---")
        # Using .get() for safety, especially if keys might be missing or have special chars
        print(f"Project Name: {config.get('project_name', 'Not Specified')}")
        print(f"Model ID (attribute access): {config.model_id if hasattr(config, 'model_id') else 'Not Specified'}")
        print(f"Learning Rate (get method): {config.get('lr', 'Not Specified')}")
        print(f"Batch Size Training (attribute access, was batch_size_training): {config.batch_size_training if hasattr(config, 'batch_size_training') else 'Not Specified'}")


        print("\n--- Example of a potentially hyphenated key from YAML (if it existed): ---")
        # If your YAML had "my-parameter: value", it would be config.my_parameter
        # We'll test with a key that's not in the current examples for demonstration
        print(f"Value for 'a-hyphenated-key': {config.get('a-hyphenated-key', 'Default for hyphenated')}")


        print("\n--- Testing non-existent key with default: ---")
        print(f"A non_existent_key: {config.get('non_existent_key_123', 'DefaultValueForMissing')}")

        print("\n--- Testing direct attribute access for a known valid key: ---")
        if hasattr(config, 'seed'):
            print(f"Seed (direct attribute): {config.seed}")
        else:
            print("'seed' attribute not found.")

        print("\n--- To run this test from the project root directory: ---")
        print("python src/config_loader.py configs/your_config_file.yaml")
        print("Example: python src/config_loader.py configs/gsm8k_base_sft.yaml")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except yaml.YAMLError as e:
        print(f"YAML Parsing Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")