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
        self._config_dict = config_dict
        for key, value in config_dict.items():
            processed_key = key.replace('-', '_')
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
        if self._config_dict is not None and key in self._config_dict:
            return self._parse_value(self._config_dict[key])
        return default

    def __repr__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items() if k != '_config_dict'}
        return f"SimpleConfig({attrs})"

    def __str__(self) -> str:
        try:
            return yaml.dump(self._config_dict, indent=2, default_flow_style=False, sort_keys=False)
        except Exception:
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