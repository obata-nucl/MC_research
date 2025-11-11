from pathlib import Path
import torch

def load_config():
    import yaml
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config.yml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config.yml : {e}")
        return None
    
    abs_paths = {}
    for key, rel_path in config["paths"].items():
        abs_paths[key] = (root_dir / rel_path).resolve()
    config["paths"] = abs_paths

    return config

def get_all_patterns(nodes_options: list[int], layers_options: list[int]) -> list[list[int]]:
    import itertools
    all_patterns = []
    for num_layer in layers_options:
        for nodes in itertools.product(nodes_options, repeat=num_layer):
            all_patterns.append(list(nodes))
    
    return all_patterns

def load_scaler(config):
    scaler_path = config["paths"]["results_dir"] / "scaler.pt"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    scaler = torch.load(scaler_path, map_location='cpu')
    return scaler

def _pattern_to_name(pattern: list[int]) -> str:
    return '-'.join(map(str, pattern)) if isinstance(pattern, (list, tuple)) else str(pattern)

def _parse_pattern_name(pattern_name: str) -> list[int]:
    return [int(x) for x in str(pattern_name).split('-') if x]

