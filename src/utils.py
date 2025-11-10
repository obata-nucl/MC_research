import yaml

from pathlib import Path

def load_config():
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

def _pattern_to_name(pattern: list[int]) -> str:
    return '-'.join(map(str, pattern)) if isinstance(pattern, (list, tuple)) else str(pattern)

def _parse_pattern_name(pattern_name: str) -> list[int]:
    return [int(x) for x in str(pattern_name).split('-') if x]