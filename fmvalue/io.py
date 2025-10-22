import yaml
from .models import Config


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


