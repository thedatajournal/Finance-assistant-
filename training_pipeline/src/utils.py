import json
import logging
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR = ROOT_DIR / "logs"
CONFIG_PATH = Path(__file__).parent / "config.yml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "training.log"),
        ],
    )


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
