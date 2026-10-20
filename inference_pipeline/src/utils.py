from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def format_history(history: list, turns: int) -> str:
    """Turn Gradio 'messages' history into a short text transcript."""
    recent = history[-2 * turns:]
    lines = []
    for msg in recent:
        who = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{who}: {msg['content']}")
    return "\n".join(lines) if lines else "None"
