import comet_ml  

import logging
import os

from comet_ml import Experiment

from src.data.dataset import load_tokenized_datasets
from src.model.models import add_lora, load_quantized_model, load_tokenizer
from src.model.trainer import build_trainer, register_adapter, save_adapter
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = load_config()

    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        workspace=os.getenv("COMET_WORKSPACE"),
        project_name=os.getenv("COMET_PROJECT_NAME"),
    )
    for section in ("model", "lora", "training"):
        experiment.log_parameters(cfg[section], prefix=section)

    base = cfg["model"]["base"]
    tokenizer = load_tokenizer(base)
    train_ds, test_ds = load_tokenized_datasets(cfg, tokenizer)
    logger.info("Train: %d examples, test: %d examples", len(train_ds), len(test_ds))

    model = add_lora(load_quantized_model(base), cfg["lora"])
    trainer = build_trainer(model, tokenizer, train_ds, test_ds, cfg)
    trainer.train()

    adapter_dir = save_adapter(trainer)
    register_adapter(experiment, adapter_dir, cfg["registry"]["model_name"])
    logger.info("Adapter saved to %s and registered in Comet", adapter_dir)
    experiment.end()


if __name__ == "__main__":
    main()
