from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from src.utils import OUTPUT_DIR


def build_trainer(model, tokenizer, train_ds, test_ds, cfg: dict) -> Trainer:
    t = cfg["training"]
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="no",
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="comet_ml",
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
    )


def save_adapter(trainer: Trainer) -> str:
    adapter_dir = OUTPUT_DIR / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    return str(adapter_dir)


def register_adapter(experiment, adapter_dir: str, model_name: str) -> None:
    """Upload the LoRA adapter to the Comet model registry."""
    experiment.log_model(model_name, adapter_dir, prepend_folder_name=False)
    experiment.register_model(model_name)
