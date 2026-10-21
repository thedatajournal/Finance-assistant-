from datasets import Dataset

from src.utils import DATA_DIR, read_json

# Must match PROMPT_TEMPLATE in the inference pipeline's chains.py
PROMPT_TEMPLATE = """You are a finance assistant. Answer the question using the news articles below. If the articles don't contain the answer, say so instead of guessing.

News articles:
{context}

Conversation so far:
{history}

Question: {question}"""


def format_context(docs: list) -> str:
    if not docs:
        return "No relevant news found."
    return "\n\n".join(f"[{i}] {d}" for i, d in enumerate(docs, 1))


def build_prompt(context: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, history="None", question=question)


def tokenize_record(record: dict, tokenizer, max_length: int) -> dict:
    """Tokenize one example. Loss is computed only on the answer tokens."""
    prompt = build_prompt(record["context"], record["question"])
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + record["answer"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids):])[: len(full_ids)]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def load_tokenized_datasets(cfg: dict, tokenizer):
    version = cfg["data"]["version"]
    max_length = cfg["model"]["max_seq_length"]
    train = read_json(DATA_DIR / f"training_data_{version}.json")
    test = read_json(DATA_DIR / f"test_data_{version}.json")

    def prepare(records):
        ds = Dataset.from_list(records)
        return ds.map(
            lambda r: tokenize_record(r, tokenizer, max_length),
            remove_columns=ds.column_names,
        )

    return prepare(train), prepare(test)
