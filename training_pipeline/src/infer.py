import argparse

from peft import PeftModel

from src.data.dataset import build_prompt
from src.model.inference import generate
from src.model.models import load_quantized_model, load_tokenizer
from src.utils import DATA_DIR, OUTPUT_DIR, load_config, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=str(OUTPUT_DIR / "adapter"))
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config()
    base = cfg["model"]["base"]

    tokenizer = load_tokenizer(base)
    model = PeftModel.from_pretrained(load_quantized_model(base), args.adapter)
    model.config.use_cache = True
    model.eval()

    version = cfg["data"]["version"]
    test = read_json(DATA_DIR / f"test_data_{version}.json")

    for record in test[: args.n]:
        prompt = build_prompt(record["context"], record["question"])
        prediction = generate(model, tokenizer, prompt)
        print("\nQUESTION:", record["question"])
        print("TEACHER :", record["answer"])
        print("MODEL   :", prediction)


if __name__ == "__main__":
    main()
