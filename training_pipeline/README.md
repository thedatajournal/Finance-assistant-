# Training Pipeline

Builds a finance Q&A dataset using a large LLM as a teacher (knowledge distillation), then fine-tunes an open-source LLM on it with LoRA. Runs are tracked in Comet-ML, and the resulting adapter is pushed to the Comet model registry for the inference pipeline.

## How It Works

1. **Data generation:** for each question in `data/input_queries.json`, relevant news is retrieved from Qdrant and the teacher model writes an answer grounded in that news. The examples are split into train and test sets.
2. **Fine-tuning:** Mistral-7B-Instruct is loaded in 4-bit and trained with QLoRA. The loss is computed only on the answer tokens.
3. **Registry:** the LoRA adapter is saved and registered in Comet as `finance-mistral-lora`.
4. **Check:** `infer.py` compares the fine-tuned model's answers to the teacher's on held-out questions.

## Running

> A GPU with at least 16 GB of memory is needed for training (Colab or Kaggle works).

```bash
pip install -r requirements.txt
```

Export these environment variables: `OPENAI_API_KEY`, `QDRANT_API_KEY`, `COMET_API_KEY`, `COMET_WORKSPACE`, `COMET_PROJECT_NAME`. Set the Qdrant URL in `src/config.yml`.

```bash
python -m src.data.data_generate   # build the dataset (run the streaming pipeline first)
python -m src.train                # fine-tune and register the adapter
python -m src.infer                # compare model answers to the teacher
```

## Notes

The prompt template in `src/data/dataset.py` must match the one in the inference pipeline.
