# Inference Pipeline

This pipeline connects the Qdrant vector database, the fine-tuned LLM, and the embedding model using LangChain, and serves them as a chat application built with Gradio.

## How It Works

1. **Initialization:** Loads the embedding model, the base LLM, and the fine-tuned LoRA adapter from the Comet model registry. It also creates a Qdrant client and builds a sequential LangChain chain to handle user queries.
2. **Retrieval:** When a query arrives, the chain embeds it with the embedding model and runs a vector search in Qdrant to fetch the most relevant news documents.
3. **Prompt building:** Combines the user's query, the retrieved news documents, and the past chat history into a single prompt.
4. **Generation:** Runs the prompt through the LLM and sends the generated response back to the Gradio UI.

## Running Locally

> **Note:** You need a GPU with at least 16 GB of memory to load the LLM with QLoRA.

### 1. Set up requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

The pipeline reads news documents from Qdrant and loads the fine-tuned LoRA adapter from the Comet model registry, so export these environment variables first:

- `QDRANT_API_KEY`
- `COMET_WORKSPACE`
- `COMET_PROJECT_NAME`
- `COMET_API_KEY`

Then update the database URL in `src/constants.py`.

### 2. Run the application

```bash
python -m src.app
```

The financial assistant will be available at http://localhost:7860. To use a different port, change it in `src/constants.py`.
