# Finance-assistant-
A modular RAG-based finance chatbot built with MLOps best practices. It answers finance and investment questions using the latest financial news.

Tech Stack

	•	Vector database: Qdrant
	•	Embedding model: sentence-transformers/all-MiniLM-L6-v2
	•	LLM: mistral-7b-instruct-v0.2
	•	Experiment tracking and model registry: Comet-ML
	•	CI/CD: GitHub Actions
	•	Cloud: AWS EC2, AWS ECR
	•	UI: Gradio
	•	Other libraries: LangChain, Transformers, Bytewax, PyTorch, OpenAI, PEFT

Architecture

The system is split into three pipelines.

1. Feature (Streaming) Pipeline

Pulls financial news in real time from an external news API, converts each article into an embedding with a BERT-based model, and stores the embeddings and metadata in the vector database. These are used later for both training and inference. See the streaming_pipeline README for setup.

2. Training Pipeline

Builds a custom finance Q&A dataset by distilling knowledge from a larger LLM, then fine-tunes an open-source LLM on it using LoRA. Experiments are tracked in Comet-ML, and the resulting LoRA adapter is saved to its model registry. See the training_pipeline README for setup.

3. Inference Pipeline

Once the first two pipelines have run, this pipeline launches the Gradio chat UI and loads the fine-tuned LLM, the embedding model, and the vector database, chained together with LangChain. For each question it:

	1.	Takes the user's query from the UI.
	2.	Embeds the query and retrieves the most relevant news articles through vector search.
	3.	Builds a prompt from the query, chat history, and retrieved articles.
	4.	Runs the prompt through the fine-tuned LLM and streams the answer back in real time.

See the inference_pipeline folder for setup.

References

	•	https://github.com/iusztinpaul/hands-on-llms