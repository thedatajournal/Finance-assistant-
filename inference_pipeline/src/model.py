import torch
from comet_ml import API
from langchain_community.llms import HuggingFacePipeline
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from src import constants
from src.utils import load_config


def download_adapter() -> str:
    """Download the fine-tuned LoRA adapter from the Comet model registry."""
    api = API(api_key=constants.COMET_API_KEY)
    api.download_registry_model(
        workspace=constants.COMET_WORKSPACE,
        registry_name=constants.COMET_MODEL_NAME,
        version=constants.COMET_MODEL_VERSION,
        output_path=constants.ADAPTER_DIR,
        expand=True,
    )
    return constants.ADAPTER_DIR


def build_llm():
    cfg = load_config()["generation"]
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model_name = constants.BASE_LLM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        if constants.USE_ADAPTER:
            model = PeftModel.from_pretrained(model, download_adapter())
    else:
        model_name = constants.FALLBACK_LLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg["max_new_tokens"],
        do_sample=True,
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe), tokenizer
