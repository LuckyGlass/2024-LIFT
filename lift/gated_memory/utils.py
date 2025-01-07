from transformers import AutoConfig, AutoTokenizer
from .model import GMLlamaForCausalLM


def preprocess(model_name_or_path: str, output_dir: str):
    """Preprocess the Llama-3.1 checkpoint into a gated-memory model checkpoint.
    Args:
        model_name_or_path (str): The Llama-3.1 checkpoint dir.
        output_dir (str): The saving dir.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    model_config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    model = GMLlamaForCausalLM.from_pretrained(model_name_or_path, config=model_config, local_files_only=True, low_cpu_mem_usage=False, _fast_init=False)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)