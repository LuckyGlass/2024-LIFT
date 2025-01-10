import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoConfig, AutoTokenizer
from typing import Sequence
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
    lora_config = LoraConfig(
        r=2,
        target_modules=["lm_head"],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=4,
        modules_to_save=[f"layers.{i}.self_attn.mem_proj" for i in range(len(model.model.layers))]+[f"layers.{i}.self_attn.gate_proj" for i in range(len(model.model.layers))],
    )
    model = get_peft_model(model, lora_config)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def process_ganeration_memgate(memgate: Sequence[Sequence[torch.Tensor]]):
    """Process outputs.attention (memgate) into a tensor.
    Args:
        memgate (Sequence[Sequence[Tensor]]): The memgate output of generate(...); the first tuple is the new token dim, the second tuple is the layer dim, the tensor is in the shape of [Batch, Head, Query length, 1].
    Returns:
        avg_memgate (Tensor): The memgate of output tokens averaged over layers and heads, in the shape of [Batch, New token].
    """
    avg_memgate = torch.stack([torch.stack([torch.mean(layer[:, :, -1, 0], dim=1) for layer in token]) for token in memgate])
    avg_memgate = torch.mean(avg_memgate, dim=1).transpose(0, 1)
    return avg_memgate
