from dataclasses import dataclass, field
from typing import Optional, Union
import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Base model or residual model setting
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    # Lora or PiSSA setting
    full_finetune: Optional[bool] = field(default=True)
    adapter_name_or_path: Optional[str] = field(default=None,metadata={"help": ("Pre-initialized PiSSA adapter path; when this is not None, the following arguments are ignored."),},)
    init_weights: Union[bool, str] = field(default=True,metadata={"help": ("True -> LoRA; `pissa` -> PiSSA; `pissa_niter_16` -> Fast SVD PiSSA"),},)
    target_modules: Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank: Optional[int] = field(default=8)
    lora_alpha: Optional[float] = field(default=32.)
    lora_dropout: Optional[float] = field(default=0.,metadata={"help": ("Must be set to 0 when using PiSSA."),},)
    # Quantization setting
    bits: int = field(default=16,metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True,metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    # TrainingArguments
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512,metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    merge : Optional[bool] = field(default=False,metadata={"help": "Merge the PiSSA adapter to the residual model or LoRA to the base model"},)
    deepspeed_is_zero3: bool = field(default=False, metadata={'help': "It needn't be assigned; the stage of deepspeed."})
    
    def __post_init__(self):
        # deepspeed_enabled = self.deepspeed is not None
        super().__post_init__()
        # if deepspeed_enabled:
        #     self.deepspeed_is_zero3 = self.hf_deepspeed_config.is_zero3()


@dataclass
class LIFTDataArguments:
    data_path: str = field(default="", metadata={"help": "The path to the dataset."})
    len_segment: int = field(default=8, metadata={"help": "The length (block) of the segment."})
    len_offset: int = field(default=3, metadata={"help": "The length (block) of the offset."})
    block_size: int = field(default=256, metadata={"help": "The size of the block."})
    input_cache_path: Optional[str] = field(default=None, metadata={"help": "The path to the input cache."})
    num_article_epochs: int = field(default=1, metadata={"help": "The number of epochs for the article."})
    num_article_qa_epochs: int = field(default=1, metadata={"help": "The number of epochs for the article and QAs."})
    generator_name_or_path: str = field(default="", metadata={"help": "The path to the generator."})
    num_syn_qa: int = field(default=1, metadata={"help": "The number of synthetic QAs."})
