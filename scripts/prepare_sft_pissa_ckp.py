import torch
import os
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


residual_model_path = "models/Llama-3-8B-Instruct-pissa-r128"
sft_adapter_path = "models/sft/checkpoint-26000"
target_model_path = "models/SFT-Llama-3-8B-Instruct-pissa-r128"

# Decompose the residual model
residual_model = AutoModelForCausalLM.from_pretrained(residual_model_path)
print(residual_model)
lora_config = LoraConfig(
    # init_lora_weights="pissa", # Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model.
    init_lora_weights="pissa_niter_4", # Initialize the PiSSA with fast SVD, which completes in just a few seconds.
    r=128,
    lora_alpha=128,
    lora_dropout=0, # Since the component of the PiSSA adapter are the principal singular values and vectors, dropout should be set to 0 to avoid random discarding.
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(residual_model, lora_config)
peft_model.print_trainable_parameters()

# Save PiSSA modules:
peft_model.peft_config["default"].init_lora_weights = True # Important
peft_model.save_pretrained(os.path.join(target_model_path, "pissa_init"))

# Unload new PiSSA adapter and merge SFT adapter
peft_model = peft_model.unload()
peft_model = PeftModel.from_pretrained(peft_model, sft_adapter_path)
peft_model = peft_model.merge_and_unload()
peft_model.save_pretrained(target_model_path)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(residual_model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.save_pretrained(target_model_path)
