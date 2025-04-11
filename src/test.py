import torch
from unsloth import FastLanguageModel

# from trl import SFTTrainer

from modelnames import BIT4

major_version, minor_version = torch.cuda.get_device_capability()

max_seq_len = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BIT4.LLAMA3_8B,
    max_seq_length=4096,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

# use .from_pretrained to load and .save_pretrained to save

print(type(model))
# print(dir(model))
print(type(tokenizer))
# print(dir(tokenizer))
print(":)")