"""
this file is really just to get used to using unsloth, not for our main expirements
"""

import torch
from datasets import load_dataset, load_from_disk
from pathlib import Path
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from trl import SFTTrainer
from transformers import TrainingArguments
from modelnames import BIT4

# from multiprocessing import freeze_support
# freeze_support()

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

path = (Path(__file__).parent / "../data/GSM8K").resolve()
dataset_train = load_from_disk(str(path / "train"))
dataset_test = load_from_disk(str(path / "test"))

# use .from_pretrained to load and .save_pretrained to save

# print(type(model))
# print(dir(model))
# print(type(tokenizer))
# print(dir(tokenizer))
# print(":)")

model = FastLanguageModel.get_peft_model(
    model,
    r=8, # any number >0 ! suggested: 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",

    # use_gradient_checkpointing = "unsloth", # "unsloth" uses less VRAM
    use_gradient_checkpointing=True,
    random_state = 3407,
    use_rslora = False,
    loftq_config = None
)

# print(dir(dataset))

prompt = """Based on given instruction and context, generate an appropriate response

### Instruction:
{}

### Context:
{}

### Response:
{}
"""

prompt = """Please answer the following question as best you can. Show the steps you take on the way to your solution.

### Question:
{}
"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    print(examples)

    #! this codenot work, the prompt above needs to be changed

    question = examples['question']
    answer = examples['answer']
    texts = []

    for q, a in zip(question, answer):
        text = prompt.format(q) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset_train = dataset_train.map(formatting_prompts_func, batched=False)
dataset_test = dataset_test.map(formatting_prompts_func, batched=False)

#! something about the trainer object is causing multiprocessing issues
# trainer = SFTTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     train_dataset = dataset_train,
#     eval_dataset = dataset_test, 
#     dataset_text_field = "text",
#     max_seq_length = max_seq_len,
#     dataset_num_proc = 2,
#     packing = False,
#     formatting_func=formatting_prompts_func,
#     args = TrainingArguments(
#         per_device_train_batch_size = 2,
#         gradient_accumulation_steps = 4,
#         warmup_steps = 2,
#         max_steps = 10,
#         learning_rate = 0.0005,
#         fp16 = not torch.cuda.is_bf16_supported(),
#         bf16 = torch.cuda.is_bf16_supported(),
#         logging_steps = 1,
#         optim = "adamw_8bit",
#         weight_decay = 0.01,
#         lr_scheduler_type = "linear",
#         seed = 3407,
#         output_dir = "outputs",
#     ),
# )

# # do the actual training using SFTTrainer
# trainer_stats = trainer.train()

# print(trainer_stats)