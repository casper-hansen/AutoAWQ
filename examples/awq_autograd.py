"""AWQ backpropagation test
"""
from colorama import Style, Fore
from datetime import datetime as dt

import datasets as ds
from transformers import AutoTokenizer, LlamaTokenizer, TextStreamer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

from awq.models.auto import AutoAWQForCausalLM
from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
import awq.modules.peft_patch


# config_path = "/home/user/models/Llama-2-7b-Chat-AWQ/"
config_path = "/home/user/models/Llama-2-7B-AWQ"
model_path = "model.safetensors"


tokenizer = AutoTokenizer.from_pretrained(config_path)
tokenizer.pad_token = tokenizer.eos_token


# Datasets
data_train = ds.load_dataset("OpenAssistant/oasst1", split="train")
data_val = ds.load_dataset("OpenAssistant/oasst1", split="validation")

## Prepare split
def prepare_split(data):
    ## Get English entries for Human and AI
    data_prompter = data.filter(lambda row: row["role"] == "prompter" and row["lang"] == "en").\
        select_columns("text").rename_column("text", "usr_text")
    data_assistant = data.filter(lambda row: row["role"] == "assistant" and row["lang"] == "en").\
        select_columns("text")
    data_assistant = data_assistant.select(range(data_prompter.shape[0]))

    ## Fuse together Human and AI texts
    data = data_prompter.add_column("ai_text", data_assistant["text"])

    ## Make LLaMA2 templated text entries
    prompt_template = \
    """<s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {user_message} [/INST] {model_answer} </s>
    """
    data.add_column("text", data["usr_text"])

    def make_text(entry):
        entry["text"] = prompt_template.format(system_prompt="", user_message=entry["usr_text"], model_answer=entry["ai_text"])
        return entry


    data = data.map(make_text).select_columns("text")
    ## Tokenize
    data = data.map(lambda x: tokenizer(x["text"]), batched=True)
    return data


data_train = prepare_split(data_train).select(range(200))
data_val = prepare_split(data_val).select(range(200))


# Load model
model = AutoAWQForCausalLM.from_quantized(config_path, 
                                          model_path, 
                                          fuse_layers=False,
                                          safetensors=True
                                          )

# Config Lora
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.5,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False
)

model = get_peft_model(model.model, lora_config)

model.print_trainable_parameters()

training_arguments = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=1,
        optim="adamw_torch",
        num_train_epochs=1,
        learning_rate=1e-4,
        # fp16=True,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_steps=100,
        logging_steps=50,
        eval_steps=None,
        load_best_model_at_end=False
    )

trainer = Trainer(
    model=model,
    train_dataset=data_train,
    eval_dataset=data_val,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()


trainer.save_model("output")
# model.config.use_cache = False

