import datasets
from awq import AutoAWQForCausalLM
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

def prepare_split(tokenizer):
    data = datasets.load_dataset("mhenrichsen/alpaca_2k_test", split="train")
    prompt_template = "<s>[INST] {prompt} [/INST] {output}</s>"

    def format_prompt(x):
        return prompt_template.format(
            prompt=x["instruction"],
            output=x["output"]
        )

    data = data.map(
        lambda x: {"text": format_prompt(x)},
    ).select_columns(["text"])
    data = data.map(lambda x: tokenizer(x["text"]), batched=True)

    return data

model_path = "TheBloke/Mistral-7B-v0.1-AWQ"

# Load model
model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=False)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Prepare data
data_train = prepare_split(tokenizer)

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
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.save_model("output")