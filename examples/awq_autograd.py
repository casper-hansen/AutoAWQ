import datasets
from awq import AutoAWQForCausalLM
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# Datasets
data_train = datasets.load_dataset("OpenAssistant/oasst1", split="train")
data_val = datasets.load_dataset("OpenAssistant/oasst1", split="validation")

## Prepare split
def prepare_split(data, tokenizer):
    ## Get English entries for Human and AI
    data_prompter = data.filter(lambda row: row["role"] == "prompter" and row["lang"] == "en").\
        select_columns("text").rename_column("text", "usr_text")
    data_assistant = data.filter(lambda row: row["role"] == "assistant" and row["lang"] == "en").\
        select_columns("text")
    data_assistant = data_assistant.select(range(data_prompter.shape[0]))

    ## Fuse together Human and AI texts
    data = data_prompter.add_column("ai_text", data_assistant["text"])

    ## Make mistral
    prompt_template = "<s>[INST] {system_prompt} {user_message}[/INST]{model_answer}</s>"
    data.add_column("text", data["usr_text"])

    def make_text(entry):
        entry["text"] = prompt_template.format(system_prompt="", user_message=entry["usr_text"], model_answer=entry["ai_text"])
        return entry

    data = data.map(make_text).select_columns("text")
    ## Tokenize
    data = data.map(lambda x: tokenizer(x["text"]), batched=True)
    return data

model_path = "mistralai/Mistral-7B-v0.1"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Prepare data
data_train = prepare_split(data_train, tokenizer).select(range(200))
data_val = prepare_split(data_val, tokenizer).select(range(200))

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