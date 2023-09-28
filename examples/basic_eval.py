from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from lm_eval import evaluator, tasks

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_path = "/code/AutoAWQ/examples/models/llama-7b-awq"
quant_file = "llama2-7b-base-4bit-AWQ.pt"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, quant_file, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)

# model_name, model, tokenizer, device, batch_size=1, max_length=-1

lm_eval_model = LMEvalAdaptor(quant_file, model, tokenizer, device=model.model.device, batch_size=1, max_length=-1)
results = evaluator.simple_evaluate(
    model=lm_eval_model,
    tasks=['wikitext'],
    batch_size=1,
    no_cache=True,
    num_fewshot=0,
)

print(evaluator.make_table(results))
