import asyncio
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

model_path = "casperhansen/mixtral-instruct-awq"

# prompting
prompt = "You're standing on the surface of the Earth. "\
         "You walk one mile south, one mile west and one mile north. "\
         "You end up exactly where you started. Where are you?",

prompt_template = "[INST] {prompt} [/INST]"

# sampling params
sampling_params = SamplingParams(
    repetition_penalty=1.1,
    temperature=0.8,
    max_tokens=512
)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# async engine args for streaming
engine_args = AsyncEngineArgs(
    model=model_path,
    quantization="awq",
    dtype="float16",
    max_model_len=512,
    enforce_eager=True,
    disable_log_requests=True,
    disable_log_stats=True,
)

async def generate(model: AsyncLLMEngine, tokenizer: PreTrainedTokenizer):
    tokens = tokenizer(prompt_template.format(prompt=prompt)).input_ids

    outputs = model.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=1,
        prompt_token_ids=tokens,
    )

    print("\n** Starting generation!\n")
    last_index = 0

    async for output in outputs:
        print(output.outputs[0].text[last_index:], end="", flush=True)
        last_index = len(output.outputs[0].text)
    
    print("\n\n** Finished generation!\n")

if __name__ == '__main__':
    model = AsyncLLMEngine.from_engine_args(engine_args)
    asyncio.run(generate(model, tokenizer))