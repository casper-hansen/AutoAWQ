import torch
import torch.nn as nn
from tqdm import tqdm
from lm_eval import evaluator
from datasets import load_dataset
from transformers import pipeline
from evaluate import load as load_metric
from lm_eval.tasks import initialize_tasks
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


def evaluate_perplexity(model, tokenizer):
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    # load and prepare dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    data = data.input_ids.to(model.device)

    seqlen = 2048
    model = model.eval()
    n_samples = data.numel() // seqlen

    nlls = []

    with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index].to(model.device)
            with torch.no_grad():
                logits = model(batch).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = data[:, start_index:end_index][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")

    ppl = _perplexity(nlls, n_samples, seqlen)

    return ppl.item()


def eval_librispeech(model_id, num_samples=100, batch_size=4):
    try:
        import jiwer, librosa, soundfile
    except ImportError:
        print("Please install the following: pip install jiwer librosa soundfile")

    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

    # Load the Whisper model pipeline for automatic speech recognition
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        batch_size=batch_size,
        device=get_device(),
        torch_dtype=torch.float16,
    )

    # Word normalizer
    normalizer = BasicTextNormalizer()

    # Load the WER metric
    wer_metric = load_metric("wer")

    texts = []
    audio = []
    for i, data in tqdm(enumerate(dataset), total=num_samples, desc="Loading dataset"):
        if len(audio) == num_samples:
            break
        audio.append(data["audio"])
        texts.append(data["text"])

    references = []
    predictions = []

    with tqdm(range(0, num_samples, batch_size), desc="Word Error Rate: -") as pbar:
        for i in pbar:
            batch_audio = audio[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]

            # inference
            results = pipe(batch_audio, batch_size=len(batch_audio))

            # normalize text
            normalized_predictions = [normalizer(result["text"]) for result in results]
            normalized_texts = [normalizer(text) for text in batch_texts]

            predictions.extend(normalized_predictions)
            references.extend(normalized_texts)

            # word error rate computation
            wer = (
                wer_metric.compute(predictions=predictions, references=references) * 100
            )
            pbar.set_description(f"Word Error Rate: {wer:.3f}%")


def eval_mmlu(
    model_path="gpt2",
    num_fewshot=1,
    batch_size=1,
    device="cuda:0",
    task_use_pretrained=False,
):
    try:
        import vllm

        VLLM_INSTALLED = True
    except ImportError:
        VLLM_INSTALLED = False

    initialize_tasks(verbosity="DEBUG")

    if VLLM_INSTALLED:
        model = "vllm"
        model_args = dict(
            pretrained=model_path,
            max_model_len=2048,
            dtype="float16",
            trust_remote_code=True,
        )

        if not task_use_pretrained:
            model_args["quantization"] = "awq"
    else:
        model = "hf"
        model_args = dict(
            pretrained=model_path,
            device_map_option=device,
            dtype="float16",
            trust_remote_code=True,
        )
    model_args = ",".join([f"{k}={v}" for k, v in model_args.items()])

    results = evaluator.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=["mmlu"],
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        log_samples=False,
    )

    print(evaluator.make_table(results))


if __name__ == "__main__":
    ### PERPLEXITY
    # model_path = 'mistralai/Mistral-7B-Instruct-v0.1'
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # evaluate_perplexity(model, tokenizer)

    ### WORD ERROR RATE
    # model_id = "distil-whisper/distil-small.en" # 3.594
    model_id = "distil-whisper/distil-medium.en"  # 3.436
    eval_librispeech(model_id)
