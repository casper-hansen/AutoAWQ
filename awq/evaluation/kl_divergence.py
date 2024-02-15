# adapted from https://gist.github.com/Ttl/0d51f739dc59254b4b2183e259c97d82

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

try:
    from scipy.stats import bayes_mvs
    from scipy.stats import t as student_t
    from scipy.stats.mstats import mquantiles_cimj

    SCIPY_INSTALLED = True
except:
    SCIPY_INSTALLED = False


@torch.jit.script
def rel_entr(x, y):
    mask = (x > 0) & (y > 0)
    result = torch.where(mask, x * torch.log(x / y), torch.zeros_like(x))
    result[(x > 0) & (y <= 0)] = float("inf")
    return result


def bin_conf(p, n, z):
    # Binomial distribution confidence bounds
    # Bayes estimator when p is degenerate
    if p == 0:
        p = 1 / (n + 2)
    if p == 1:
        p = 1 - 1 / (n + 2)
    return z * torch.sqrt(p * (1 - p) / n)


def eval_kl_divergence(
    ref_model: PreTrainedModel,
    eval_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    seqlen: int,
):
    if not SCIPY_INSTALLED:
        raise Exception(
            "SciPy needs to be installed for KL Divergence evaluation: pip install scipy"
        )

    # load dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    data = data.input_ids.to(ref_model.device)

    n_samples = data.numel() // seqlen

    alpha = 0.01
    kls = []
    top1 = 0
    top5 = 0
    top10 = 0
    eval_top5 = 0
    eval_top10 = 0
    samples = 0
    i = 0

    # start eval
    with tqdm(range(n_samples), desc="KL Div") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch_len = end_index - start_index
            batch = data[:, start_index:end_index]

            # get logits
            with torch.no_grad():
                y1 = ref_model(batch)[0]
                y2 = eval_model(batch)[0]

            # kl divergence
            y1_probs = torch.softmax(y1, dim=-1)
            y2_probs = torch.softmax(y2, dim=-1)
            relative_entropy = rel_entr(y1_probs, y2_probs)
            kl_div = torch.sum(relative_entropy, dim=-1).squeeze(0)
            kls.append(torch.nan_to_num(kl_div).tolist())

            # stats
            eval_argmax = torch.argmax(y2, axis=-1).squeeze(0)
            ref_argmax = torch.argmax(y1, axis=-1).squeeze(0)
            eval_part5 = torch.topk(y2, k=5, dim=-1).indices[:, :, -5].squeeze(0)
            ref_part5 = torch.topk(y1, k=5, dim=-1).indices[:, :, -5].squeeze(0)
            eval_part10 = torch.topk(y2, k=10, dim=-1).indices[:, :, -10].squeeze(0)
            ref_part10 = torch.topk(y1, k=10, dim=-1).indices[:, :, -10].squeeze(0)
            top1 += (eval_argmax == ref_argmax).sum().item()
            top5 += ((ref_argmax == eval_part5).sum()).item()
            top10 += ((ref_argmax == eval_part10).sum()).item()
            eval_top5 += ((eval_argmax == ref_part5).sum()).item()
            eval_top10 += ((eval_argmax == ref_part10).sum()).item()
            samples += batch_len

            progress_bar.set_description(
                f"KL Div: {torch.mean(torch.Tensor(kls)):.4g}, "
                f"Top 1: {top1 / samples:.4g}, "
                f"Top 5: {top5 / samples:.4g}, "
                f"Top 10: {top10 / samples:.4g}"
            )

    z = student_t.ppf(1 - alpha / 2, samples)
    m_conf = z * np.sqrt(np.mean([k**2 for k in kls]) / len(kls))
    m, _, __ = bayes_mvs(kls, 1 - alpha)
    q90 = np.quantile(kls, 0.90)
    q95 = np.quantile(kls, 0.95)
    q99 = np.quantile(kls, 0.99)
    q_bounds = mquantiles_cimj(kls, prob=[0.90, 0.95, 0.99])

    print(" -- ")
    print(" ** Reference model:", ref_model.config.model_type)
    print(" ** Evaluation model:", eval_model.config.model_type)
    print(" -- ")
    print(f" ** KL Divergence: {m[0]:.6g}, [{m[1][0]:.6g} - {m[1][1]:.6g}]")
    print(f" ** q90: {q90:.4g}, [{q_bounds[0][0]:.4g} - {q_bounds[1][0]:.4g}]")
    print(f" ** q95: {q95:.4g}, [{q_bounds[0][1]:.4g} - {q_bounds[1][1]:.4g}]")
    print(f" ** q99: {q99:.4g}, [{q_bounds[0][2]:.4g} - {q_bounds[1][2]:.4g}]")
    print(f"max: {np.max(kls):.4g}")
    print(" -- ")
    print("Reference top token in eval top-n probability:")
    print(
        f" ** ref_top1: {top1 / samples:.4g} ± {bin_conf(top1/samples, samples, z):.4g}"
    )
    print(
        f" ** ref_top5: {top5 / samples:.4g} ± {bin_conf(top5/samples, samples, z):.4g}"
    )
    print(
        f" ** ref_top10: {top10 / samples:4g} ± {bin_conf(top10/samples, samples, z):.4g}"
    )
    print("Eval top token in reference top-n probability:")
    print(
        f" ** eval_top5: {eval_top5 / samples:.4g} ± {bin_conf(eval_top5/samples, samples, z):.4g}"
    )
    print(
        f" ** eval_top10: {eval_top10 / samples:4g} ± {bin_conf(eval_top10/samples, samples, z):.4g}"
    )


if __name__ == "__main__":
    # ref_model_path = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    # eval_model_path = "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"
    ref_model_path = eval_model_path = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(ref_model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, device_map="auto")
    eval_model = AutoModelForCausalLM.from_pretrained(
        eval_model_path, device_map="auto"
    )

    eval_kl_divergence(ref_model, eval_model, tokenizer, seqlen=1024)
