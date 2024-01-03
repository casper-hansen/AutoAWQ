# adapted from https://gist.github.com/Ttl/0d51f739dc59254b4b2183e259c97d82

import torch
import random
from datasets import load_dataset
from transformers import PreTrainedTokenizer

@torch.jit.script
def rel_entr(x, y):
    mask = (x > 0) & (y > 0)
    result = torch.where(mask, x * torch.log(x / y), torch.zeros_like(x))
    result[(x > 0) & (y <= 0)] = float('inf')
    return result

def bin_conf(p, n, z):
    # Binomial distribution confidence bounds
    # Bayes estimator when p is degenerate
    if p == 0:
        p = 1 / (n + 2)
    if p == 1:
        p = 1 - 1 / (n + 2)
    return z * torch.sqrt(p*(1-p)/n)

def eval_kl_divergence(model1, model2, tokenizer: PreTrainedTokenizer, seqlen: int):
    # load dataset
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    data = tokenizer("\n\n".join(data['text']), return_tensors='pt')

    # prepare dataset
    bos = 1 if tokenizer.bos_token_id is not None else 0
    tokens = [tokens[i:i+seqlen-bos] for i in range(0, len(data), seqlen-bos)]
    if tokenizer.bos_token_id is not None:
        for i in range(len(tokens)):
            tokens[i].insert(0, bos)
    random.shuffle(tokens)

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
    for input_ids in tokens:
        # get logits
        with torch.no_grad():
            y1 = model1(input_ids)[0]
            y2 = model2(input_ids)[0]

        # kl divergence
        for i in range(len(y1)):
            y1_probs = torch.softmax(y1[i], dim=-1)
            y2_probs = torch.softmax(y2[i], dim=-1)
            kl_div = torch.sum(rel_entr(y1_probs, y2_probs))
            kl_div = torch.nan_to_num(kl_div)
            kls.append(kl_div)
        
        # stats
        eval_argmax = torch.argmax(y2, axis=-1)
        ref_argmax = torch.argmax(y1, axis=-1)
        eval_part5 = torch.topk(y2, 5, dim=-1)
        ref_part5 = torch.topk(y1, 5, dim=-1)
        eval_part10 = torch.topk(y2, 10, dim=-1)
        ref_part10 = torch.topk(y1, 10, dim=-1)
        top1 += sum([eval_argmax[i] == ref_argmax[i] for i in range(len(y1))])
        top5 += sum([ref_argmax[i] in eval_part5[i] for i in range(len(y1))])
        top10 += sum([ref_argmax[i] in eval_part10[i] for i in range(len(y1))])
        eval_top5 += sum([eval_argmax[i] in ref_part5[i] for i in range(len(y1))])
        eval_top10 += sum([eval_argmax[i] in ref_part10[i] for i in range(len(y1))])
        print(f"[{i}] kl {torch.mean(kls):.4g}, top1 {top1 / samples:.4g}", flush=True)
    
    student_t = torch.distributions.studentT.StudentT(1 - alpha/2)
    student_t(samples)