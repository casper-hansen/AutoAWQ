import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

class Perplexity:
    def __init__(self, model, tokenizer, dataset_path='wikitext', dataset_name=None, split='test', text_column='text'):
        self._model = model
        self._tokenizer = tokenizer
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._split = split
        self._text_column = text_column
        self._text = self._prepare_data()
        self._total_nll = 0.0
        self._count = 0
        self.max_length = 2048

    def _get_device(self):
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda:0'
        else:
            return 'cpu'

    def _prepare_data(self):
        if self._dataset_path == 'wikitext':
            self._dataset_name = 'wikitext-2-raw-v1'
        
        data = load_dataset(self._dataset_path, self._dataset_name, split=self._split)
        text_list = [' \n' if s == '' else s for s in data[self._text_column]]
        return ''.join(text_list)

    def calculate_perplexity(self, stride=512, tokens=None, n_rounds=-1):
        if tokens is None:
            tokens = self._tokenizer(self._text, truncation=False, return_tensors="pt").input_ids.to(self._model.device)
        
        seq_len = tokens.size(1)
        prev_end_loc = 0
        all_perplexity = []

        if n_rounds != -1:
            full_range = [list(range(0, seq_len, stride))[:n_rounds]]
        else:
            full_range = range(0, seq_len, stride)

        with tqdm(full_range, desc="Perplexity: - ") as progress:
            for begin_loc in progress:
                nll = self._process_batch(tokens, begin_loc, prev_end_loc)
                self._total_nll += nll.item()
                self._count += 1
                prev_end_loc = min(begin_loc + self.max_length, seq_len)

                # Compute the running perplexity and update the tqdm description
                running_perplexity = np.exp(self._total_nll / self._count)
                all_perplexity.append(running_perplexity)
                progress.set_description(f"Perplexity: {running_perplexity:.4f}")

        return all_perplexity

    def _process_batch(self, tokens, begin_loc, prev_end_loc):
        end_loc = min(begin_loc + self.max_length, tokens.size(1))
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = tokens[:, begin_loc:end_loc].to(self._model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = self._model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        return neg_log_likelihood