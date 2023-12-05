import torch
import logging
from typing import List, Union
from datasets import load_dataset


def get_calib_dataset(data: Union[str, List[str]] = "pileval",
                      tokenizer=None,
                      n_samples=512,
                      block_size=512,
                      split="train",
                      text_column="text",
                      custom_calib_file=None):
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup",
                                   split="validation")
        elif data == "custom":
            dataset = load_dataset("json",
                                   data_files=custom_calib_file,
                                   split="train")
            if text_column != "input_ids":
                print(
                    f'warning: text_column for custom medical dataset maybe wrong, we fix it as input_ids automatically'
                )
            text_column = "input_ids"
        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=0)  # same as LMDeploy

    elif isinstance(data, list):
        dataset = [{text_column: text} for text in data]
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element.")

    samples = []
    n_run = 0
    for data in dataset:
        line = data[text_column]
        if isinstance(line, str):
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        else:
            line_encoded = line.copy()
        if len(line_encoded) > block_size:
            if text_column == 'input_ids':
                print('have been concated')
            else:
                continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    logging.debug(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size:(i + 1) * block_size]
        for i in range(n_split)
    ]
