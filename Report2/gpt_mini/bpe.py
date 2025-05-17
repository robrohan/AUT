# your subclass of torch.utils.data.Dataset that emits example
# torch LongTensor of lengths up to 1024, with integers from [0,50257)
import torch
import json
from torch.utils.data import Dataset
import gpt_mini.midi_encoder as me
import numpy as np
import pickle

class CharDataset(Dataset):
    """
    This is from the original GPT example, and since we are turning midi into
    plain  32bit integers, this can work out of the box. The final example in
    this repo uses the text version because it can make for a smaller set of
    tokens since sentencepiece will use fragments of text instead of single
    "integer notes" - it does work though, take it for a spin
    """
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f"data as {data_size} characters {vocab_size} unique")

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class MidiDataset(Dataset):
    """
    Break midi into tokens that we can encode and decode.
    """
    def __init__(self, file_path, tokenizer_path: str, max_length=128):
        self.file_path = file_path
        self.data = self._load_data(file_path)
        # self.tokenizer = spm.SentencePieceProcessor()
        # self.tokenizer.load(tokenizer_path)
        f = open(tokenizer_path, 'rb')
        self.tokenizer = pickle.load(f)
        f.close()
        self.max_length = max_length

    def _load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx].strip()
        tokens = self.tokenizer.encode(file)
        tokens = tokens[0].ids
        # Truncate if longer
        tokens = tokens[: self.max_length]
        if len(tokens) < self.max_length:
            # Pad if shorter
            tokens = tokens + [0] * (self.max_length - len(tokens))

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y
