import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, tokens, context_length=128):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        return max(0, len(self.tokens) - self.context_length - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + self.context_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def prepare_dataset(tokens, context_length=128):
    return TokenDataset(tokens, context_length=context_length)


def split_dataset(dataset, train_ratio=0.9):
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    return torch.utils.data.random_split(dataset, [n_train, n_val])
