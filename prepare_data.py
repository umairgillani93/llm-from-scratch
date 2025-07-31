import os 
import torch
from tokenizer import CustomTokenizer
from torch.utils.data import Dataset, DataLoader
from constants import vocab

# create your own Dataset class of the type torch Datset
class CustomDataset(Dataset):
    def __init__(self, txt, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []

        # gives the token ids
        self.token_ids = tokenizer.encode(txt)

        for i in range(0, len(self.token_ids) - max_len, stride):
            input_chunk = self.token_ids[i: i + max_len]
            target_chunk = self.token_ids[i + 1: i + max_len + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def data_loader(txt, tokenizer, batch_size = 2, max_len = 4, stride = 1,
        shuffle = True, drop_last = True, num_workers = 0):
    dataset = CustomDataset(txt, tokenizer, max_len, stride)
    print(dataset.__len__())
    return DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            drop_last = drop_last,
            num_workers = num_workers)
    


if __name__ == "__main__":
    txt = """Artificial intelligence is transforming industries. 
    From healthcare to finance, AI is revolutionizing how businesses operate. 
    The future of AI promises even greater advancements. However, ethical considerations must be addressed."""
    max_len = 3
    stride = 2
    tokenizer = CustomTokenizer(vocab)
    data_loader = data_loader(
            txt,
            tokenizer
            )


    for input_, target in data_loader:
        print(f'input: {input_}')
        print(f'target: {target}')

