import os 
import torch
import torch.nn as nn
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


class LoadData:
    def __init__(self):
        pass
    def data_loader(txt, input_ids, batch_size = 8, max_len = 4, stride = 1,
            shuffle = True, drop_last = True, num_workers = 0):
        dataset = CustomDataset(txt, tokenizer, max_len, stride)
        print(dataset.__len__())
        return DataLoader(
                dataset,
                batch_size = batch_size,
                shuffle = shuffle,
                drop_last = drop_last,
                num_workers = num_workers)
    

class EmbeddingLayer:
    def __init__(self, vocab_size, emb_dim):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def create_embedding(self, input_tensor):
        return self.embedding(input_tensor)



if __name__ == "__main__":
    txt = """
    “wait!” she screamed. “don’t go!” but he was already halfway out the door.
#        She ran, faster than she ever thought possible, but stopped. Why?
"""
    max_len = 3
    stride = 2
    tokenizer = CustomTokenizer(vocab)
    input_ids = tokenizer.encode(txt)
    emb_dim = 13
    vocab_size = len(vocab) 
    ld = LoadData.data_loader(
            txt,
            input_ids
            )
    emb_layer = EmbeddingLayer(vocab_size, emb_dim)
    for input_, target in ld:
        print(f'input embeddings: {emb_layer.create_embedding(input_)}')
        print(f'target embeddings: {emb_layer.create_embedding(target)}')
        print("=============" * 10)




