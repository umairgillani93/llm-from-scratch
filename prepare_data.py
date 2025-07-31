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
        #print(dataset.__len__())
        return DataLoader(
                dataset,
                batch_size = batch_size,
                shuffle = shuffle,
                drop_last = drop_last,
                num_workers = num_workers)
    

class TokenEmbeddingLayer:
    def __init__(self, vocab_size, emb_dim):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def create_token_embedding(self, input_tensor):
        return self.embedding(input_tensor)

class PositionEmbeddingLayer:
    def __init__(self, context_size, emb_dim):
        self.context_size = context_size
        self.emb_dim = emb_dim
        self.pos_embedding = nn.Embedding(context_size, emb_dim)

    def create_pos_embedding(self, input_tensor):
        input_tensor = torch.clamp(input_tensor, min=0, max=self.context_size - 1)
        return self.pos_embedding(input_tensor)



if __name__ == "__main__":
    txt = """
    The invention of the internet revolutionized communication, creating an interconnectedness among people across the globe. With just a few clicks, individuals can access an almost limitless repository of information, engage in complex interactions, and participate in various online ecosystems. The interactivity of digital platforms has led to new forms of socializing, such as virtual meetings, gaming, and social media.

    Technological advancements continue to transform traditional industries. Artificial intelligence (AI) has become a cornerstone of modern innovation, fueling automation, machine learning, and predictive analytics. These tools are applied across sectors like healthcare, finance, and education, contributing to greater efficiency, reducing human error, and improving outcomes. However, such progress raises ethical concerns, especially regarding privacy and data security.

    Humanity faces challenges in the realm of environmental sustainability. The consequences of climate change are becoming more apparent with rising global temperatures and melting glaciers. In response, governments and industries are pushing for sustainable practices in energy production. Renewable energy sources, like solar and wind power, represent a critical step toward mitigating the effects of human activity on the planet.

    Philosophers have long pondered the meaning of life, existence, and consciousness. From ancient thinkers like Plato and Aristotle to modern theorists such as Descartes and Nietzsche, the exploration of human nature and reality has been central to philosophical inquiry. These ideas have influenced ethics, political theory, and societal structures.
"""
    max_len = 4
    stride = 2
    batch_size = 8
    tokenizer = CustomTokenizer(vocab)
    input_ids = tokenizer.encode(txt)
    emb_dim = 256
    vocab_size = len(vocab) 
    ld = LoadData.data_loader(
            txt,
            input_ids
            )
    token_emb_layer = TokenEmbeddingLayer(vocab_size, emb_dim)
    pos_emb_layer = PositionEmbeddingLayer(max_len, emb_dim)
    for input_, target in ld:
        input_emb = token_emb_layer.create_token_embedding(input_) + pos_emb_layer.create_pos_embedding(input_)
        target_emb = token_emb_layer.create_token_embedding(target) + pos_emb_layer.create_pos_embedding(target)
        print(f'input emb: {input_emb}')
        print(f'target emb: {target}')
        print("=============" * 10)




