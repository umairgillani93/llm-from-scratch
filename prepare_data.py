import os 
from tokenizer import CustomTokenizer
from torch.utils.data import Dataset, DataLoader
from constants import vocab

# create your own Dataset class of the type torch Datset
class CustomDataset:
    def __init__(self, txt, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []

        # gives the token ids
        self.token_ids = tokenizer.encode(txt)

        for i in range(0, len(self.token_ids)):
            print(f'inputs: {self.token_ids[:i]}')
            print(f'labels: {self.token_ids[i]}')

if __name__ == "__main__":
    txt = """Artificial intelligence is transforming industries. 
    From healthcare to finance, AI is revolutionizing how businesses operate. 
    The future of AI promises even greater advancements. However, ethical considerations must be addressed."""
    max_len = 3
    stride = 2
    tokenizer = CustomTokenizer(vocab)
    cd = CustomDataset(txt, tokenizer, max_len, stride)
    print(cd.token_ids)

