import os
import sys
import re


with open('./text.txt', 'r') as f:
    text = f.read()
output = re.split(r'([,.:;?_!"()\']|--|\s)', text)
output = [x.strip() for x in output if x.strip()]
all_words = sorted(set(output))
unk_tokens = ["<|unk|>", "<|endoftext|>"]
all_words.extend(unk_tokens)
vocab = {c:i for i,c in enumerate(all_words)}

#print(len(vocab))

#tokenizer = CustomTokenizer(vocab)
#text1 = "let's try with this sentence first."
#text2 = "she ran faster than she ever thought"
#full_text = " <|endoftext|> ".join([text1, text2])
#print(full_text)
#ids = tokenizer.encode(full_text)
#print(tokenizer.decode(ids))

