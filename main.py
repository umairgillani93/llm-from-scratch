import os
import sys
import re 

text = """
“Wait!” she screamed. “Don’t go!” But he was already halfway out the door.
She ran, faster than she ever thought possible, but stopped. Why?
He wasn’t running from her. He was running to her.
“Are you—?” she gasped. “I thought you—”
He smiled, quietly, holding out a box. “It’s for you.”
She froze. “What is it?”
A ring. No words needed.
Her breath caught, and then—“Yes.”
They kissed, finally, with the whole world quiet around them.
And they lived happily ever after.
"""

# tokenizer the words and punctuations
output = re.split(r'([,.:;?_!"()\']|--|\s)', text)
output = [x.strip() for x in output if x.strip()]

# create vocabl

vocab_size = len(output)
all_words = sorted(set(output))

unknown_tokens = ["<|unk|>", "<|endoftext|>"]
all_words.extend(unknown_tokens)
vocab = {c:i for i,c in enumerate(all_words)}

#print(vocab)
# replace unknown words by <|unk|> tokens

user_text = "hey this is a temp text to see how well tokenizer is doing"
tokenized_user_text = [
    vocab[x.strip()] if x.strip() not in unknown_tokens else vocab["<|unk|>"]
    for x in user_text.split()  # Split by space
    if x.strip()  # Avoid processing empty strings
]

print(tokenized_user_text)
