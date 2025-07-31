import os 
import sys
import re

class CustomTokenizer:
    # __init_()
    # encode()
    # decode
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {s:i for i,s in vocab.items()}

    def encode(self, text):
        processed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        processed_text = [x.strip() for x in text.split() if x.strip()]
        processed_text = [x if x in self.str_to_int else "<|unk|>" 
                for x in processed_text]
        print(processed_text)
        
        ids = [self.str_to_int[x] for x in processed_text]

        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)

        return text

#if __name__ == "__main__":
#    text = """
#        “wait!” she screamed. “don’t go!” but he was already halfway out the door.
#        She ran, faster than she ever thought possible, but stopped. Why?
#        He wasn’t running from her. He was running to her.
#        “Are you—?” she gasped. “I thought you—”
#        He smiled, quietly, holding out a box. “It’s for you.”
#        She froze. “What is it?”
#        A ring. No words needed.
#        Her breath caught, and then—“Yes.”
#        They kissed, finally, with the whole world quiet around them.
#        And they lived happily ever after.
#    """
#    output = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#    output = [x.strip() for x in output if x.strip()]
#    all_words = sorted(set(output))
#    unk_tokens = ["<|unk|>", "<|endoftext|>"]
#    all_words.extend(unk_tokens)
#    vocab = {c:i for i,c in enumerate(all_words)}
#
#    tokenizer = CustomTokenizer(vocab)
#    text1 = "let's try with this sentence first."
#    text2 = "she ran faster than she ever thought"
#    full_text = " <|endoftext|> ".join([text1, text2])
#    print(full_text)
#    ids = tokenizer.encode(full_text)
#    print(tokenizer.decode(ids))
    
