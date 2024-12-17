import torch 
#Processes text into tensor with encoding.
class Data:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        each_char = sorted(list(set(self.text)))
        self.vocab_size = len(each_char)
        self.stoi = {ch: i for i, ch in enumerate(each_char)} 
        self.itos = {i: ch for i, ch in enumerate(each_char)}
    

