import torch 

# Define a class to process text data for neural network training.
class Data:
    # Constructor takes a file path and reads text from the file.
    def __init__(self, file_path):
        # Open the file with the specified path in read mode with UTF-8 encoding.
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()  # Read the entire file content into a string.
        
        # Create a sorted list of the unique characters in the text.
        each_char = sorted(list(set(self.text)))
        
        # Determine the number of unique characters.
        self.vocab_size = len(each_char)
        
        # Create a dictionary to convert characters to integer indices.
        self.stoi = {ch: i for i, ch in enumerate(each_char)} 
        
        # Create a dictionary to convert integer indices back to characters.
        self.itos = {i: ch for i, ch in enumerate(each_char)}
    
    # Method to encode a string into a list of indices based on the character.
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    # Method to decode a list of indices back into a string.
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
    
    # Method to get the encoded data as a PyTorch tensor.
    def get_data(self):
        # Convert the encoded text (list of indices) to a tensor of type long.
        return torch.tensor(self.encode(self.text), dtype=torch.long)



    

