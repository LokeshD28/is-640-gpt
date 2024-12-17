#import the torch function
import torch
import torch.nn.functional as F
import torch.nn as nn


LEARNING_RATE = 0.001
BATCH_SIZE = 16 
BLOCK_SIZE = 8 
EVAL_INTERVAL = 10
EVAL_ITERS = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer() :
    def __init__( self,data,gpt_model) :
        self.data = data
        self.model = gpt_model
        self.m =  self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)

    
    def __init_train_val_data(self) :
            n = int(0.9*len(self.data)) # first 90% will be train, rest val
            self.train_data = self.data[:n]
            self. val_data = self.data[n:]

    # data loading
    
