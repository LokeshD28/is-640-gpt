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
    def __get_batch(self,split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    #This is a PyTorch decorator that disables gradient tracking during the execution of the method it decorates.
    @torch.no_grad()
    def __estimate_loss(self,):
            out = {}
            self.model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(EVAL_ITERS)
                for k in range(EVAL_ITERS):
                    X, Y = self.__get_batch(split)
                    logits, loss = self.model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            self.model.train()
            return out

    
    def print_learning_rate(self):
        # Access the current learning rate from the optimizer
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
            print(f"Current learning rate: {current_lr}")
            
    def train(self,train_iterations) :
 
        self.__init_train_val_data()
        for iter in range(train_iterations):

            
            if iter % EVAL_INTERVAL == 0:
                self.print_learning_rate()
                losses = self.__estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = self.__get_batch('train')


            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        
       return  torch.zeros((1, 1), dtype=torch.long, device=device)

            


    
