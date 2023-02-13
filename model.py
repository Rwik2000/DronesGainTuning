import torch.nn as nn
import torch.nn.functional as F
import torch

class GainTune(nn.Module):
    def __init__(self, batch_size=1):
        super().__init__()
        self.flag = 1
        self.batch_size = batch_size
        # self.h = torch.randn(2, batch_size, 3)
        self.rnn = nn.RNN(5, 3, 3, bidirectional=False, batch_first=True)
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        # if self.flag:
        #     h = self.h0
        #     self.flag = 0
        # print(h)
        x = x.float()
        h = self.init_hidden()

        x,h = self.rnn(x,h)
        x = self.linear(x.squeeze(1))
        return x,h 

    def init_hidden(self,):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(3, self.batch_size, 3)
        return hidden