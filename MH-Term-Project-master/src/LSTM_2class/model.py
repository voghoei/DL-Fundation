"""
This structure is highly inspired by 
https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
"""

import os, sys, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LSTMNet(nn.Module):
    def __init__(self, opt):
        """
        LSTM constructor. Define the embedding layer, LSTM block with 2 layers and using bidirectional structure

        @param: opt: type, object
                opt.maxlen = 305
                opt.n_words = None
                opt.embed_size = 300
                opt.batch_size = 40
                opt.max_epochs = 30
                opt.dropout = 0.5
                opt.ngram = 31 # Currently only support odd numbers
                opt.hidden_dim = 256
                opt.n_layers = 2
                opt.bidirectional = True
        return: void
        """
        
        super().__init__()

        self.embedding = nn.Embedding(opt.n_words, opt.embed_size, padding_idx=opt.ngram//2)
        self.embedding.weight.data = self.embedding.weight.data + torch.tensor(opt.W_emb)
        
        self.rnn = nn.LSTM(opt.embed_size, 
                           opt.hidden_dim, 
                           num_layers=opt.n_layers, 
                           bidirectional=opt.bidirectional, 
                           dropout=opt.dropout)
        
        self.fc = nn.Linear(opt.hidden_dim * opt.n_layers, opt.num_class)
        
        self.dropout = nn.Dropout(opt.dropout)
        

    def forward(self, text, opt):
        """
        Forward function of the LSTM structure. Given a Tensor of input data and return a Tensor of output data

        @param: text: type, tensor with shape [batch size, sent len]
        @param: opt: type, object
        return: tensor of output scores with shape [batch size, num class]
        """
        
        # transpose the input text tensor
        text = torch.t(text)
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        #embedded = [sentence len, batch size, embedding dim]
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, torch.tensor([opt.maxlen] * opt.batch_size))
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #output = [sentence len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden.squeeze(0))