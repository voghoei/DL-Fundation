"""
    This structure is highly inspired by
    https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
"""

import os, sys, cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SimpleCNN(nn.Module):
    def __init__(self, opt):
        """
        CNN constructor. Define the embedding layer, 3 conv3D layer with different window sizes followed one dense layer.
        
        @param: opt: type, object
                opt.maxlen = 305
                opt.n_words = None
                opt.embed_size = 300
                opt.batch_size = 40
                opt.max_epochs = 30
                opt.dropout = 0.5
                opt.ngram = 31 # Currently only support odd numbers
                opt.n_layers = 2
                opt.filter_sizes = [2,3,4]
                opt.n_filters = 100
        return: void
       
        """
        super(SimpleCNN, self).__init__()
        
        self.embed_x = nn.Embedding(opt.n_words, opt.embed_size)
        self.embed_x.weight.data = self.embed_x.weight.data + torch.tensor(opt.W_emb,requires_grad=False)

        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1,
                                              out_channels = opt.n_filters,
                                              kernel_size = (fs, opt.embed_size))
                                    for fs in opt.filter_sizes
                                    ])
            
        self.fc = nn.Linear(len(opt.filter_sizes) * opt.n_filters, opt.num_class)
            
        self.dropout = nn.Dropout(opt.dropout)
    
    def forward(self, x, opt):
        """
        Forward function of the CNN structure. Given a Tensor of input data and return a Tensor of output data.
    
        @param: text: type, tensor with shape [batch size, sent len].
        @param: opt: type, object.
        return: tensor of output scores with shape [batch size, num class].
        """
        # transpose the input text tensor
        x_vectors = self.embed_x(x)
        # x_vectors = [sent len, batch size]
        
        embedded = x_vectors.unsqueeze(1)
        #embedded = [sentence len, batch size, embedding dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        
        return self.fc(cat)
        
