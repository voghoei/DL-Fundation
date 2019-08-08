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
        # Define the layers
        self.embed_x = nn.Embedding(opt.n_words, opt.embed_size)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1,
                                              out_channels = opt.n_filters,
                                              kernel_size = (fs, opt.embed_size))
                                    for fs in opt.filter_sizes
                                    ])
            
        self.fc = nn.Linear(len(opt.filter_sizes) * opt.n_filters, 4)
            
        self.dropout = nn.Dropout(opt.dropout)
        
        # Init the weights
        self.embed_x.weight.data = self.embed_x.weight.data + torch.tensor(opt.W_emb,requires_grad=False)
    
    def forward(self, x, opt):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        # Embeddings
        x_vectors = self.embed_x(x) # b*s*e
        
        embedded = x_vectors.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        
        #cat = [batch size, n_filters * len(filter_sizes)]
        
        return self.fc(cat)
        
        
     

    

