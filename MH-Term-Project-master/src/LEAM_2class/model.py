import os, sys, cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LeamNet(nn.Module):
    def __init__(self, opt):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        # comment notation
        #  b: batch size, s: sequence length, e: embedding dim, c : num of class
        super(LeamNet, self).__init__()
        # Define the layers
        self.embed_x = nn.Embedding(opt.n_words, opt.embed_size) 
        self.embed_y = nn.Embedding(opt.num_class, opt.embed_size)
        self.att_conv = nn.Conv1d(opt.num_class,opt.num_class,kernel_size=opt.ngram,padding=opt.ngram/2)
        self.H1_dropout_x = nn.Dropout(opt.dropout)
        self.H1_x = nn.Linear(opt.embed_size, opt.H_dis)
        self.H2_dropout_x = nn.Dropout(opt.dropout)
        self.H2_x = nn.Linear(opt.H_dis, opt.num_class)
        self.H1_dropout_y = nn.Dropout(opt.dropout)
        self.H1_y = nn.Linear(opt.embed_size, opt.H_dis)
        self.H2_dropout_y = nn.Dropout(opt.dropout)
        self.H2_y = nn.Linear(opt.H_dis, opt.num_class)

        # Init the weights
        self.embed_x.weight.data = self.embed_x.weight.data + torch.tensor(opt.W_emb,requires_grad=True)
        self.embed_y.weight.data = self.embed_y.weight.data + torch.tensor(opt.W_class_emb,requires_grad=True)
        
        #self.embed_x.weight.data = self.embed_x.weight.data + torch.tensor(opt.W_emb,requires_grad=False)
        #self.embed_y.weight.data = self.embed_y.weight.data + torch.tensor(opt.W_class_emb,requires_grad=False)
    
    def forward(self, x, x_mask, opt):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        # Embeddings
        x_vectors = self.embed_x(x) # b*s*e
        # y_vectors = self.embed_y(y) # b*e
        
        # Attention Part
        W_class_tran = torch.t(self.embed_y.weight.data) #e*c
        x_mask = torch.unsqueeze(torch.tensor(x_mask),-1) #b*s*1
        x_vectors_1 = torch.mul(x_vectors, x_mask) #b*s*e
        
        #G = (C^T V) ./ G
        x_vectors_norm = F.normalize(x_vectors_1, p=2, dim=2) #b*s*e
        W_class_norm = F.normalize(W_class_tran, p=2, dim = 0) #e*c
        G = torch.matmul(x_vectors_norm, W_class_norm) #b*s*c
        G = G.permute(0, 2, 1) #b*c*s
        x_full_vectors = x_vectors
        
        # ReLU + Conv
        Att_v = self.att_conv(G).clamp(min=0) #b*c*s
        Att_v = Att_v.permute(0, 2, 1)
        
        # Max pooling
        Att_v,indx = torch.max(Att_v, dim=-1, keepdim=True)
        
        # Softmax
        exp_logits = torch.exp(Att_v)
        exp_logits_masked = torch.mul(exp_logits, x_mask)
        exp_logits_sum = torch.sum(exp_logits_masked, dim=1)
        exp_logits_sum = torch.unsqueeze(exp_logits_sum,1)
        partial_softmax_score = torch.div(exp_logits_masked, exp_logits_sum)

        # Get attentive weight
        x_att = torch.mul(x_full_vectors, partial_softmax_score)
        H_enc = torch.sum(x_att, dim=1)
        H_enc = torch.squeeze(H_enc)
        
        # 2 layer nn classification
        H1_out_x = self.H1_x(self.H1_dropout_x(H_enc)).clamp(min=0)
        logits = self.H2_x(self.H2_dropout_x(H1_out_x))
        
        H1_out_y = self.H1_y(self.H1_dropout_y(self.embed_y.weight.data)).clamp(min=0)
        logits_class = self.H2_y(self.H2_dropout_y(H1_out_y))
        
        
        return logits, logits_class, Att_v
