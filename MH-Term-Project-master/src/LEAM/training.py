# -*- coding: utf-8 -*-

import os, sys, cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import _uniout

from model import *
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


opt = Options() # Please specify the options in utils.py file
loadpath = "../../data/word_dic.p"
embpath = "../../data/word_emb_ver_1.0.p"
opt.num_class = 4
opt.class_name = ['normal', 'depression', 'traumatic', 'bipolar']

x = cPickle.load(open(loadpath, "rb"))
train, val, test = x[0], x[1], x[2]
train_lab, val_lab, test_lab = x[6], x[7], x[8]
wordtoix, ixtoword = x[9], x[10]
#print(train_lab)
#train_lab = [np.array([[1.],[0.],[0.],[0.]]) if x == 'normal' else np.array([[0.],[1.],[0.],[0.]]) if x == 'depression' else np.array([[0.],[0.],[1.],[0.]]) if x == 'bipolar' else np.array([[0.],[0.],[0.],[1.]]) for x in train_lab]
#val_lab = [np.array([[1.],[0.],[0.],[0.]]) if x == 'normal' else np.array([[0.],[1.],[0.],[0.]]) if x == 'depression' else np.array([[0.],[0.],[1.],[0.]]) if x == 'bipolar' else np.array([[0.],[0.],[0.],[1.]]) for x in val_lab]
#test_lab = [np.array([[1.],[0.],[0.],[0.]]) if x == 'normal' else np.array([[0.],[1.],[0.],[0.]]) if x == 'depression' else np.array([[0.],[0.],[1.],[0.]]) if x == 'bipolar' else np.array([[0.],[0.],[0.],[1.]]) for x in test_lab]
del x
print("load data finished")
#print(val_lab)

train_lab = np.array(train_lab, dtype='float32')
val_lab = np.array(val_lab, dtype='float32')
test_lab = np.array(test_lab, dtype='float32')
opt.n_words = len(ixtoword)


opt.W_emb = np.array(cPickle.load(open(embpath, 'rb')),dtype='float32')[0]
opt.W_class_emb = load_class_embedding( wordtoix, opt)
uidx = 0
max_val_accuracy = 0.
max_test_accuracy = 0.

# Build model
model = LeamNet(opt).to(device)

# Component of loss function
loss_func = nn.BCEWithLogitsLoss().to(device)

# Optimizer setting
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

# Preparation of regularization in loss
class_y = torch.tensor(np.identity(opt.num_class)).type(torch.FloatTensor).to(device)

if not os.path.isdir(opt.save_path):
    os.mkdir(opt.save_path)

train_loss = []
train_acc = []
val_acc = []

for epoch in range(opt.max_epochs):
    print("Starting epoch %d over %d" % (epoch, opt.max_epochs))
    # Get minibatch
    kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
    for _, train_index in kf:
        uidx += 1 
        # Data preparation
        sents = [train[t] for t in train_index]
        x_labels = [train_lab[t] for t in train_index]
        x_labels = np.array(x_labels)
        x_labels = x_labels.reshape((len(x_labels), opt.num_class))
        x_batch, x_batch_mask = prepare_data_for_emb(sents, opt)
        x_batch = torch.LongTensor(x_batch).to(device)
        x_batch_mask = torch.tensor(x_batch_mask).to(device)
        
        # Forward model
        logits, logits_class, Att_v = model(x_batch, x_batch_mask, opt)
        
        # Loss calc
        loss1 = loss_func(logits, torch.tensor(x_labels).to(device)).to(device)
        loss2 = loss_func(logits_class, class_y).to(device)

        loss = loss1 + opt.class_penalty * loss2
        
        train_loss.append([uidx, loss.item()])
        
        optimizer.zero_grad()
        
        # Model optimization
        loss.backward()
        optimizer.step()
        
        if uidx % opt.valid_freq == 0:
            print("Iteration %d: Training loss %f " % (uidx, loss))
            
            # Acc Calc
        
            prob = nn.Softmax()(logits).to(device)
            correct_prediction = torch.eq(torch.argmax(prob, 1).to(device), torch.argmax(torch.tensor(x_labels).to(device), 1)).to(device)
            accuracy = torch.mean(correct_prediction.type(torch.float64)).to(device)
            print("Train accuracy %f " % accuracy)
            train_acc.append([uidx, accuracy.item()])
            
            
            # Val
            val_correct = 0.0
            # sample evaluate accuaccy on val data
            kf_val = get_minibatches_idx(len(val), opt.batch_size, shuffle=True)
            for _, val_index in kf_val:

                # val set preparation
                val_sents = [val[t] for t in val_index]
                val_labels = [val_lab[t] for t in val_index]
                val_labels = np.array(val_labels)
                val_labels = val_labels.reshape((len(val_labels), opt.num_class))
                x_val_batch, x_val_batch_mask = prepare_data_for_emb(val_sents, opt)
                
                x_val_batch = torch.LongTensor(x_val_batch).to(device)
                x_val_batch_mask = torch.tensor(x_val_batch_mask).to(device)

                # Evaluations
                logits_val, logits_class_val, Att_v  = model(x_val_batch, x_val_batch_mask, opt)

                # Acc calc

                prob_val = nn.Softmax()(logits_val).to(device)
                correct_prediction_val = torch.eq(torch.argmax(prob_val, 1).to(device), torch.argmax(torch.tensor(val_labels).to(device), 1)).to(device)
                # print(torch.argmax(torch.tensor(val_labels).to(device), 1))
                val_accuracy = torch.mean(correct_prediction_val.type(torch.float64)).to(device)

                val_correct += val_accuracy * len(val_index)

            val_current_acc = val_correct / len(val)
            print("Validation accuracy %f " % val_current_acc)
            val_acc.append([uidx, val_current_acc.item()])
        
        
    torch.save(model.state_dict(), opt.save_path + "model_ver_1.1.0.pt")
    np.save(opt.save_path + "trace_history_ver_1.1.0.npy",[train_loss,train_acc,val_acc])

print("training finished!")
