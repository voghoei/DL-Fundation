# -*- coding: utf-8 -*-

import os, sys, cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import _uniout

from model import *
from utils import *

if len(sys.argv) != 2:
    sys.exit("Usage: $ python testing.py <path to model file, ending with .pt>")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


opt = Options() # Please specify the options in utils.py file
loadpath = "../../data/word_dic_2class.p"
embpath = "../../data/word_emb_2class_ver_1.0.p"
opt.num_class = 2
opt.class_name = ['normal', 'ill']

x = cPickle.load(open(loadpath, "rb"))
train, val, test = x[0], x[1], x[2]
train_lab, val_lab, test_lab = x[6], x[7], x[8]
wordtoix, ixtoword = x[9], x[10]

del x
print("load data finished")

test_lab = np.array(test_lab, dtype='float32')
opt.n_words = len(ixtoword)


opt.W_emb = np.array(cPickle.load(open(embpath, 'rb')),dtype='float32')[0]
opt.W_class_emb =  load_class_embedding( wordtoix, opt)
uidx = 0
max_val_accuracy = 0.
max_test_accuracy = 0.

# Load model
model = LeamNet(opt).to(device)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(sys.argv[1]))
else:
    model.load_state_dict(torch.load(sys.argv[1],map_location='cpu'))

# Get testing samples to test
test_sent, test_mask = prepare_data_for_emb(test, opt)
test_sent = torch.LongTensor(test_sent).to(device)
test_mask = torch.tensor(test_mask).to(device)

test_lab = np.array(test_lab)
test_lab = test_lab.reshape((len(test_lab), opt.num_class))

logits,logits_class, Att_v = model(test_sent, test_mask, opt)

prob = nn.Softmax()(logits).to(device)
correct_prediction = torch.eq(torch.argmax(prob, 1).to(device), torch.argmax(torch.tensor(test_lab), 1).to(device)).to(device)
accuracy = torch.mean(correct_prediction.type(torch.float64)).to(device)
print("Test accuracy %f " % accuracy)

# Saving the predictions and the attentions
if not os.path.isdir(opt.save_path+"attentions"):
    os.mkdir(opt.save_path+"attentions")

test_sents_words = []
for j in np.array(test_sent.cpu()):
    sent = [ixtoword[i] for i in j]
    test_sents_words.append(sent)

predictions = np.array(torch.argmax(prob, 1).cpu())
ground_truths = np.array(torch.argmax(torch.tensor(test_lab), 1).cpu())

# Generating attention map for all testing samples.
for i, words, alphas_values, ground_truth, prediction in zip(range(Att_v.shape[0]), test_sents_words, Att_v, ground_truths, predictions):
    with open(opt.save_path+"attentions/visualization_{}.html".format(i), "w") as html_file:
        for word, alpha in zip(words, alphas_values / alphas_values.max()):
            if word == "END":
                continue
            html_file.write('<font style="background: rgba(255, 255, 0, %f)"><meta charset="UTF-8">%s</font>\n' % (alpha*1.1, word.encode('utf-8')))
        html_file.write('<br/><font><meta charset="UTF-8">Ground Truth: %s, Prediction: %s</font>\n' % (opt.class_name[ground_truth].encode('utf-8'), opt.class_name[prediction].encode('utf-8')))


