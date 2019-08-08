import os, sys, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics

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

# x = cPickle.load(open(loadpath, "rb"))
with open(loadpath, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    x = u.load()

train, val, test = x[0], x[1], x[2]
train_lab, val_lab, test_lab = x[6], x[7], x[8]
wordtoix, ixtoword = x[9], x[10]
del x
print("load data finished")

test_lab = np.array(test_lab, dtype='float32')
opt.n_words = len(ixtoword)


# opt.W_emb = np.array(cPickle.load(open(embpath, 'rb')),dtype='float32')[0]
with open(embpath, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    embedding_vec = u.load()
opt.W_emb = np.array(embedding_vec, dtype='float32')[0]
opt.W_class_emb =  load_class_embedding( wordtoix, opt)
uidx = 0
max_val_accuracy = 0.
max_test_accuracy = 0.

# Load model
model = LSTMNet(opt).to(device)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(sys.argv[1]))
else:
    model.load_state_dict(torch.load(sys.argv[1],map_location='cpu'))

# Get 1000 samples to test
test_sent, test_mask = prepare_data_for_emb(test[:1000], opt)
test_sent = torch.LongTensor(test_sent).to(device)
test_mask = torch.tensor(test_mask).to(device)

test_lab = np.array(test_lab[:1000])
test_lab = test_lab.reshape((len(test_lab), opt.num_class))

logits = model(test_sent, opt)

prob = nn.Softmax()(logits).to(device)
correct_prediction = torch.eq(torch.argmax(prob, 1).to(device), torch.argmax(torch.tensor(test_lab), 1).to(device)).to(device)
accuracy = torch.mean(correct_prediction.type(torch.float64)).to(device)
print("Test accuracy %f " % accuracy)

predictions = np.array(torch.argmax(prob, 1).cpu())
ground_truths = np.array(torch.argmax(torch.tensor(test_lab), 1).cpu())
print(metrics.confusion_matrix(predictions, ground_truths))
