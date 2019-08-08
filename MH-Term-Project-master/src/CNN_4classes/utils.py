import sys
import os
import numpy as np

class Options(object):
    def __init__(self):
        self.fix_emb = True
        self.restore = False
        self.W_emb = None
        self.W_class_emb = None
        self.maxlen = 305
        self.n_words = None
        self.embed_size = 300
        self.lr = 5e-4
        self.batch_size = 40
        self.max_epochs = 300
        self.dropout = 0.5
        self.part_data = False
        self.portion = 1.0
        self.save_path = "../model_trace/"
        self.log_path = "./log/"
        self.print_freq = 10
        self.valid_freq = 10

        self.optimizer = 'Adam'
        self.clip_grad = None
        self.class_penalty = 1.0
        self.ngram = 31 # Currently only support odd numbers
        self.H_dis = 600
    
        self.filter_sizes = [2,3,4]
        self.n_filters = 100


    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value


def load_class_embedding( wordtoidx, opt):
    print("load class embedding")
    name_list = [ k.lower().split(' ') for k in opt.class_name]
    id_list = [ [ wordtoidx[i] for i in l] for l in name_list]
    #print(len(opt.W_emb[0]))
    value_list = [ [ opt.W_emb[i] for i in l]    for l in id_list]
    value_mean = [ np.mean(l,0)  for l in value_list]
    return np.asarray(value_mean)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
    return zip(range(len(minibatches)), minibatches)

def prepare_data_for_emb(seqs_x, opt):
    maxlen = opt.maxlen
    lengths_x = [len(s) for s in seqs_x]
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
            else:
                new_seqs_x.append(s_x[:maxlen])
                new_lengths_x.append(maxlen)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
        x_mask[idx, :lengths_x[idx]] = 1. # change to remove the real END token
    return x, x_mask


def read_file(file_name):
    f = open(file_name, "r")
    return f.read().split('\n')
    #return f.read().split('\n')


def convert_word_to_ix(data,wordtoix):
    result = []
    for sent in data:
        temp = []
        for w in sent:
            if w in wordtoix:
                temp.append(wordtoix[w])
            else:
                temp.append(1)
        temp.append(0)
        result.append(temp)
    return result
