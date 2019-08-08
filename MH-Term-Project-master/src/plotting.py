import sys
import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) != 2:
    sys.exit("Usage: $ python plotting.py <path to training trace file in ../model_trace/ folder, ending with npy>")

history_track = np.load(sys.argv[1])

train_loss = history_track[0]
train_acc = history_track[1]
val_acc = history_track[2]

t = np.array(train_loss)[:,0]
loss = np.array(train_loss)[:,1]

acc_t = np.array(train_acc)[:,0]
train_acc_y = np.array(train_acc)[:,1]
val_acc_y = np.array(val_acc)[:,1]

fig = plt.figure(figsize=(14,10))
plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma']
plt.plot(acc_t,val_acc_y,linewidth=3.0)
plt.plot(acc_t,train_acc_y,linewidth=3.0)
plt.legend(['Validation Accuracy', 'Training Accuracy'], loc='upper left')
plt.xlabel("Training Steps")
plt.ylabel("Accuracy")
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.title("Training & Validation accuracy")
plt.savefig('accuracy.png')

fig = plt.figure(figsize=(14,10))
plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma']
plt.plot(t,loss,linewidth=3.0)
plt.legend(['Training Loss'], loc='upper right')
plt.xlabel("Training Steps")
plt.ylabel("Regularized Loss")
plt.title("Training loss")
plt.savefig('loss.png')
