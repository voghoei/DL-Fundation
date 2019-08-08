from sklearn.model_selection import train_test_split
import cPickle
import pandas as pd


def convert_word_to_ix(data,wordtoix):
    result = []
    for sent in data:
        temp = []
        for w in sent.split():
            if w in wordtoix:
                temp.append(wordtoix[w])
            else:
                temp.append(1)
        temp.append(0)
        result.append(temp)
    return result

data_path = "../../data/cleaned_data.csv"

df = pd.read_csv(data_path)
y = [[0,0] for _ in df['label']]
for i in range(len(df['label'].values)):
    if df['label'].values[i] > 0:
        y[i][1] = 1
    else:
        y[i][0] = 1



X = df['text'].values

vocab = {}

for i in range(len(X)):
    unique_words = set(X[i].split())
    for word in unique_words:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1

v = [a for a, b in vocab.iteritems() if b >= 2] # small frequency then abandon


# create ixtoword and wordtoix lists
ixtoword = {}
# period at the end of the sentence. make first dimension be end token
ixtoword[0] = 'END'
ixtoword[1] = 'UNK'
wordtoix = {}
wordtoix['END'] = 0
wordtoix['UNK'] = 1
ix = 2
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_val = X_train[400:]
y_val = y_train[400:]
print(X_val)
#print(y_train)
X_train = X_train[:400]
y_train = y_train[:400]

train_text = [''.join(s) for s in X_train]
val_text = [''.join(s) for s in X_val]
test_text = [''.join(s) for s in X_test]

X_train = convert_word_to_ix(X_train,wordtoix)
X_val = convert_word_to_ix(X_val,wordtoix)
X_test = convert_word_to_ix(X_test,wordtoix)

cPickle.dump([X_train, X_val, X_test, train_text, val_text, test_text, y_train, y_val, y_test, wordtoix, ixtoword], open("../../data/word_dic_2class.p", "wb"))
