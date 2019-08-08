import cPickle
import numpy as np

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

model = loadGloveModel("../../data/glove.6B.300d.txt")

vector_size = 300

loadpath = "../../data/word_dic_2class.p"

data = cPickle.load(open(loadpath, "rb"))
X_train, X_val, X_test = data[0], data[1], data[2]
y_train, y_val, y_test = data[6], data[7], data[8]
wordtoix, ixtoword = data[9], data[10]

#print X_val

embedding_vectors = np.random.uniform(-0.25, 0.25, (len(wordtoix), vector_size))
cc_vocab = list(model.keys())
count = 0
mis_count = 0
for word in wordtoix.keys():
    idx = wordtoix.get(word)
    if word in cc_vocab:
        embedding_vectors[idx] = model[word]
        count += 1
    else:
        mis_count += 1
    print count
print("num of vocab in embedding vec: {}".format(count))
print("num of vocab not in embedding vec: {}".format(mis_count))

cPickle.dump([embedding_vectors.astype(np.float32)], open("../../data/word_emb_2class_ver_1.0.p", "wb"))
