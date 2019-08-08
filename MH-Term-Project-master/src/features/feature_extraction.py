import numpy as np
import pickle, pandas, textblob, string, sys
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble


def get_clean_data():
    """
    read the clean data, which was already preprossed and return some clean data

    return: a tuple of clean data (text, train_text, val_text, test_text, y_train_encode, y_val_encode, y_test_encode)
    """
    try:
        with open('../../data/word_dic.p', 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            X_train, X_val, X_test, train_text, val_text, test_text, y_train, y_val, y_test, wordtoix, ixtoword = u.load()
    except:
        print('cannot read the clean data')
        exit(1)

    text = train_text + val_text + test_text
    num_train, num_val, num_test = len(X_train), len(X_val), len(X_test)    
    y_train_encode, y_val_encode, y_test_encode = [], [], []

    for i in range(num_train):
        for k in range(4):
            if y_train[i][k] == 1:
                y_train_encode.append(k)
                break

    for i in range(num_val):
        for k in range(4):
            if y_val[i][k] == 1:
                y_val_encode.append(k)
                break

    for i in range(num_test):
        for k in range(4):
            if y_test[i][k] == 1:
                y_test_encode.append(k)
                break

    return text, train_text, val_text, test_text, y_train_encode, y_val_encode, y_test_encode


def generate_word_count(clean_data):
    """
    gerenerate the word count feature
    
    @param: text: type, a tuple, a tuple of clean data (text, train_text, val_text, test_text, y_train_encode, y_val_encode, y_test_encode)
    return: void
    """
    text, train_text, val_text, test_text, y_train_encode, y_val_encode, y_test_encode = clean_data

    count_vec = CountVectorizer(token_pattern=r'w{1,}')
    count_vec.fit(text)

    X_train_count = count_vec.transform(train_text)
    X_val_count = count_vec.transform(val_text)
    X_test_count = count_vec.transform(test_text)

    pickle.dump([X_train_count, X_val_count, X_test_count, y_train_encode, y_val_encode, y_test_encode], open("../../data/word_count.p", "wb"))


def generate_TFIDF(clean_data):
    """
    gerenerate the word level TF-IDF feature
    
    @param: text: type, a tuple, a tuple of clean data (text, train_text, val_text, test_text, y_train_encode, y_val_encode, y_test_encode)
    return: void
    """
    text, train_text, val_text, test_text, y_train_encode, y_val_encode, y_test_encode = clean_data

    # word level tf-idf
    tfidf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vec.fit(text)

    X_train_tfidf =  tfidf_vec.transform(train_text)
    X_val_tfidf =  tfidf_vec.transform(val_text)
    X_test_tfidf =  tfidf_vec.transform(test_text)

    pickle.dump([X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train_encode, y_val_encode, y_test_encode], open("../../data/tfidf.p", "wb"))


def generate_ngram(clean_data):
    """
    gerenerate the 2-gram and 3-gram
    
    @param: text: type, a tuple, a tuple of clean data (text, train_text, val_text, test_text, y_train_encode, y_val_encode, y_test_encode)
    return: void
    """
    text, train_text, val_text, test_text, y_train_encode, y_val_encode, y_test_encode = clean_data

    # ngram level tf-idf 
    tfidf_vec_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vec_ngram.fit(text)

    X_train_tfidf_ngram = tfidf_vec_ngram.transform(train_text)
    X_val_tfidf_ngram = tfidf_vec_ngram.transform(val_text)
    X_test_tfidf_ngram = tfidf_vec_ngram.transform(test_text)

    pickle.dump([X_train_tfidf_ngram, X_val_tfidf_ngram, X_test_tfidf_ngram, y_train_encode, y_val_encode, y_test_encode], open("../../data/tfidf_ngram.p", "wb"))


def main():
    clean_data = get_clean_data()
    generate_word_count(clean_data)
    generate_TFIDF(clean_data)
    generate_ngram(clean_data)


if __name__ == '__main__':
    main()
