import numpy as np
import pickle, pandas, textblob, string, sys, config
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble


def get_data(feature):
    """
    @param: feature: type, string, could be 'word_count', 'tfidf', or 'ngram'
    return: a tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if feature == 'word_count':
        file_path = '../../data/word_count.p'
    elif feature == 'tfidf':
        file_path = '../../data/tfidf.p'
    elif feature == 'tfidf_ngram':
        file_path = '../../data/tfidf_ngram.p'
    else:
        print('error in featur config')
        exit(1)

    try:
        with open(file_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            return u.load()
    except:
        print('cannot read the clean data')
        exit(1)


def train_model(classifier, X_train, y_train, X_test, y_test):
    """
    train a basic NLP model

    @param: classifier: type, an classifier object
    @param: X_train: type, list, the training dataset
    @param: y_train: type, list, the label of the training dataset
    @param: X_test: type, list, the test dataset
    @param: y_test: type, list, the label of the test dataset
    return: a tuple of (accuracy, confusion matrix)
    """

    # fit the training dataset on the classifier
    classifier.fit(X_train, y_train)
    
    # predict the labels on test dataset
    predictions = classifier.predict(X_test)
    
    return metrics.accuracy_score(predictions, y_test), metrics.confusion_matrix(predictions, y_test)


def train(parameters):
    """
    model training and print out the accuracy

    @param: parameters: type, an object that contains parsed arguments
    return: void
    """
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(parameters.feature)
    accuracy = -1
    if parameters.model == 'naive_bayes':
        accuracy, confusion_matrix = train_model(naive_bayes.MultinomialNB(), X_train, y_train, X_test, y_test)
    elif parameters.model == 'random_forest':
        accuracy = 0
        confusion_matrix = None
        for _ in range(10):
            accu, matrix = train_model(ensemble.RandomForestClassifier(n_estimators=50), X_train, y_train, X_test, y_test)
            if accu > accuracy:
                accuracy = accu
                confusion_matrix = matrix
    elif parameters.model == 'SVM':
        accuracy = train_model(svm.SVC(gamma='auto'), X_train, y_train, X_test, y_test)

    if accuracy > 0:
        print("%s, %s: %f" % (parameters.model, parameters.feature, accuracy))
        print(confusion_matrix)


def main():
    parameters = config.Config()
    train(parameters)


if __name__ == '__main__':
    main()
