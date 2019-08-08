class Config:
    def __init__(self):
        # could be chosen from 'word_count', 'tfidf', or 'ngram'
        self.feature = 'tfidf'

        # model 'naive_bayes', 'random_forest', or 'SVM'
        self.model = 'random_forest'