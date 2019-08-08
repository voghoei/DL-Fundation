"""
---------------------------
This script preprocessed the manully labeled text dataset collected from Reddit
    1. remove punctuations
    2. remove links
    3. stemming
    4. remove stop words
Refference: https://www.kaggle.com/itratrahman/nlp-tutorial-using-python
---------------------------
"""

import enchant
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

data = pd.read_csv('data.csv')

def removep(text):
    '''
    a function for removing the punctuation
    '''
    d = enchant.Dict("en_US")
    text = [word.encode('utf-8').lower().translate(None, '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~') for word in text.split() if d.check(word)]
    # joining the list of words with space separator
    return " ".join(text)
data['text'] = data['combined'].apply(removep)

def removelink(text):
    '''
    a function for removing the links
    '''
    text = [re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE) for word in text.split()]
    # joining the list of words with space separator
    return " ".join(text)
data['text'] = data['combined'].apply(removelink)

def stemming(text):
    '''
    a function for stemming
    '''
    text = [WordNetLemmatizer().lemmatize(word,'v') for word in text.split()]
    # joining the list of words with space separator
    return " ".join(text)
data['text'] = data['combined'].apply(stemming)

# nltk.download('stopwords')
sw = stopwords.words('english')
def stopwords(text):
    '''
    a function for removing the stopword
    '''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)
data['text'] = data['combined'].apply(stopwords)

# save to csv
data.tocsv('cleaned.csv')
