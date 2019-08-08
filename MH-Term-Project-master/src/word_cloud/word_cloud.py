import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def get_text():
    """
    get the text data from the preprocessed data file

    @param: None
    return: a tuple of (text_normal, text_depression, text_PTSD, text_bipolar)
    """
    with open('../../data/word_dic.p', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        X_train, X_val, X_test, train_text, val_text, test_text, y_train, y_val, y_test, wordtoix, ixtoword = u.load()

    text = train_text + val_text + test_text
    label = y_train + y_val + y_test  # 0: normal, 1: depression, 2: PTSD, 3: Bipolar

    text_normal, text_depression, text_PTSD, text_bipolar = '', '', '', ''

    for i in range(len(label)):
        if label[i][0] == 1:
            text_normal += ' ' + text[i]
        elif label[i][1] == 1:
            text_depression += ' ' + text[i]
        elif label[i][2] == 1:
            text_PTSD += ' ' + text[i]
        else:
            text_bipolar += ' ' + text[i]

    return text_normal, text_depression, text_PTSD, text_bipolar


def plot_word_cloud(text, key):
    """
    plot a word cloud graph

    @param: text: type, string, text could be text_normal, text_depression, text_PTSD, text_bipolar
    @param: key: type, string, key could be 'normal', 'depression', 'PTSD', 'bipolar'
    return: None
    """
    wc = WordCloud().generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.savefig('word_cloud_' + key + '.png')
    plt.clf()


def main():
    text_normal, text_depression, text_PTSD, text_bipolar = get_text()
    for text, key in [(text_normal, 'normal'), (text_depression, 'depression'), (text_PTSD, 'PTSD'), (text_bipolar, 'bipolar')]:
        plot_word_cloud(text, key)

if __name__ == '__main__':
    main()
