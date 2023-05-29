import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def clean_text(text):
    """
    Removes and replaces all whitespaces with single spaces. Transforms all letters to lowercase.
    Returns preprocessed text.
    """
    text = text.lower()
    return re.sub(r'\s+', ' ', text)


stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    """
    Removes stopwords from given text.
    :param text: a text string
    :return: list of words from given text without stopwords
    """
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


stemmer = SnowballStemmer("english")


def stemming(sentence):
    """
    Transforms words in a sentence into stems.
    :param sentence:
    :return: list of words from given sentence in stem form
    """
    stem_sentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stem_sentence += stem
        stem_sentence += " "
    stem_sentence = stem_sentence.strip()
    return stem_sentence
