''' Talking Points '''

from goose import Goose

import operator
import nltk
import numpy as np

import pprint

pp = pprint.PrettyPrinter(indent=4)

def grab_link(in_url):
    """
    Extract article information from Goose
    Kwargs:
        in_url (str): input url string for article extraction

    """
    try:
        pp.pprint('Getting article: ' + in_url)
        article = Goose().extract(url=in_url)
        return article
    except ValueError:
        print 'Goose failed to extract article from url'
        return None
    return None


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))

    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    # stems = stem_tokens(tokens, stemmer)
    return tokens


def get_corpus_topics(tfidf, model, n_topics):
    """
    Vocabulary ID to work mapping
    Kwargs:
        tfidf: TF-IDF model
        model: NMF/LDA model
        n_topics: number of topics

    """
    id2word = {}
    topics = []
    for k in tfidf.vocabulary_.keys():
        id2word[tfidf.vocabulary_[k]] = k

    for topic_index in xrange(n_topics):
        topic_importance = dict(zip(id2word.values(),
                                    list(model.components_[topic_index])))
        sorted_topic_imp = sorted(topic_importance.items(),
                                  key=operator.itemgetter(1),
                                  reverse=True)
        topics.append([i[0] for i in sorted_topic_imp])

    ''' list of all words sorted in descending order of
    importance for all topics '''
    return topics


def print_top_topics(topics, n_topics=10):
    """
    Print top topics words
    Kwargs:
        topics (list): list of topic words
        n_topics (int): number of topics words to print
    """
    topic_index = 0
    for i in topics[:n_topics]:
        topic_words = i[0:9]
        print 'Topic: {} -- {}\n'.format(topic_index,
                                         [str(j) for j in topic_words])
        topic_index += 1


def get_top_topics(W, n_topics):
    """
    Return top topics from CorpusW
    Kwargs:
        W (np.array): CorpusW decomposed from NMF
        n_topics (int): number of topics words to return
    """
    top_topics = []
    for row in W:
        top_topics.append(np.argsort(row)[::-1][:n_topics])

    return top_topics

