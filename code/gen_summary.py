# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumspeech.speechcorpus import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

if __name__ == '__main__':

    raw_corpus_html = {
            'obama': '../data/speech_corpus/obama_raw_html',
            'romney': '../data/speech_corpus/romney_raw_html',
            'test': '../data/speech_corpus/simple_html'
            }

    raw_corpus_text = {
            'obama': '../data/Obama',
            'romney': '../data/Romney',
            'test': '../data/tests/simple'
            }

    curated_urls = {
            'obama': '../data/obama.links',
            'romney': '../data/romney.links',
            'trump': '../data/trump.clean.links',
            'test': '../data/tests/test.links'
            }

    __n_topics_to_extract = 4
    __n_topics_to_associate_per_speech = 1

    __n_topics_to_display = 5
    __n_speeches_to_print = 5
    __n_sentences_per_speech = 10

    ''' build speech corpus '''
    sc = SpeechCorpus(url_path=curated_urls['test'])

    ''' verify corpus after creation '''
    sc.get_speeches()

    ''' initialize the corpus for next steps '''
    sc.initialize_corpus(n_corpus_topics=__n_topics_to_extract,
                         n_doc_topics=__n_topics_to_associate_per_speech)

    ''' vectorize the corpus -
    we can select the type of vectorizer to use with the option 'vectorizer'

    A selection of text vectorizers available in sklearn.feature_extraction.text

    The default vectorizer is TfidfVectorizer
    '''
    sc.vectorize_corpus()

    ''' Decompose the document-term matrix into document-topic and
    topic-term matrices with decomposition

    Models supported for decomposition are:
        - sklearn.decomposition.NMF (default)
        - sklearn.decomposition.LatentDirichletAllocation

    If model selected is NMF, there is option to select the solver used
    for Non-negative Matrix Factorization. Use the nmf_init argument to provide
    a string as argument to sklearn.decomposition.NMF .

    nmf_init defaults to 'random'

    '''
    sc.fit(model=NMF)

    ''' at this point, it is possible to pull different attributes of
    corpus and speeches - and use it debug/evaluate '''
    sc.corpus_tf_info()

    ''' extract summaries from corpus.
    this generates sentence rankings for every speech

    During this step, the vocabulary of entire corpus is used
    to normalize the vocabulary of the speech.

    There is an option to explore additional vectorizers when looking
    for similarity to topic vectors.

    The default vectorizer is TfidfVectorizer
    '''
    sc.extract_summaries()

    ''' iterate on corpus speeches and print summaries '''
    for i in sc.corpus:
        pp.pprint('')
        pp.pprint('Speech: {}'.format(i.get_title()))
        pp.pprint(i.get_summary(10))

