
'''
Talking Points
'''

import string

import operator
import pprint

import numpy as np
from collections import defaultdict

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import NMF

from utils import *


stemmer = PorterStemmer()
pp = pprint.PrettyPrinter(indent=4)


def extract_corpus_topics(corpus_path, n_corpus_topics, n_doc_topics=1, n_summary_sentences=5):

    ''' Parse contents of speech directory and get dictionaries '''
    #proc_speech, raw_speech = parse_speeches(corpus_path)
    #proc_speech, raw_speech, speech_titles = create_corpus_from_html(corpus_path)

    ### URL-based extraction test
    proc_speech, raw_speech, speech_titles = create_corpus_from_web(corpus_path)



    ''' TFIDF vectorization and generate vocabularies '''
    corpus_tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs_corpus = corpus_tfidf.fit_transform(proc_speech.values())

    #print "Corpus TF vector shape: {}".format(tfs_corpus.shape)

    ''' Get the vocabulary from TF-IDF - for tokenizing in future steps '''
    corpus_vocab = corpus_tfidf.get_feature_names()

    ''' create a NMF model '''
    corpus_model = NMF(n_components=n_corpus_topics, init='random', random_state=0)
    corpusW = corpus_model.fit_transform(tfs_corpus)

    #print "Shape of W (decomposition output) = {}".format(Wcorpus.shape)
    ''' get *all* words for each topic '''
    topics = get_corpus_topics(corpus_tfidf, corpus_model, n_corpus_topics)

    '''
    For each document/speech -- extract the top sentences
    Create a dict of dicts to populate sentences for every speech in the corpus
    '''
    speech_sentences = defaultdict(dict)
    raw_sentences = defaultdict(dict)

    ''' Create a sentence TF-IDF instance using the corpus vocabulary '''
    sentence_tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', vocabulary=corpus_vocab)

    ''' top topics for each document '''
    best_topic_indices = get_top_topics(corpusW, n_doc_topics)

    '''
    (1) iterate over raw speech text and speech_sentences
    (2) get sentence term-frequency vectors based on the vocabulary of the corpus
    (3) check the cosine similarity of every sentences' TF vector with that of the top topics for that document
    '''
    for index,doc in enumerate(raw_speech.iterkeys()):
        print "*"*120
        pp.pprint('Processing: ' + str(doc))
        pp.pprint('Document Title: ' + str(speech_titles[doc]))

        doc_blob = TextBlob(raw_speech[doc])
        sentence_count = 0
        for sentence in doc_blob.sentences:
            ''' strip punctuation from the sentence now '''
            sentence_no_punct = str(sentence).translate(None, string.punctuation)
            speech_sentences[doc][sentence_count] = sentence_no_punct
            raw_sentences[doc][sentence_count] = sentence
            sentence_count += 1

        speech_tfs = sentence_tfidf.fit_transform(speech_sentences[doc].values()).todense()

        ''' iterate over the speech's most-relevant topics - and get cosine similarity '''
        top_topics_of_doc = best_topic_indices[index]

        for topic_index in top_topics_of_doc:

            pp.pprint('Top Topic: ' + str(topic_index))
            pp.pprint('Top Topic Words: ' + str(topics[topic_index][:10]))
            print ""

            topic_vector = corpus_model.components_[topic_index]
            sentence_similarity = {}
            for s_index, s_tf in enumerate(speech_tfs):
                ''' calcuating the cosine similarity with this sort of reshape op -- to get rid of a sklearn warning '''
                sentence_similarity[s_index] = cosine_similarity(s_tf,topic_vector.reshape((1,-1)))[0][0]

            ''' sort the sentence_similarity and pull the indices of top sentences '''
            top_n_sentences = [i[0] for i in sorted(sentence_similarity.items(), key=operator.itemgetter(1), reverse=True)[:n_summary_sentences]]
            bottom_n_sentences = [i[0] for i in sorted(sentence_similarity.items(), key=operator.itemgetter(1), reverse=False)[:n_summary_sentences]]
            print "Most Important Sentences..."
            for i in top_n_sentences:
                pp.pprint(str(raw_sentences[doc][i]))

            print ""
            print "Least Important Sentences..."
            for i in bottom_n_sentences:
                pp.pprint(str(raw_sentences[doc][i]))

    return corpus_vocab, corpusW, topics, corpus_model
