
import os
import re
import string
import operator
import pprint
import tabulate

import numpy as np

from goose import Goose
import unidecode

import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import NMF

def grab_link(in_url):
    ''' extract article information using Python Goose '''
    try:
        article = Goose().extract(url=in_url)
        return article
    except ValueError:
        print 'Goose failed to extract article from url'
        return None
    return None

def create_corpus_from_web(url_file, raw=False):
    raw_sp2txt = {}
    proc_sp2txt = {}
    speech_titles = {}
    U = open(url_file)
    url_list = [url.strip() for url in U.readlines()]
    for doc_index, url in enumerate(url_list):
        pprint.pprint('Grabbing URL: ' + str(url))

        article = grab_link(url)
        if not (article and article.cleaned_text and article.title):
            pprint.pprint('Skipping. No content in URL: ' + url)
            continue

        title = unidecode.unidecode_expect_nonascii(article.title)

        speech_titles[doc_index] = title

        _raw_input = article.cleaned_text
        text = unidecode.unidecode_expect_nonascii(_raw_input)
        re.sub("[\W\d]", " ", text.lower().strip())
        lowers = text.replace('\n',' ').replace('\r',' ')
        while "  " in lowers:
            lowers = lowers.replace('  ',' ')


        ''' store raw text -- for sentence extraction '''
        raw_sp2txt[doc_index] = lowers

        ''' store no_punctuation for NMF '''
        no_punctuation = lowers.translate(None, string.punctuation)
        proc_sp2txt[doc_index] = no_punctuation

    U.close()
    return proc_sp2txt, raw_sp2txt, speech_titles


def create_corpus_from_html(raw_html_path, raw=False):
    raw_sp2txt = {}
    proc_sp2txt = {}
    speech_titles = {}
    for subdir, dirs, files in os.walk(raw_html_path):
        for doc_index, each_file in enumerate(files):
            file_path = subdir + os.path.sep + each_file
            htmlfile = open(file_path, 'r')
            raw_content = htmlfile.read()
            article = Goose().extract(raw_html=raw_content)
            if not (article and article.cleaned_text and article.title):
                continue

            #print 'Processing article: ', article.title
            speech_titles[doc_index] = unidecode.unidecode_expect_nonascii(article.title)
            text = unidecode.unidecode_expect_nonascii(article.cleaned_text)

            re.sub("[\W\d]", " ", text.lower().strip())
            lowers = text.replace('\n',' ').replace('\r',' ')
            while "  " in lowers:
                lowers = lowers.replace('  ',' ')

            ''' store raw text -- for sentence extraction '''
            raw_sp2txt[doc_index] = lowers

            ''' store no_punctuation for NMF '''
            no_punctuation = lowers.translate(None, string.punctuation)
            proc_sp2txt[doc_index] = no_punctuation

            htmlfile.close()

    return proc_sp2txt, raw_sp2txt, speech_titles

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))

    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    #stems = stem_tokens(tokens, stemmer)
    return tokens


def parse_speeches(corpus_path):
    raw_sp2txt = {}
    proc_sp2txt = {}
    for subdir, dirs, files in os.walk(corpus_path):
        for each_file in files:
            #pprint.pprint("-- processing: {}".format(each_file))
            file_path = subdir + os.path.sep + each_file
            fhandle = open(file_path, 'r')
            _raw_input = fhandle.read()
            text = unidecode.unidecode_expect_nonascii(_raw_input)
            re.sub("[\W\d]", " ", text.lower().strip())
            lowers = text.replace('\n',' ').replace('\r',' ')
            while "  " in lowers:
                lowers = lowers.replace('  ',' ')

            ''' store raw text -- for sentence extraction '''
            raw_sp2txt[each_file] = lowers

            ''' store no_punctuation for NMF '''
            no_punctuation = lowers.translate(None, string.punctuation)
            proc_sp2txt[each_file] = no_punctuation

    return proc_sp2txt, raw_sp2txt


def get_corpus_topics(tfidf, model, n_topics):
    ''' vocabulary ID to word mapping '''
    id2word = {}
    topics = []
    for k in tfidf.vocabulary_.keys():
        id2word[tfidf.vocabulary_[k]] = k

    for topic_index in xrange(n_topics):
        #pprint.pprint("\n-- Top words in topic:")
        topic_importance = dict(zip(id2word.values(),list(model.components_[topic_index])))
        sorted_topic_imp = sorted(topic_importance.items(), key=operator.itemgetter(1),reverse=True)
        #pprint.pprint([i[0] for i in sorted_topic_imp[:15]])
        topics.append([i[0] for i in sorted_topic_imp])

    ''' list of all words sorted in descending order of importance for all topics '''
    return topics

def print_top_topics(topics, n_topics=10):

    print "\n\n"
    topic_index = 0
    for i in topics[:n_topics]:
        topic_words = i[0:9]
        print 'Topic: {} -- {}\n'.format(topic_index,[str(j) for j in topic_words])
        topic_index += 1


def get_top_topics(W, n_topics):
    top_topics = []
    for row in W:
        top_topics.append(np.argsort(row)[::-1][:n_topics])

    return top_topics


