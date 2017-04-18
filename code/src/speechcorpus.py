''' Talking Points '''

import os
import re
import string
import operator
import pprint
import logging

import numpy as np

from collections import defaultdict

from textblob import TextBlob
from goose import Goose
import unidecode

import hashlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.metrics.pairwise import cosine_similarity

from utils import *

# Globals
logging.basicConfig()
LOG = logging.getLogger("SpeechCorpus")
LOG.setLevel(logging.INFO)

pp = pprint.PrettyPrinter(indent=4)

class Speech(object):

    """ Operations for Speech Text """

    def __init__(self, speech_id=None, title=None, content=None):
        """
        Kwargs:
            speech_id (str):    unique ID associated with the speech
            title (str):        title of the speech
            content (str):      unicode string containing the entire text
                                of the speech (unprocessed)
        """

        self.speech_id = speech_id
        self.title = title
        self.raw_content = content

        ''' speech attributes '''

        ''' processed_txt contains the output of cleanup,
        with punctuation for sentence extraction '''
        self.processed_txt = self._process_content()

        ''' raw sentences of the speech '''
        self.raw_sentences = self._extract_raw_sentences()

        ''' topic attributes '''
        self.topics = []
        self.topic_words = {}

        ''' sentence similarity metric with topics '''
        self.sentence_similarity = {}

        '''
        (1) sentence ranking based on similarity, in descending order
        (2) lower indices correspond to index of most-important sentences
            and vice-versa
        (3) use these rankings used to extract most/least important sentences
        '''
        self.sentence_ranking = []

    def __repr__(self):
        """
        Returns:
            (str) The unique speech_id and speech title

        """

        return '{}: {}'.format(self.speech_id, self.title)

    def get_id(self):
        """
        Returns:
            (str) The unique speech_id

        """

        return self.speech_id

    def get_title(self):
        """
        Returns:
            (str) The speech title

        """

        return self.title

    def get_unprocessed_content(self):
        """
        Returns:
            (str) The raw unicode string (all text) parsed from the speech

        """

        return self.raw_content

    def get_processed_content(self):
        """
        Returns:
            (str) The cleaned-up string of parsed speech, with punctuation

        """

        return self.processed_txt

    def get_unpunctuated_content(self):
        """
        Returns:
            (str) The cleaned-up string from processed text after
            punctuation removal for vectorization and vocabulary generation

        """

        return self.processed_txt.translate(None, string.punctuation)

    def get_raw_sentences(self, n_sentences=5):
        """
        Returns:
            (list) The list of first n_sentences from the raw text of
            the speech

        """

        return self.raw_sentences[:n_sentences]

    def get_assigned_topics(self):
        """
        Returns:
            (list) The list of topics assigned to this speech after
            topic modeling from the corpus

        """

        return self.topics

    def _process_content(self):
        """
        Returns:
            (str) The cleaned up string of text from raw content of the speech

            This routine performs substitution/lower casing and
            double space removal for sentence extraction

        """

        re.sub("[\W\d]", " ", self.raw_content.lower().strip())
        lowers = self.raw_content.replace('\n', ' ').replace('\r', ' ')

        while "  " in lowers:
            lowers = lowers.replace('  ', ' ')
        return lowers

    def _extract_raw_sentences(self):
        """
        Returns:
            (list): The list of all strings corresponding to sentences
            of the speech

        """

        doc_blob = TextBlob(self.processed_txt)
        return doc_blob.sentences

    def get_summary(self, n_summary_sentences=5, most_important=True):
        """
        Returns:
            (list) The list of summary sentences of the speech
            extracted from the sentence ranking extracted from

        """

        indices = []
        summ_sentences = []
        if not most_important:
            ''' get the least-important sentences in reverse! '''
            indices = [i for i in self.sentence_ranking[-n_summary_sentences::][::-1]]
            summ_sentences = [self.raw_sentences[i] for i in indices]
        else:
            indices = [i for i in self.sentence_ranking[:n_summary_sentences]]
            summ_sentences = [self.raw_sentences[i] for i in indices]

        return dict(zip(indices, summ_sentences))

class SpeechCorpus(object):

    """ Speech Corpus Generation and Operations """

    def __init__(self,
                 html_path=None,
                 txt_path=None,
                 url_path=None,
                 n_corpus_topics=10,
                 n_doc_topics=1):

        self.html_path = html_path
        self.txt_path = txt_path
        self.url_path = url_path

        self._n_corpus_topics = n_corpus_topics
        self._n_doc_topics = n_doc_topics

        ''' speech article related attributes '''
        self.titles = []
        self.text_content = []

        ''' corpus attributes '''
        self.corpus = []
        self._corpus_vectorizer = None
        self.corpus_tf_vec = None
        self.corpus_vocab = None
        self.corpus_model = None
        self.corpusW = None
        self.topics = []
        self._id2word = {}

        self.top_topics_of_corpus = []

        ''' create the corpus during initialization '''
        self._create_corpus()

    def __repr__(self):
        """
        Returns:
            (None) Prints the collection of speeches contained in the corpus

        """
        corpus_repr = '\n'.join(str(i) for i in self.corpus)
        return '{}'.format(corpus_repr)

    def _create_corpus(self):
        """
        Create speech corpus from inputs
        """

        if any([self.html_path, self.txt_path, self.url_path]) is False:
            raise NotImplementedError

        if self.html_path is not None:
            self._doc2corpus(doc_type='html')

        elif self.txt_path is not None:
            self._doc2corpus(doc_type='txt')

        elif self.url_path is not None:
            self._web2corpus()

        else:
            raise NotImplementedError

        '''
        create a new speech instance for every valid speech
        -- extract article title
        -- create a unique ID from the raw text content of the speech
        -- create a speech object with the info and append corpus
        '''
        for sp_index in xrange(len(self.titles)):
            ''' create _speech_id by md5 hashing the article content '''
            _speech_id = hashlib.md5(self.text_content[sp_index]).hexdigest()

            sp = Speech(_speech_id,
                        self.titles[sp_index],
                        self.text_content[sp_index])
            self.corpus.append(sp)

    def vectorize_corpus(self, vectorizer=TfidfVectorizer):
        """
        Vectorize the corpus for summary extraction
        Kwargs:
            vectorizer (function): name of the vectorizer

        """

        self._corpus_vectorizer = vectorizer(tokenizer=tokenize,
                                             stop_words='english')
        _cleaned_sp = [_sp.get_unpunctuated_content() for _sp in self.corpus]
        self.corpus_tf_vec = self._corpus_vectorizer.fit_transform(_cleaned_sp)
        self.corpus_vocab = self._corpus_vectorizer.get_feature_names()

    def fit(self, model=NMF, nmf_init='random'):
        """
        Fit a model on vectorized speech corpus
        Kwargs:
            model (function): name of the model to perform decomposition

        """
        if not ((model.__module__ == 'sklearn.decomposition.nmf') or
                (model.__module__ == 'sklearn.decomposition.online_lda')):
            raise NotImplementedError

        if model.__module__ == 'sklearn.decomposition.nmf':
            self.corpus_model = model(n_components=self._n_corpus_topics,
                                      init=nmf_init,
                                      random_state=1980)

        if model.__module__ == 'sklearn.decomposition.online_lda':
            self.corpus_model = model(n_topics=self._n_corpus_topics,
                                      n_jobs=-1,
                                      verbose=10)

        '''
        the ordering of speeches is incorporated into
        corpus_tf_vec and corpusW

        this ordering is utilized during generating summaries
        '''
        self.corpusW = self.corpus_model.fit_transform(self.corpus_tf_vec)
        self._generate_corpus_topics()

    def _generate_corpus_topics(self):
        """
        Vocabulary ID to word mapping

        Reads in tf vectors and from the document to vocabulary mapping,
        extracts topic words for n_corpus_topics
        """

        for k in self._corpus_vectorizer.vocabulary_.keys():
            self._id2word[self._corpus_vectorizer.vocabulary_[k]] = k

        for topic_index in xrange(self._n_corpus_topics):
            topic_importance = dict(zip(self._id2word.values(),
                                        list(self.corpus_model.components_[topic_index])))
            sorted_topic_imp = sorted(topic_importance.items(),
                                      key=operator.itemgetter(1),
                                      reverse=True)
            self.topics.append([i[0] for i in sorted_topic_imp])

    def extract_summaries(self, vectorizer=TfidfVectorizer):
        """
        Returns:
            (None)
            Performs summary extraction from the corpus based on decomposition/
            latent feature extraction from the vectorized corpus
        """

        sentence_tfidf = vectorizer(tokenizer=tokenize,
                                    stop_words='english',
                                    vocabulary=self.corpus_vocab)

        self.top_topics_of_sp = self.get_top_topics()

        '''
        (1) iterate over raw speech text and speech_sentences
        (2) get sentence term-frequency vectors based on the
            vocabulary of the corpus
        (3) check the cosine similarity of every sentences' TF vector
            with that of the top topics for that document
        '''
        for _index, _sp in enumerate(self.corpus):

            speech_sentences = defaultdict(int)
            for _sentence_count, _each_sentence in enumerate(_sp.raw_sentences):
                speech_sentences[_sentence_count] = str(_each_sentence).translate(None, string.punctuation)

            speech_tfs = sentence_tfidf.fit_transform(speech_sentences.values()).todense()

            '''
            (1) iterate over each speech's most relevant topics
            (2) get cosine similarity between speech_tfs and topic_vectors
            '''

            ''' assign topics to speech; at this point,
            we should be able to query/retrieve topics from speech objects '''
            _sp.topics = self.top_topics_of_sp[_index]

            for _topic_index in _sp.topics:
                pp.pprint("Top Topic: " + str(_topic_index))
                pp.pprint("Top Topic Words: "  + str(self.topics[_topic_index][:10]))
                pp.pprint(" ")

                topic_vector = self.corpus_model.components_[_topic_index]
                for s_index, s_tf in enumerate(speech_tfs):
                    ''' calculating the cosine simiarlity '''
                    _sp.sentence_similarity[s_index] = cosine_similarity(s_tf, topic_vector.reshape((1, -1)))[0][0]

                ''' sort sentence similarity and rank them in descending order '''
                _sp.sentence_ranking = [i[0] for i in sorted(_sp.sentence_similarity.items(),
                                                             key=operator.itemgetter(1),
                                                             reverse=True)]

    def _doc2corpus(self, doc_type):

        for _subdir, _dirs, _files in os.walk(self.html_path):
            for _each_file in _files:
                _file_path = _subdir + os.path.sep + _each_file
                if (doc_type is 'html'):
                    _htmlfile = open(_file_path, 'r')
                    _article = Goose().extract(raw_html=_htmlfile.read())
                    if not (_article and
                            _article.cleaned_text and
                            _article.title):
                        continue

                    self.titles.append(unidecode.unidecode_expect_nonascii(_article.title))
                    self.text_content.append(unidecode.unidecode_expect_nonascii(_article.cleaned_text))
                    _htmlfile.close()

                if (doc_type is 'txt'):
                    _fhandle = open(_file_path, 'r')

                    """ currently we do not have a means to extract title
                    explicitly from processed text
                    """
                    self.titles.append(None)
                    self.text_content.append(unidecode.unidecode_expect_nonascii(_fhandle.read()))
                    _fhandle.close()

    def _web2corpus(self):

        U = open(self.url_path)
        _urls = [url.strip() for url in U.readlines()]
        for _each_url in _urls:
            _article = grab_link(_each_url)
            if not (_article and _article.cleaned_text and _article.title):
                continue

            self.titles.append(unidecode.unidecode_expect_nonascii(_article.title))
            self.text_content.append(unidecode.unidecode_expect_nonascii(_article.cleaned_text))

        U.close()

    def corpus_tf_info(self):
        pp.pprint('Corpus TF vector info: {}'.format(self.corpus_tf_vec.shape))
        pp.pprint('CorpusW (doc-to-topics): {}'.format(self.corpusW.shape))
        pp.pprint('CorpusH (topics-to-vocab): {}'.format(self.corpus_model.components_.shape))

    def get_corpus_vocabulary(self):
        """
        Returns:
            (list) The words that form the corpus vocabulary

        """
        return self.corpus_vocab

    def get_speech_to_topic(self):
        """
        Returns:
            (np.array) The numpy array from the output of model decomposition

        """
        return self.corpusW

    def get_id2word(self):
        """
        Returns:
            (dict) The dictionary containing the mapping between topic index
            and words in the vocabulary

        """

        return self._id2word

    def get_corpus_topics(self):
        """
        Returns:
            (list) The list of corpus topics.
            Each topic itself is a list of words that describe the latent topic

        """

        return self.topics

    def get_top_topics(self):
        """
        Return top topics from the corpus
        Kwargs:
            W (np.array): CorpusW decomposed from NMF
            n_topics (int): number of topic words to return
        """

        top_topics = []
        for row in self.corpusW:
            top_topics.append(np.argsort(row)[::-1][:self._n_doc_topics])

        return top_topics

