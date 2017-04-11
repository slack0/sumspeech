''' Talking Points '''

import re
import logging
import os

import Goose

import string
import hashlib

from sklearn.feature_extraction.text import TfidfVectorizer

from utils import *

# Globals
logging.basicConfig()
LOG = logging.getLogger("SpeechCorpus")
LOG.setLevel(logging.INFO)


class Speech(object):

    """Speech Operations for a Corpus"""

    def __init__(self, speech_id=None, title=None, content=None):
        """
        Kwargs:
            speech_id (str):
            title (str):

        """
        self.speech_id = speech_id
        self.title = title
        self.raw_content = content

        ''' speech attributes '''

        ''' processed_txt contains the output of cleanup,
        with punctuation for sentence extraction '''
        self.processed_txt = self._process_content()

        self.topics = []
        self.summary_sentences = []

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
            (str) The unprocessed string of all text gathered from parsing
            the speech
        """

        return self.raw_content

    def get_processed_content(self):
        """
        Returns:
            (str) The cleaned-up string of all text gathered from parsing
            the speech
        """

        return self.processed_txt

    def get_unpunctuated_content(self):
        """
        Returns:
            (str) processed text after clean and punctuation removal
            for vectorization and vocabulary generation
        """

        return self.processed_txt.translate(None, string.punctuation)

    def _process_content(self):
        """
        Returns:
            (str) cleaned up string of text from raw content of the speech

            performs substitution/lower casing and double space removal
            for sentence extraction

        """
        re.sub("[\W\d]", " ", self.content.lower().strip())
        lowers = self.content.replace('\n', ' ').replace('\r', ' ')

        while "  " in lowers:
            lowers = lowers.replace('  ', ' ')
        return lowers

    def get_raw_sentences(self, n_sentences=None):
        """
        Returns:
            (list) The list of first n_sentences from the raw text of
            the speech

            if n_sentences is None, then return the first five sentences
        """
        raise NotImplementedError

    def get_assigned_topics(self):
        return self.topics

    def get_summary_sentences(self, n_sentences=None):
        ''' return the top n_sentences from rawtxt '''
        raise NotImplementedError


class SpeechCorpus(object):
    def __init__(self,
                 html_path=None,
                 txt_path=None,
                 url_path=None,
                 n_corpus_topics=10,
                 n_doc_topics=1,
                 n_summary_sentences=5):

        self.html_path = html_path
        self.txt_path = txt_path
        self.url_path = url_path

        self._n_corpus_topics = n_corpus_topics
        self._n_doc_topics = n_doc_topics
        self._n_summary_sentences = n_summary_sentences

        ''' speech article related attributes '''
        self.titles = []
        self.text_content = []

        ''' corpus attributes '''
        self.corpus = []
        self.corpus_tf_vectors = None
        self.corpus_vocab = None
        self.corpus_model = None
        self.corpusW = None

        ''' create the corpus during initialization '''
        self._create_corpus()

    def _create_corpus(self):
        if all([self.html_path, self.txt_path, self.url_path]) is False:
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
        _corpus_vectorizer = vectorizer(tokenizer=tokenize,
                                        stop_words='english')
        _processed_sp = [_each_speech.get_unprocessed_content()
                         for _each_speech in corpus]
        self.corpus_tf_vectors = _corpus_vectorizer.fit_transform(_processed_sp)
        self.corpus_vocab = _corpus_vectorizer.get_feature_names()

    def fit(self, model=NMF):
        """
        Fit a model on vectorized speech corpus
        Kwargs:
            model (function): name of the model to perform decomposition
        """
        self.corpus_model = model(n_components=self._n_corpus_topics,
                                  init='random',
                                  random_state=0)
        self.corpusW = self.corpus_model.fit_transform(self.corpus_tf_vectors)

    def corpus_tf_info(self):
        print "\nCorpus TF vector info: {}".format(self.corpus_tf_vectors.shape)
        print "\nCorpusW (doc-to-topics): {}".format(self.corpusW.shape)

    def get_corpus_vocabulary(self):
        pass

    def get_speech_to_topic(self):
        ''' return corpusW '''
        pass

    def get_corpus_topics(self):
        pass

    def get_top_topics(self):
        pass

    def summaries(self):
        return _gen_summaries()

    def _gen_summaries(self):
        pass

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

                    self.titles.append(
                        unidecode.unidecode_expect_nonascii(_article.title))
                    self.text_content.append(
                        unidecode.unidecode_expect_nonascii(
                            _article.cleaned_text))
                    _htmlfile.close()

                if (doc_type is 'txt'):
                    _fhandle = open(_file_path, 'r')

                    ''' currently we do not have a means to extract title
                    explicitly from processed text
                    '''
                    self.titles.append(None)
                    self.text_content.append(
                        unidecode.unidecode_expect_nonascii(_fhandle.read()))
                    _fhandle.close()

    def _web2corpus(self):

        U = open(self.url_path)
        _urls = [url.strip() for url in U.readlines()]
        for _each_url in _urls:
            _article = grab_link(_url)
            if not (_article and _article.cleaned_text and _article.title):
                continue

            self.titles.append(
                unidecode.unidecode_expect_nonascii(_article.title))
            self.text_content.append(
                unidecode.unidecode_expect_nonascii(_article.cleaned_text))

        U.close()
