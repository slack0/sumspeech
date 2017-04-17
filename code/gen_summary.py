
from src.corpus import SpeechCorpus
from src.utilities import *


raw_corpus_html = {
        'obama': '../data/speech_corpus/obama_raw_html',
        'romney': '../data/speech_corpus/romney_raw_html',
        'test': '../data/speech_corpus/simple_html'
        }

raw_corpus_text = {
        'obama' : '../data/Obama',
        'romney': '../data/Romney',
        'test': '../data/tests/simple'
        }

curated_urls = {
        'obama': '../data/obama.links',
        'romney': '../data/romney.links',
        'trump': '../data/trump.clean.links',
        'test': '../data/tests/test.links'
        }

__n_topics_to_extract = 10
__n_topics_to_associate_per_speech = 3

__n_topics_to_display = 5
__n_speeches_to_print = 5
__n_sentences_per_speech = 5


sc = SpeechCorpus(links=curated_urls['obama']),
                  num_corpus_topics=__n_topics_to_extract,
                  num_speech_topics=__n_topics_to_associate_per_speech)

sc.describe()

sc.top_topics(topic_count=__n_topics_to_display)

sc.top_sentences(speech_count=__n_speeches_to_print,
                 sentence_count=__n_sentences_per_speech)

