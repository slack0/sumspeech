
from src.speechcorpus import *
from src.utils import *

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
__n_topics_to_associate_per_speech = 1

__n_topics_to_display = 5
__n_speeches_to_print = 5
__n_sentences_per_speech = 5


sc = SpeechCorpus(url_path=curated_urls['test'],
                  n_corpus_topics=__n_topics_to_extract,
                  n_doc_topics=__n_topics_to_associate_per_speech)

# sc.describe()

# sc.top_topics(topic_count=__n_topics_to_display)

# sc.top_sentences(speech_count=__n_speeches_to_print,
#                  sentence_count=__n_sentences_per_speech)

