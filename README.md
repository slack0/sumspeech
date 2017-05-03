
# sumspeech

Summarization tool for Political Speeches and Interviews.

## Introduction

Speech summarization is a necessity in many domains. It is needed in cases
where the audience needs to get a gist of a speech or an article in a way that
preserves the intent and the meaning of the speech. This belongs to a class of
problems under 
[Automatic summarization](https://en.wikipedia.org/wiki/Automatic_summarization).


Summary generation is difficult. It is context dependent.  The objective and
the metrics of summarization process themselves are difficult to define in some
situations. Generating or Extracting information that is representative of an
entire article, discussion, topic or speech is nuanced, as the author or
speaker may convey meaning implicitly by referring to other parts of the speech
or article.  This problem is worsened if there are reference to a past or
future discussion.  The challenges posed by text summarization has attracted a
lot of interest in academic and industry research communities. 


[Abstractive summarization](https://www.google.com/search?q=abstractive+summarization) 
is still an active area of work, and so is 
[Extractive summarization](https://www.google.com/search?q=extractive+summarization).

State of the art abstractive summarization techniques focus on attention
mechanisms using RNNs to train sentence generators. Extractive summarizers, on
the other hand, find the most informative sentences in a given article.
[TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
and [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) 
are a couple of well-known techniques for extractive summarization.

A good implementation of well-known/SoTA techniques for extractive
summarization are available in the tool [sumy](https://github.com/miso-belica/sumy).

## Summary Generation from Political Speeches

The effectiveness of extractive summary generation depends to a great degree on
the type of document.  The primary motivation for this work is to develop
insights and build a profile of a speaker based on their speech. Speeches,
interviews, and conversations in general reveal what people believe in and
where they stand on issues. Speech patterns such as vocabulary usage,
sentiments expressed and topics discussed are some of the main attributes that
pop out directly from speeches. Using these primary attributes, we explore
summary generation.

As good as some of the extractive summarizers are, they fall short of capturing
the context information available in articles or speeches from a speaker. This
work aims to utilize this contextual information (available in a corpora of
articles/speeches) to extract summaries.

### Methodology

In this work, we consider the problem of summary extraction from
documents/speeches based on the knowledge of what topics the speech is
about.

The main intuition here is that the inference about the topic is
valuable in evaluating which parts of a document or a speech are
relevant to it.  With this intuition, this tool provides a summary of
the most important topics of speeches based on speaking style (word
usage) of a specific person (i.e., from a speech corpus). Based on topic
extraction and sentence/document vectorization, the tool extracts most
relevant / important sentences from a speech by ranking topic similarity
to the sentence similarity.

Results show that this technique outperforms similar summarization
techniques that rely only on sentence similarity.

### Topic Extraction

The first step in summary generation from speeches is to identify the
topic(s) associated with any given speech. Topic extraction is effective
if it is done on a corpus of speeches as opposed to analyzing each
individual speech. To extract the topics of a corpus of speeches (a.k.a
corpus), we perform vectorization of the speeches using TF-IDF
(term-frequency inverse document frequency). The TF-IDF vectorization
provides vectorized word representations with vocabulary set to the
entire corpus. Using the vectorized representation of the corpus, we
then perform non-negative matrix factorization (NMF) to bring out the
latent topics of the corpus. NMF provides the mapping between 
speeches-to-topics and topics-to-vocabulary.

The speeches-to-topic mapping reveals interesting details about the
distribution of topics related to each speech within the corpus. The
figure below shows the distribution of topics related to six speeches
from Obama. The specificity of a speech is clearly evident from this
visual. Obviously, some speeches are concerned with specific topics,
while others discuss a combination of topics. It also reveals that the
vocabulary used by the speaker (Obama in this case) was specific enough
to be captured into distinct topics.

![alt
text](https://raw.githubusercontent.com/slack0/talking-points/master/data/topic_distribution.png
"Distribution of topics for speeches")

### Topic Mapping to Speeches

Every speech/document is a combination of topics. For instance, a press
conference given by Obama may cover ongoing wars, the military, economy,
health care, congress, income inequality and education. In contrast, a
speech to a business forum may just be about the state of economy. The
topic distributions in each of these two cases will be different. 

Speeches can be considered as distributions over topics. And topics as
distributions over vocabulary. The mapping between (speeches, topics)
and (topics, vocabulary) is obtained from the matrix factorization.

To get the most relevant summary of a speech, it is necessary to know
which topics the speech is about. We use this intuition to map /
associate a top topics to a speech. We use the topic vector for every
speech and sort the vector in descending order to pick the top topic for
a speech. Every topic is associated with a vocabulary vector. The top
words associated with a topic can again be obtained by sorting the
topic-vocabulary vector in descending order.

Consider the transcript of a speech given by Obama at a veterans convention.

```
Speech to the 113th Convention of the Veterans of Foreign Wars

Commander DeNoyer, thank you for your introduction, and your service in
Vietnam and on behalf of America's veterans. I want to thank your
executive director, Bob Wallace; your next commander, who I look forward
to working with, John Hamilton.  And to Gwen Rankin, Leanne Lemley, and
the entire Ladies Auxiliary, thank you for your patriotic service to
America.

I stand before you as our hearts still ache over the tragedy in Aurora,
Colorado.  Yesterday I was in Aurora  , with families whose loss is hard
to imagine -- with the wounded, who are fighting to recover; with a
community and a military base in the midst of their grief. And they told
me of the loved ones they lost. And here today, it's fitting to recall
those who wore our nation's uniform:

These young patriots were willing to serve in faraway lands, yet they
were taken from us here at home.  And yesterday I conveyed to their
families a message on behalf of all Americans: We honor your loved ones.
We salute their service. And as you summon the strength to carry on and
keep bright their legacy, we stand with you as one united American
family.

Veterans of Foreign Wars, in you I see the same shining values, the
virtues that make America great. When our harbor was bombed and fascism
was on the march, when the fighting raged in Korea and Vietnam, when our
country was attacked on that clear September morning, when our forces
were sent to Iraq -- you answered your country's call. Because you know
what Americans must always remember -- our nation only endures because
there are patriots who protect it.

...


More at:
http://www.americanrhetoric.com/speeches/barackobama/barackobama113vfw.htm

```

The topic clearly is associated with a discussion about wars, veterans
and their contributions to America. The top topic associated with this
speech, as represented by the topic vector/words is:

'iraq', 'veterans', 'war', 'troops', 'military', 'afghanistan', 'security', 'afghan'


Using the above topic vector, the following sentences are extracted from the speech.


```

After 10 years of war, and given the progress we've made, I felt it was
important that the American people -- and our men and women in uniform -- know
our plan to end this war responsibly.

Because we're leading around the world, people have a new attitude toward
America.

If the choice is between tax cuts that the wealthiest Americans don't need and
funding our troops that they definitely need to keep our country strong, I will
stand with our troops every single time.

For the first time ever, we've made military families and veterans a top
priority not just at DOD, not just at the VA, but across the government.

You know how this can work better, so let's get it done, together.

Four years ago, I said that I'd do everything I could to help our veterans
realize the American Dream, to enlist you in building a stronger America.

So today, our economy is growing and creating jobs, but it's still too hard for
too many folks to find work, especially our younger veterans, our veterans from
Iraq and Afghanistan.

With new tools like our online Veterans Jobs Bank, we're connecting veterans
directly to jobs.

It's one of the reasons I've proposed to Congress a Veterans Jobs Corps to put
our veterans back to work protecting and rebuilding America.

And today, I am again calling on Congress: Pass this Veterans Jobs Corps and
extend the tax credits for businesses that hire veterans so we can give these
American heroes the jobs and opportunities that they deserve.


```

A qualitative analysis of the above summary sentences shows it clearly captures
the salient points of the speech.


### Sentence Extraction for Summarization

To extract the most relevant sentences that summarize a given speech, we
first vectorize the sentences in the speech/document. Note that this
step typically uses the vocabulary of the document (and not of the
corpus) to generate the sentence vector. However, in order to compare
the similarity of a topic vector distributed over the vocabulary of an
entire corpus, we have to normalize the document vocabulary to that of
the entire corpus. To achieve this normalization, we take the vocabulary
of the corpus from the TF-IDF transform (performed for topic extraction)
and use that to obtain TF-IDF of every sentence in a document/speech.

To find the most relevant sentence, we compute the cosine similarity
between TF-IDF vector of the sentence and that of the topic vector
associated with the sentence/document. We then rank order the sentences
according to their cosine similarity -- and return the top sentences.


## Installation

sumspeech is built using [scikit-learn](http://scikit-learn.org/stable/),
[Goose](https://github.com/grangier/python-goose) and
[TextBlob](https://textblob.readthedocs.io/en/dev/). Install the dependencies as follows.

```
pip install -r requirements.txt
```

## Usage with Python API

sumspeech can be used as a library in your project or as a stand-alone API.

Once the corpus is intialized with all the documents, the API allows interactive exploration of different decomposition models - [Non-Negative Matrix Factorization](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF) and [Latent Dirichlet Allocation](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html).

Users can also select different vectorizers to explore different document to top mapping outcomes, and its affect on sentence ranking for summary extraction. In the current implementation, the allowed vectorizers are: [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) and [HashingVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html).

To explore different models and vectorizers, just re-run SpeechCorpus.vectorize\_corpus(), SpeechCorpus.fit() with desired models/options followed by SpeechCorpus.extract\_summary(). This sequence of operations would re-generate topic-to-sentence affinity and sentence rankings for summary extraction.

```

# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumspeech.speechcorpus import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

if __name__ == '__main__':

    ### dictionaries with paths to different formats of corpora 

    ### path to raw html files of documents 
    raw_corpus_html = {
            'obama': '../data/speech_corpus/obama_raw_html',
            'romney': '../data/speech_corpus/romney_raw_html',
            'test': '../data/speech_corpus/simple_html'
            }

    ### path to raw text files 
    raw_corpus_text = {
            'obama': '../data/Obama',
            'romney': '../data/Romney',
            'test': '../data/tests/simple'
            }

    ### path to files with URLs 
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

    ''' 
      Steps for sentence extraction:
        - Build speech corpus 
        - Vectorize the corpus
        - Fit a model
        - Extract and print summaries
    '''

    sc = SpeechCorpus(url_path=curated_urls['test'])
    sc.get_speeches()

    sc.initialize_corpus(n_corpus_topics=__n_topics_to_extract,
                         n_doc_topics=__n_topics_to_associate_per_speech)

    sc.vectorize_corpus()

    ### default model = NMF 
    ### default nmf_init = 'random'
    sc.fit(model=NMF, nmf_init='nndsvd')

    ### debug/get information about corpus and topics 
    sc.corpus_tf_info()               ### print corpus TF-IDF vector info
    sc.get_corpus_vocabulary()        ### print the entire corpus vocabulary
    sc.get_top_topics()               ### print top topics associated withevery speech/document in the corpus

    ### extract summaries (generate sentence rankings) 
    sc.extract_summaries()

    ### Iterate on corpus speeches and print summaries 
    for i in sc.get_speeches():
        print ''
        print('Speech: {}'.format(i.get_title())) ### print speech title
        print(i.get_summary(10))                  ### print the top 10 summary sentences

```


