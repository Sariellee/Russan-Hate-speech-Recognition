import re
import string
from typing import List, Any

import nltk
import pymorphy2
from nltk import ngrams
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer


class Preprocessor:

    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()

    def preprocess(self, text: str) -> str:
        text_string = text

        space_pattern = '\s+'
        giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                           '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        mention_regex = '@[\w\-]+'
        hashtag_regex = '#[\w\-]+'
        retweet_regex = 'RT'
        e_regex = 'ё'
        e_big_regex = 'Ё'
        parsed_text = re.sub(space_pattern, ' ', text_string)
        parsed_text = re.sub(giant_url_regex, '', parsed_text)  # urls do not give information
        parsed_text = re.sub(mention_regex, '@', parsed_text)
        parsed_text = re.sub(hashtag_regex, '#', parsed_text)
        parsed_text = re.sub(retweet_regex, 'retweethere', parsed_text)
        parsed_text = re.sub(e_regex, 'е', parsed_text)
        parsed_text = re.sub(e_big_regex, 'Е', parsed_text)

        return parsed_text

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.morph.parse(token)[0].normal_form for token in tokens]

    def tokenize(self, sentence: str) -> List[str]:
        return self.lemmatize(nltk.word_tokenize(sentence))

    def get_TFIDF_features(self, df: DataFrame, filter_stopwords: bool = False) -> Any:
        stopwords = nltk.corpus.stopwords.words("russian")

        vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize,
            preprocessor=self.preprocess,
            ngram_range=(1, 3),
            stop_words=stopwords if filter_stopwords else None,  # somehow the score is better if we keep the stopwords
            use_idf=True,
            smooth_idf=False,
            norm=None,
            decode_error='replace',
            max_features=10000,
            min_df=5,
            max_df=0.501
        )

        tfidf = vectorizer.fit_transform(df['text']).toarray()
        vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
        idf_vals = vectorizer.idf_
        idf_dict = {i: idf_vals[i] for i in vocab.values()}

        return tfidf, vocab, idf_dict
