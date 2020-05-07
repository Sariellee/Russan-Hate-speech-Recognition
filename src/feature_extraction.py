import string
from typing import List, Tuple

import nltk
import pymorphy2
from nltk import ngrams
from pandas import DataFrame
from textstat.textstat import *
import numpy as np


class FeatureExtractor:

    def __init__(self, offensive_words_path: str = 'data/final_offensive_words_list.csv'):
        with open(offensive_words_path) as f:
            offensive_words = f.readlines()
        self.offensive_words = set(list(map(lambda word: word.strip(), offensive_words)))

        self.morph = pymorphy2.MorphAnalyzer()

    def find_offensive_words(self, sentence: str) -> int:
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))  # remove punct
        sentence = nltk.word_tokenize(sentence.lower())  # tokenize
        word_list = sentence + list(map(lambda tup: " ".join(tup), ngrams(sentence, 2)))  # add 2-grams
        word_list += list(map(lambda tup: " ".join(tup), ngrams(sentence, 3)))

        off_words = list(
            filter(lambda x: x in self.offensive_words,
                   word_list))  # this is our preliminary offensive words in sentence

        # now pymorphy2 variations of the words that have not been found earlier
        sentence = list(filter(lambda word: word not in off_words, sentence))

        for word in sentence:
            morphs = []
            for variation in self.morph.parse(word):
                morphs.append(variation.normal_form)
            off_words += (list(set(morphs).intersection(self.offensive_words)))

        return len(off_words)

    def find_capsed_words(self, sentence: str) -> int:
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))  # remove punct
        sentence = sentence.translate(str.maketrans('', '', string.digits))  # remove digits (they are always "capsed")
        words = nltk.word_tokenize(sentence)  # tokenize

        return len([word for word in words if word.upper() == word and len(word) != 1])

    def _find_common_features(self, row: Tuple) -> List:

        # one-hot sentiment
        sentiment = row[1]['sentiment']
        sentiment = {
            'neg': 1 if sentiment == -1 else 0,
            'pos': 1 if sentiment == 0 else 0,
            'neu': 1 if sentiment == 1 else 0
        }

        sentence = " ".join(row[1]['text'])  # this is whole sentence
        words = row[1]['text']  # this is sentence tokenized

        syllables = textstat.syllable_count(sentence)  # count syllables in words
        chars_count = sum(len(w) for w in words)  # num chars in words without the space
        chars_total_count = len(sentence)  # with spaces
        terms_count = len(sentence.split())  # number of words in sentence
        word_count = len(words)  # number of words in a sentence
        avg_syl = round(float((syllables + 0.001)) / float(word_count + 0.001),
                        4)  # average number of syllables per sentence
        unique_terms_count = len(set(words))  # number of unique terms

        capsed_count = row[1]['caps_words_count']
        offensive_words_count = row[1]['offensive_words_count']

        ###Modified FK grade, where avg words per sentence is just num words/1
        # FK grade for one sentence (avg_words_per_sentence==word_count)
        FKRA = round(float(0.39 * float(word_count) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)
        ##Modified FRE score, where sentence fixed to 1
        FRE = round(206.835 - 1.015 * (float(word_count) / 1.0) - (84.6 * float(avg_syl)), 2)

        # twitter_objs = count_twitter_objs(sentence) #Count #, @, and http://
        retweet = 0
        if "retweethere" in words:
            retweet = 1
        features = [FKRA, FRE, syllables, avg_syl, chars_count, chars_total_count, terms_count, word_count,
                    unique_terms_count, sentiment['neg'], sentiment['pos'], sentiment['neu'],
                    retweet, capsed_count, offensive_words_count]
        return features

    def get_feature_names(self):
        return ["FKRA", "FRE", "syllables_count", "avg_syl_per_word", "chars_count", "chars_total_count",
                "terms_count", "words_count", "unique_words_count", "sentiment neg", "sentiment pos", "sentiment neu",
                "is_retweet", "capsed_count", "offensive_word_count"]

    def get_feature_array(self, df: DataFrame) -> np.array:
        feats = []
        for t in df.iterrows():
            feats.append(self._find_common_features(t))
        return np.array(feats)
