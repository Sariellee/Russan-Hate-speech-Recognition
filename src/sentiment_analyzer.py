from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
from pandas import DataFrame


class SentimentAnalyzer:

    def __init__(self):
        self.tokenizer = RegexTokenizer()
        self.model = FastTextSocialNetworkModel(tokenizer=self.tokenizer)

    def _map_sentiment(self, sentiment):
        if sentiment == "positive":
            return 1
        if sentiment == "negative":
            return -1
        else:
            return 0

    def sentiment_label_dataframe(self, df: DataFrame) -> DataFrame:
        df['sentiment'] = None
        for i, row in df.iterrows():
            results = self.model.predict([row['text']], k=30)
            sentiment = max(list(results[0].items()), key=lambda prob: prob[1])[0]
            sentiment = self._map_sentiment(sentiment)
            df.at[i, 'sentiment'] = sentiment

        return df

    def sentiment_label_sentence(self, sentence: str):
        results = self.model.predict([sentence], k=30)
        sentiment = max(list(results[0].items()), key=lambda prob: prob[1])[0]
        return self._map_sentiment(sentiment)
