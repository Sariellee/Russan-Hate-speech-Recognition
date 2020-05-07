import nltk
import numpy as np
from pandas import DataFrame
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from dataset import read_dataset, balance_dataset
from feature_extraction import FeatureExtractor
from preprocessing import Preprocessor
from sentiment_analyzer import SentimentAnalyzer

nltk.download('punkt')
nltk.download('stopwords')

if __name__ == "__main__":

    # read the dataset
    df = read_dataset()

    # make the dataset balanced (stratify)
    df = balance_dataset(df)

    # preprocessing stage 1: Regular expressions & Filters
    print("preprocessing stage 1..")
    preprocessor = Preprocessor()
    df['text'] = df['text'].apply(preprocessor.preprocess)

    # extract the important features
    print("extraction of features..")
    feature_extractor = FeatureExtractor()
    df['offensive_words_count'] = df['text'].apply(feature_extractor.find_offensive_words)
    df['caps_words_count'] = df['text'].apply(feature_extractor.find_capsed_words)

    # if there is no sentiment data, label it automatically
    if not df['sentiment'].any():
        print("initiating sentiment analyzer..")
        df = SentimentAnalyzer().sentiment_label_dataframe(df)

    print("acquring TF-IDF features..")
    tfidf, vocab, idf_dict = preprocessor.get_TFIDF_features(df, filter_stopwords=False)

    # preprocessing stage 2: Tokenize & Lemmatize
    print("preprocessing stage 2..")
    df['text'] = df['text'].apply(preprocessor.tokenize)
    df['text'] = df['text'].apply(preprocessor.lemmatize)

    print("extracting features..")
    feats = feature_extractor.get_feature_array(df)
    M = np.concatenate([tfidf, feats], axis=1)

    variables = [''] * len(vocab)
    for k, v in vocab.items():
        variables[v] = k

    feature_names = variables + feature_extractor.get_feature_names()

    #########
    # MODEL #
    #########

    X = DataFrame(M)
    y = df['hate_speech'].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

    print("training model..")
    model = LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge', multi_class='ovr',
                      max_iter=3000, dual=False)
    model.fit(x_train, y_train)

    y_preds = model.predict(x_test)

    report = classification_report(y_test, y_preds)

    print(report)

    plot_confusion_matrix(model, x_test, y_test)
