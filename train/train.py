import sys
import os
import pandas as pd
from utils.data_loader import load_and_process_data
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from models.naive_bayes import MultinomialNaiveBayes
from gensim.corpora.dictionary import Dictionary
from models.Fasttext import FastText
from models.TextCNN import TextCNN
from sklearn.metrics.classification import precision_score, recall_score, f1_score, classification_report
from models.bert import BERTClassifier


if __name__ == '__main__':
    if os.path.exists(os.path.join(os.getcwd(), 'data/total_processed_data.csv')):
        processed_data = pd.read_csv(os.path.join(os.getcwd(), 'data/total_processed_data.csv'))
    else:
        processed_data = load_and_process_data('data/百度题库')
    processed_data = processed_data.dropna(subset=['processed_item'])
    processed_data = shuffle(processed_data).reset_index(drop=True)

    ###### Naive Bayes
    vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\b\w+\b')
    X = vectorizer.fit_transform(processed_data['item']).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, processed_data['label'],
                                                        test_size=0.1, random_state=2020)

    ### Test model with different parameters
    # alpha = 1, fit_prior=False
    my_mnb = MultinomialNaiveBayes(alpha=1, fit_prior=False)
    my_mnb = my_mnb.fit(X_train, y_train)
    my_y_pred = my_mnb.predict(X_test)

    sklearn_mnb = MultinomialNB(alpha=1, fit_prior=False)
    sklearn_mnb = sklearn_mnb.fit(X_train, y_train)
    sklearn_y_pred = sklearn_mnb.predict(X_test)

    assert (my_y_pred == sklearn_y_pred).all()

    # alpha = 1, fit_prior=True
    my_mnb = MultinomialNaiveBayes(alpha=1, fit_prior=True)
    my_mnb = my_mnb.fit(X_train, y_train)
    my_y_pred = my_mnb.predict(X_test)

    sklearn_mnb = MultinomialNB(alpha=1, fit_prior=True)
    sklearn_mnb = sklearn_mnb.fit(X_train, y_train)
    sklearn_y_pred = sklearn_mnb.predict(X_test)

    assert (my_y_pred == sklearn_y_pred).all()

    # alpha = 0.8, fit_prior=True
    my_mnb = MultinomialNaiveBayes(alpha=0.8, fit_prior=True)
    my_mnb = my_mnb.fit(X_train, y_train)
    my_y_pred = my_mnb.predict(X_test)

    sklearn_mnb = MultinomialNB(alpha=0.8, fit_prior=True)
    sklearn_mnb = sklearn_mnb.fit(X_train, y_train)
    sklearn_y_pred = sklearn_mnb.predict(X_test)

    assert (my_y_pred == sklearn_y_pred).all()


    ###### my defined fasttext
    train_data, test_data = train_test_split(processed_data[['label', 'item']],
                                             test_size=0.1, random_state=2020)

    fasttext = FastText(class_num=3, class_type='multi-class', ngram_range=2)
    fasttext.fit(train_data['item'], train_data['label'], epochs=5)
    y_pred = fasttext.predict(test_data['item'])
    y_true = fasttext.y_encoder.transform(test_data['label'])
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')

    ##### textCNN
    ## multi-class test
    train_data, test_data = train_test_split(processed_data[['subject', 'processed_item']],
                                             test_size=0.1, random_state=2020)

    text_cnn = TextCNN(class_num=4, class_type='multi-class')
    text_cnn.fit(train_data['processed_item'], train_data['subject'],
                 validation_data=(test_data['processed_item'], test_data['subject']), epochs=2)
    y_true = text_cnn.y_encoder.transform(test_data['subject'])
    y_pred = text_cnn.predict(test_data['processed_item'])
    print(classification_report(y_true, y_pred))

    ## multi-label test
    train_data, test_data = train_test_split(processed_data[['labels', 'processed_item']],
                                             test_size=0.1, random_state=2020)
    text_cnn = TextCNN(class_num=97, class_type='multi-label')
    text_cnn.fit(train_data['processed_item'], train_data['labels'],
                 validation_data=(test_data['processed_item'], test_data['labels']), epochs=20)
    y_true = text_cnn.y_encoder.transform([item.split(' ') for item in test_data['labels']])
    y_pred = text_cnn.predict(test_data['processed_item'])
    print(classification_report(y_true, y_pred))

    ##### BertClassifier
    ## multi-class test
    if os.path.exists(os.path.join(os.getcwd(), 'data/total_processed_data.csv')):
        processed_data = pd.read_csv(os.path.join(os.getcwd(), 'data/total_processed_data.csv'))
    else:
        processed_data = load_and_process_data('data/百度题库', clean=False, use_jieba=False)
    processed_data = processed_data.dropna(subset=['processed_item'])
    processed_data = shuffle(processed_data).reset_index(drop=True)

    train_data, test_data = train_test_split(processed_data[['subject', 'processed_item']],
                                             test_size=0.1, random_state=2020)

    bert_classifier = BERTClassifier(class_num=4, class_type='multi-class', model_name='bert-base-chinese')
    bert_classifier.fit(train_data['processed_item'], train_data['subject'], epochs=1)
    y_true = bert_classifier.y_encoder.transform(test_data['subject'])
    y_pred = bert_classifier.predict_in_batch(test_data['processed_item'])
    print(classification_report(y_true, y_pred))

    ## multi-label test
    train_data, test_data = train_test_split(processed_data[['labels', 'processed_item']],
                                             test_size=0.1, random_state=2020)
    bert_classifier = BERTClassifier(class_num=97, class_type='multi-label', model_name='bert-base-chinese')
    bert_classifier.fit(train_data['processed_item'], train_data['labels'], epochs=5)
    y_true = bert_classifier.y_encoder.transform([item.split(' ') for item in test_data['labels']])
    y_pred = bert_classifier.predict_in_batch(test_data['processed_item'])
    print(classification_report(y_true, y_pred))

