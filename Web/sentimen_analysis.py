from google_play_scraper import app, Sort, reviews_all
import pandas as pd
import numpy as np
import re
import nltk.corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from indoNLP.preprocessing import replace_slang, replace_word_elongation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
import graphviz
from sklearn.tree import export_graphviz
import os

data = []

def fetch_data():
    result = reviews_all(
        'superapps.polri.presisi.presisi',
        lang = 'id',
        country = 'id',
        sort = Sort.NEWEST,
    )
    data = pd.DataFrame(np.array(result), columns = ['review'])
    data = data.join(pd.DataFrame(data.pop('review').tolist()))
    data = data[['userName', 'score', 'at', 'content']]
    data.to_csv("review.csv", index=False)

def read_data():
    data = pd.read_csv('reviews.csv')
    data = data.dropna()
    data.isna().sum()
    sentiment_counts = data['sentiment_label'].value_counts()
    sentiment_counts.to_csv('sentiment_counts.csv', header=['count'])
    return data

def read_preprocessed_data():
    data = pd.read_csv('preprocessed_review.csv')
    data = data.dropna()
    data.isna().sum()
    data = data[data['sentiment_label'] != 2]
    return data

def clean_text(text):
    text = re.sub(r'\\x[A-Za-z0-9./]+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    return text

def stemming(words):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    stemmed_words = []

    exclude = ['lemot', 'guna', 'tidak', 'belum', 'bisa', 'belom', 'masalah', 'seneng', 'lama', 'gini']
    for w in words:
        if w in exclude:
            stemmed_words.append(w)
        else:
            stemmed_words.append(stemmer.stem(w))
    
    print(stemmed_words)
    return stemmed_words

def read_additional_stopwords():
    with open('stopword.txt', 'r') as file:
        additional_stopwords = [line.strip() for line in file.readlines()]
    return additional_stopwords

def read_additional_synonyms():
    synonyms_dict = {}
    with open("remove_synonym.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            words = line.strip().split(', ')
            if len(words) > 1:
                base_word = words[0]
                synonyms = words[1:]
                for synonym in synonyms:
                    if synonym != base_word:
                        synonyms_dict[synonym] = base_word
    return synonyms_dict

def replace_synonyms(text, synonyms_dict):
    words = text
    for i in range(len(words)):
        if words[i] in synonyms_dict:
            words[i] = synonyms_dict[words[i]]
    return ' '.join(words)

def plot_confusion_matrix(conf_matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix Heatmap')
    plt.savefig('static/image/confussion_matrix.png', format='png', bbox_inches='tight', transparent=True)

def preprocessing_data():
    data = read_data()
    data = data.dropna()
    data.duplicated().sum()
    
    data['lower_content'] = data['content'].str.lower()

    data['clean_content'] = data['lower_content'].apply(clean_text)

    data['tokenized'] = data['clean_content'].apply(lambda x: word_tokenize(x))

    data['stemmed'] = data['tokenized'].apply(stemming)

    nltk.download('stopwords')
    stop = []
    stop = stopwords.words('indonesian')
    additional_stopwords = read_additional_stopwords() 
    stop.extend(additional_stopwords)
    exclude_words = ['tidak']
    data['removed_stopwords'] = data['stemmed'].apply(lambda x: [word for word in x if word not in stop or word in exclude_words])

    data['replace_slang'] = data['removed_stopwords'].apply(lambda x: replace_word_elongation(" ".join(x)).split())
    data['replace_slang'] = data['replace_slang'].apply(lambda x: replace_slang(" ".join(x)).split())
    data['final'] = data['replace_slang'].apply(lambda x: replace_synonyms(x, read_additional_synonyms()))

    data.to_csv("preprocessed_review.csv")

def wordcloud_all():
    data = read_preprocessed_data()

    data_negatif = data[data['sentiment_label'] == 0]
    data_positif = data[data['sentiment_label'] == 1]

    all_text_s0 = ' '.join(data_negatif["final"])
    wordcloud_negatif = WordCloud(colormap='Accent', width=1280, height=720, mode='RGBA', background_color='white').generate(all_text_s0)
    plt.figure(figsize=(9, 6))
    plt.imshow(wordcloud_negatif, interpolation='bilinear')
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.savefig('static/image/wordcloud_negative.png', format='png', bbox_inches='tight', transparent=True)

    all_text_s1 = ' '.join(data_positif["final"])
    wordcloud_positif = WordCloud(colormap='Accent', width=1280, height=720, mode='RGBA', background_color='white').generate(all_text_s1)
    plt.figure(figsize=(9, 6))
    plt.imshow(wordcloud_positif, interpolation='bilinear')
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.savefig('static/image/wordcloud_positive.png', format='png', bbox_inches='tight', transparent=True)

def random_forest_modelling():
    data = read_preprocessed_data()
    
    x = data['final']
    y = data['sentiment_label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    train_count = len(x_train)
    test_count = len(x_test)

    train_test_counts = pd.DataFrame({'dataset': ['x_train', 'x_test'], 'count': [train_count, test_count]})
    train_test_counts.to_csv('train_test_counts.csv', index=False)

    tfidf_vectorizer = TfidfVectorizer()
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(x_train_tfidf, y_train)
    y_pred_rf = random_forest_classifier.predict(x_test_tfidf)
    joblib.dump(random_forest_classifier, 'random_forest_model.pkl')

    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
    labels = ['Negatif', 'Positif']
    confusion_matrix_df = pd.DataFrame(conf_matrix_rf)
    confusion_matrix_df.to_csv('confusion_matrix.csv')
    plot_confusion_matrix(conf_matrix_rf, labels)
    
    loaded_model = joblib.load('random_forest_model.pkl')

    y_pred_loaded_model = loaded_model.predict(x_test_tfidf)
    classification_rep_rf = classification_report(y_test, y_pred_loaded_model, target_names=['Negatif', 'Positif'])

    classification_rep_dict = classification_report(y_test, y_pred_loaded_model, target_names=['Negatif', 'Positif'], output_dict=True)

    accuracy = classification_rep_dict['accuracy']
    precision_pos = classification_rep_dict['Positif']['precision']
    precision_neg = classification_rep_dict['Negatif']['precision']
    recall_pos = classification_rep_dict['Positif']['recall']
    recall_neg = classification_rep_dict['Negatif']['recall']
    f1_score_pos = classification_rep_dict['Positif']['f1-score']
    f1_score_neg = classification_rep_dict['Negatif']['f1-score']

    result_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision Positive', 'Precision Negative', 'Recall Positive', 'Recall Negative', 'F1 Score Positive', 'F1 Score Negative'],
        'Value': [accuracy, precision_pos, precision_neg, recall_pos, recall_neg, f1_score_pos, f1_score_neg]
    })

    result_df.to_csv('classification_metrics.csv', index=False)

    data_test = pd.DataFrame({'Comment': x_test, 'Actual_Sentiment': y_test})
    data_test['Predicted_Sentiment'] = y_pred_loaded_model

    data_test.to_csv('predicted_sentiments.csv', index=False)

    fig, ax = plt.subplots(figsize=(4, 1))
    ax.text(0.01, 0.5, classification_rep_rf, {'fontsize': 12}, fontfamily='monospace')
    ax.axis('off')

    plt.savefig('static/image/classification_report.png', bbox_inches='tight', dpi=300)

    for i, tree in enumerate(random_forest_classifier.estimators_):
        dot_data = export_graphviz(tree, out_file=None,
                                   feature_names=tfidf_vectorizer.get_feature_names_out(),
                                   class_names=['negatif', 'positif'],
                                   filled=True, rounded=True,
                                   special_characters=True)

        graph = graphviz.Source(dot_data)

        filename = os.path.join('C:\\Users\\vicky\\OneDrive\\Documents\\SKRIPSI\\Project\\Web\\tree', f"decision_tree_{i}.png")
        graph.render(filename)

        print("Decision tree", i, "exported as", filename)

def get_tfidf_feature_count():
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    feature_count = len(tfidf_vectorizer.get_feature_names_out())
    return feature_count

def predict_sentiment(input_text):
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    loaded_model = joblib.load('random_forest_model.pkl')

    input_text_lower = input_text.lower()
    cleaned_text = clean_text(input_text_lower)
    tokenized_text = word_tokenize(cleaned_text)
    nltk.download('stopwords')
    stop = []
    stop = stopwords.words('indonesian')
    additional_stopwords = read_additional_stopwords() 
    stop.extend(additional_stopwords)
    stemmed_text = stemming(tokenized_text)
    removed_stopwords = [word for word in stemmed_text if word not in stop]
    replaced_slang = replace_word_elongation(" ".join(removed_stopwords)).split()
    replaced_slang = replace_slang(" ".join(replaced_slang)).split()
    synonyms_dict = read_additional_synonyms()
    final_text = replace_synonyms(replaced_slang, synonyms_dict)

    input_tfidf = tfidf_vectorizer.transform([final_text])

    predicted_sentiment = loaded_model.predict(input_tfidf)

    return predicted_sentiment[0], input_text_lower, cleaned_text, tokenized_text, stemmed_text, removed_stopwords, replaced_slang, final_text


def count_features_per_tree():
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

    random_forest_classifier = joblib.load('random_forest_model.pkl')

    feature_names = tfidf_vectorizer.get_feature_names_out()

    for i, tree in enumerate(random_forest_classifier.estimators_):
        features_used = tree.tree_.feature
        features_used = features_used[features_used >= 0]
        num_features_used = len(set(features_used))
        print(f"Tree {i + 1} uses {num_features_used} features.")


# preprocessing_data()
# wordcloud_all()
# random_forest_modelling()
# read_data()
# print(get_tfidf_feature_count())
# count_features_per_tree()