from flask import *
from flask_paginate import Pagination, get_page_args
import csv
import sentimen_analysis as sa
import os
from sklearn.metrics import confusion_matrix
app = Flask(__name__)

from flask import flash
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'C:\\Users\\vicky\\OneDrive\\Documents\\SKRIPSI\\Project\\Web\\static\\files\\'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_data(offset=0, per_page=100):
    try:
        with open('reviews.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            data = list(reader)
    except UnicodeDecodeError:
        with open('reviews.csv', 'r', encoding='latin1') as file:
            reader = csv.reader(file)
            next(reader)
            data = list(reader)
    return data[offset: offset + per_page]

def get_data_preprocessed(offset=0, per_page=100):
    try:
        with open('preprocessed_review.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            data = list(reader)
    except UnicodeDecodeError:
        with open('preprocessed_review.csv', 'r', encoding='latin1') as file:
            reader = csv.reader(file)
            next(reader)
            data = list(reader)
    return data[offset: offset + per_page]

def get_data_testing(offset=0, per_page=100):
    try:
        with open('predicted_sentiments.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            data = list(reader)
    except UnicodeDecodeError:
        with open('predicted_sentiments.csv', 'r', encoding='latin1') as file:
            reader = csv.reader(file)
            next(reader)
            data = list(reader)
    return data[offset: offset + per_page]

def read_confusion_matrix_from_csv(file_path):
    confusion_matrix_data = {'0': {'0': 0, '1': 0}, '1': {'0': 0, '1': 0}}
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            label_actual = row[0]
            confusion_matrix_data[label_actual]['0'] = int(row[1])
            confusion_matrix_data[label_actual]['1'] = int(row[2])
    return confusion_matrix_data

def read_classification_metrics_from_csv(file_path):
    metrics = {}
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metrics[row['Metric']] = row['Value']
    return metrics

@app.route('/')
def index(): 
    return render_template('index.html', homeactive = True)

@app.route('/data')
def data():
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    per_page = 100
    offset = (page - 1) * per_page
    pagination_data = get_data(offset=offset, per_page=per_page)

    try:
        with open('reviews.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            total = sum(1 for row in reader)
    except UnicodeDecodeError:
        with open('reviews.csv', 'r', encoding='latin1') as file:
            reader = csv.reader(file)
            next(reader)
            total = sum(1 for row in reader)

    pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap5')
    return render_template('data.html', data=pagination_data, dataactive=True, pagination=pagination)

@app.route('/preprocessing')
def preprocessing(): 
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    per_page = 100
    offset = (page - 1) * per_page
    pagination_data = get_data_preprocessed(offset=offset, per_page=per_page)

    try:
        with open('preprocessed_review.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            total = sum(1 for row in reader)
    except UnicodeDecodeError:
        with open('preprocessed_review.csv', 'r', encoding='latin1') as file:
            reader = csv.reader(file)
            next(reader)
            total = sum(1 for row in reader)

    pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap5')
    return render_template('preprocessing.html', data = pagination_data, hasilactive = True, pagination=pagination)

@app.route('/sentimen')
def sentimen(): 
    data = sa.read_preprocessed_data()
    data_negatif = data[data['sentiment_label'] == 0]
    data_positif = data[data['sentiment_label'] == 1]

    return render_template('sentimen.html', data_all = len(data), data_negatif = len(data_negatif), data_positif = len(data_positif), hasilactive = True)

@app.route('/wordcloud')
def wordcloud(): 
    return render_template('wordcloud.html', hasilactive = True)

@app.route('/hasiltest')
def hasiltest(): 
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    per_page = 100
    offset = (page - 1) * per_page
    pagination_data = get_data_testing(offset=offset, per_page=per_page)

    try:
        with open('predicted_sentiments.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            total = sum(1 for row in reader)
    except UnicodeDecodeError:
        with open('predicted_sentiments.csv', 'r', encoding='latin1') as file:
            reader = csv.reader(file)
            next(reader)
            total = sum(1 for row in reader)

    pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap5')
    return render_template('hasiltest.html', data = pagination_data, hasilactive = True, pagination = pagination)

@app.route('/report')
def report():
    confusion_matrix_data = read_confusion_matrix_from_csv('confusion_matrix.csv')

    metrics = read_classification_metrics_from_csv('classification_metrics.csv')

    return render_template('report.html', confusion_matrix_data=confusion_matrix_data, metrics = metrics)
    
from flask import request

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    if request.method == 'POST':
        input_option = request.form.get('inputOption')
        if input_option == 'comment':
            comment = request.form['comment']
            hasil_prediksi, input_text_lower, cleaned_text, tokenized_text, stemmed_text, removed_stopwords, replaced_slang, final_text = sa.predict_sentiment(comment)
            return render_template('hasilprediksi.html', hasil_prediksi=[hasil_prediksi], kalimat_input=[comment], input_texts_lower=[input_text_lower], cleaned_texts=[cleaned_text], tokenized_texts=[tokenized_text], stemmed_texts=[stemmed_text], removed_stopwords=[removed_stopwords], replaced_slangs=[replaced_slang], final_texts=[final_text])
        elif input_option == 'file':
            if 'fileUpload' in request.files:
                file = request.files['fileUpload']
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    hasil_prediksi = []
                    kalimat_input = []
                    input_texts_lower = []
                    cleaned_texts = []
                    tokenized_texts = []
                    stemmed_texts = []
                    removed_stopwords = []
                    replaced_slangs = []
                    final_texts = []
                    for line in lines:
                        hasil, input_text_lower, cleaned_text, tokenized_text, stemmed_text, removed_stopword, replaced_slang, final_text = sa.predict_sentiment(line.strip())
                        hasil_prediksi.append(hasil)
                        kalimat_input.append(line.strip())
                        input_texts_lower.append(input_text_lower)
                        cleaned_texts.append(cleaned_text)
                        tokenized_texts.append(tokenized_text)
                        stemmed_texts.append(stemmed_text)
                        removed_stopwords.append(removed_stopword)
                        replaced_slangs.append(replaced_slang)
                        final_texts.append(final_text)
                    return render_template('hasilprediksi.html', hasil_prediksi=hasil_prediksi, kalimat_input=kalimat_input, input_texts_lower=input_texts_lower, cleaned_texts=cleaned_texts, tokenized_texts=tokenized_texts, stemmed_texts=stemmed_texts, removed_stopwords=removed_stopwords, replaced_slangs=replaced_slangs, final_texts=final_texts)
    return render_template('prediksi.html', prediksiactive=True)


if __name__ == '__main__':
    app.run(debug = True)