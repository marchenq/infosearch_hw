from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pandas as pd
import pymorphy2
import argparse
import re
import os

stopwords = set(stopwords.words('russian'))

morph = pymorphy2.MorphAnalyzer()

vectorizer = TfidfVectorizer()


"""
Препроцессинг: лемматизация, приведение к нижнему регистру, удаление стоп-слов и пунктуации
"""
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = []
    for token in text.split():
        token = token.lower()
        if token.isalpha() and token not in stopwords:
            token = morph.normal_forms(token.strip())[0]
            tokens.append(token)
    return ' '.join(tokens)


"""
Чтение документов, на выходе - словарь вида {"название_документа": "список_лемм", ... }
"""
def read(folder):
    texts = {}
    for root, dirs, files in os.walk(folder):
        for name in files:
            with open(os.path.join(root, name), 'r', encoding='utf-8-sig') as f:
                text = ''.join(f.readlines()[:-3])
            preprocessed_text = preprocess(text)
            texts[name.split('.ru.txt')[0]] = preprocessed_text
    return texts


"""
Функция индексации корпуса, на выходе - матрица терм-документ, где строки - названия документов, столбцы - леммы, 
ячейки - метрика TF-IDF
"""
def corpus_index(corpus):
    X = vectorizer.fit_transform(corpus.values())
    td = pd.DataFrame(X.todense(), index=corpus.keys(), columns=vectorizer.get_feature_names())
    return td


"""
Функция индексации запроса, на выходе - вектор запроса
"""
def query_index(query):
    preprocessed_query = preprocess(query)
    query_vector = vectorizer.transform([preprocessed_query])
    return query_vector


"""
Подсчёт косинусных расстояний между запросом и каждым документом, на выходе - список документов, найденных по запросу, 
по убыванию расстояния
"""
def search(td, query_vector):
    cos = cosine_similarity(td, query_vector)
    td['cos_sim'] = cos
    td = td[td['cos_sim'] != 0]
    td = td.sort_values(by=['cos_sim'], ascending=False)
    return td.index


"""
Точка входа: чтение файлов, получение матрицы терм-документ, обработка запроса, поиск и вывод найденных документов
"""
def main(path, query):
    texts = read(path)

    td = corpus_index(texts)
    query_vector = query_index(query)

    found = search(td, query_vector)

    print('По запросу "{}" были найдены следующие документы (в порядке убывания):'.format(query))
    for i, document in enumerate(list(found), start=1):
        print('\t{}. {}'.format(i, document))


"""
В качестве аргументов необходимо указать путь к папке с документами и запрос, например:
    -path='C:/Users/Igor/Desktop/NLP-22/friends-data' -query='Книжный шкаф почти готов'
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify the folder and the query.')
    parser.add_argument("-path", dest="path", required=True)
    parser.add_argument("-query", dest="query", required=True)
    args = parser.parse_args()

    main(path=args.path, query=args.query)
