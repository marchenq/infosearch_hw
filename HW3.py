from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from scipy import sparse
import numpy as np
import pymorphy2
import argparse
import re
import json

stopwords = set(stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()

count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')


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
Чтение документов, на выходе - словарь вида {"вопрос": "ответ", ... }
"""
def read(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]
    qa = {}
    for thread in corpus:
        thread = json.loads(thread)
        question = preprocess(thread['question'])
        if thread['answers']:
            answer = sorted(thread['answers'],
                            key=lambda k: int(k['author_rating'].get('value', 0))
                            if k['author_rating'].get('value', 0) != '' else 0,
                            reverse=True)[0]['text']
            qa[question] = answer
    return qa


"""
Функция индексации корпуса, на выходе получаем разреженную матрицу
"""
def corpus_index(corpus, k=2, b=0.75):
    rows, columns, matrices = [], [], []

    count = count_vectorizer.fit_transform(corpus)
    tfidf = tfidf_vectorizer.fit_transform(corpus)

    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)

    len_d = count.sum(axis=1)
    avdl = len_d.mean()

    # расчет знаменателя
    B_1 = (k * (1 - b + b * len_d / avdl))
    B_1 = np.expand_dims(B_1, axis=-1)

    for i, j in zip(*tfidf.nonzero()):
        rows.append(i)
        columns.append(j)
        A = idf[0][j] * tfidf[i, j] * (k + 1)
        B = tfidf[i, j] + B_1[i]
        matrix = A / B
        matrices.append(matrix[0][0])

    return sparse.csr_matrix((matrices, (rows, columns)))


"""
Функция индексации запроса, на выходе - вектор запроса
"""
def query_index(query):
    preprocessed_query = preprocess(query)
    query_vector = tfidf_vectorizer.transform([preprocessed_query])
    return query_vector


"""
Функция определения релевантности документов для запроса
"""
def search(query, corpus):
    questions = corpus_index(list(corpus.keys()))
    answers = np.array(list(corpus.values()))

    query = query_index(query)
    BM25 = np.dot(questions, query.T).toarray()
    scores = np.argsort(BM25, axis=0)[::-1]
    return answers[scores.ravel()]


"""
Точка входа: чтение файлов и поиск ответов по запросу
"""
def main(path, query, amount):
    corpus = read(path)
    docs = search(query, corpus)

    print('По запросу "{}" были найдены следующие ответы (выводится топ-{}): '.format(query, amount))
    for doc in enumerate(docs[:amount], start=1):
        print('{}. {}'.format(doc[0], doc[1]))


"""
В качестве аргументов необходимо указать путь к папке с документами, 
запрос и максимальное число ответов для вывода, например:
    -path="data.jsonl" -query="Как дела? -amount=5"
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify the folder, the query and the desired amount of answers.')
    parser.add_argument("-path", dest="path", required=True)
    parser.add_argument("-query", dest="query", required=True)
    parser.add_argument("-amount", dest="amount", required=True)
    args = parser.parse_args()

    main(path=args.path, query=args.query, amount=args.amount)
