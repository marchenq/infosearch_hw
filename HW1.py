from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np
import pymorphy2
import argparse
import re
import os

# nltk.download("stopwords")
stopwords = set(stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()

# ; Объявление констант
CHARACTERS = [['Моника', 'Мон'],
              ['Рэйчел', 'Рейч', 'Рэйч'],
              ['Чендлер', 'Чэндлер', 'Чен'],
              ['Фиби', 'Фибс'],
              ['Росс'],
              ['Джоуи', 'Джои', 'Джо']]


"""
Препроцессинг: лемматизация, приведение к нижнему регистру, удаление стоп-слов и пунктуации
"""
def preprocess(file):
    with open(file, 'r', encoding='utf-8-sig') as f:
        text = ''.join(f.readlines()[:-3])
    text = re.sub(r'[^\w\s]', '', text)
    tokens = []
    for token in text.split():
        token = token.lower()
        if token.isalpha() and token not in stopwords:
            token = morph.normal_forms(token.strip())[0]
            tokens.append(token)
    return tokens


"""
Построение обратной матрицы в виде словаря в формате:
"слово": {
    "документ": число_вхождений_в_документе,
    ...
},
...
"""
def dictionary(texts):
    reversed_dictionary = {}
    for k, v in texts.items():
        for x in v:
            b = reversed_dictionary.setdefault(x, {})
            b[k] = b.get(k, 0) + 1
    return reversed_dictionary


"""
Поиск ответов на вопросы с помощью обратного индекса в виде словаря
"""
def dictionary_answers(texts):
    reversed_dictionary = dictionary(texts)

    # Считаем частотность слова по всем текстам
    counts = {}
    for word, documents in reversed_dictionary.items():
        for document, count in documents.items():
            counts[word] = counts.get(word, 0) + count

    sorted_dict = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    most_frequent_word = list(sorted_dict.items())[0][0]
    least_frequent_word = list(sorted_dict.items())[-1][0]

    # Слово должно быть представлено в 165 документах
    found_in_every_document = [word for word, documents in reversed_dictionary.items() if len(documents) == 165]

    # Считаем число вхождений имён персонажей
    chars = {}
    for character in CHARACTERS:
        count = 0
        for name in character:
            try:
                count += counts[name.lower()]
            except KeyError:
                pass
        chars[character[0]] = 0
        chars[character[0]] += count
    most_frequent_char = sorted(chars.items(), key=lambda x: x[1], reverse=True)[0][0]

    return most_frequent_word, least_frequent_word, found_in_every_document, most_frequent_char


"""
Поиск ответов на вопросы с помощью обратного индекса в виде матрицы
"""
def matrix_answers(texts):
    corpus = [' '.join(x) for x in texts.values()]
    vectorizer = CountVectorizer(analyzer='word')
    X = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()

    # Формируем матрицу частотностей
    freq_matrix = np.asarray(X.sum(axis=0)).ravel()
    most_frequent_word = features[np.argmax(freq_matrix)]
    least_frequent_word = features[np.argmin(freq_matrix)]

    # Ищем слова, для которых нет 0 ни в одном тексте
    not_empty = np.apply_along_axis(lambda x: 0 not in x, 0, X.toarray())
    indices = np.where(not_empty)[0]
    found_in_every_document = [features[i] for i in indices]

    # Считаем число вхождений имён персонажей
    chars = {}
    for character in CHARACTERS:
        count = 0
        for name in character:
            index = vectorizer.vocabulary_.get(name.lower())
            if index:
                count += X.T[index].sum()
        chars[character[0]] = 0
        chars[character[0]] += count
    most_frequent_char = sorted(chars.items(), key=lambda x: x[1], reverse=True)[0][0]

    return most_frequent_word, least_frequent_word, found_in_every_document, most_frequent_char


"""
Функция для вывода на экран ответов на вопросы
"""
def print_answers(most_frequent_word, least_frequent_word, found_in_every_document, most_frequent_char, mode):
    print('======================================================================================' + '\n' +
          'Обратный индекс в виде {}'.format(mode) + '\n' +
          '======================================================================================')
    print('a) Самое частотное слово: {}'.format(most_frequent_word))
    print('b) Самое редкое слово (среди других с частотой = 1): {}'.format(least_frequent_word))
    print('c) Набор слов, которые есть во всех документах коллекции: {}'.format(', '.join(found_in_every_document)))
    print('d) Самый популярный персонаж: {}'.format(most_frequent_char), '\n')


"""
Точка входа, формируется словарь с текстами, где ключ - номер документа, значение - список лемм в документе
На выходе получаем ответы на вопросы
"""
def main(path):
    texts = {}
    i = 1
    for root, dirs, files in os.walk(path):
        for name in files:
            preprocessed_text = preprocess(os.path.join(root, name))
            texts[i] = preprocessed_text
            i += 1

    D_most_frequent_word, D_least_frequent_word, D_found_in_every_document, D_most_frequent_char = dictionary_answers(texts)
    M_most_frequent_word, M_least_frequent_word, M_found_in_every_document, M_most_frequent_char = matrix_answers(texts)

    print_answers(D_most_frequent_word,
                  D_least_frequent_word,
                  D_found_in_every_document,
                  D_most_frequent_char, 'обратного словаря')
    print_answers(M_most_frequent_word,
                  M_least_frequent_word,
                  M_found_in_every_document,
                  M_most_frequent_char, 'матрицы')


"""
В качестве аргумента необходимо указать путь к папке с документами, например:
    -path C:/Users/Igor/Desktop/NLP-22/friends-data
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify the folder.')
    parser.add_argument("-path", dest="path", required=True)
    args = parser.parse_args()
    main(path=args.path)
