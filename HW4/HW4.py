from transformers import AutoModel, AutoTokenizer
from scipy import sparse
import numpy as np
import argparse
import torch
import json

model = AutoModel.from_pretrained('sberbank-ai/sbert_large_nlu_ru')
tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/sbert_large_nlu_ru')

"""
Чтение документов, на выходе - словарь вида {"вопрос": "ответ", ... }
"""
def read(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]
    qa = {}
    for thread in corpus:
        thread = json.loads(thread)
        question = thread['question']
        if thread['answers']:
            answer = sorted(thread['answers'],
                            key=lambda k: int(k['author_rating'].get('value', 0))
                            if k['author_rating'].get('value', 0) != '' else 0,
                            reverse=True)[0]['text']
            qa[question] = answer
    return qa


"""
Функция векторизации корпуса, на выходе получаем разреженную матрицу
"""
def bert(texts):
    vectors = []
    for text in texts:
        t = tokenizer(text, padding=True, truncation=True,  max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        vectors.append(embeddings[0].cpu().numpy())
    return sparse.csr_matrix(vectors)


"""
Функция определения релевантности документов для запроса
"""
def search(query, corpus):
    # questions = bert(list(corpus.keys()))
    questions = sparse.load_npz('bert_questions.npz')
    answers = np.array(list(corpus.values()))

    query = bert([query])
    BERT = np.dot(questions, query.T).toarray()
    scores = np.argsort(BERT, axis=0)[::-1]
    return answers[scores.ravel()]


"""
Точка входа: чтение файлов и поиск ответов по запросу
"""
def main(path, query, amount):
    corpus = read(path)
    docs = search(query, corpus)

    print('По запросу "{}" были найдены следующие ответы (выводится топ-{}): '.format(query, amount))
    for doc in enumerate(docs[:int(amount)], start=1):
        print('{}. {}'.format(doc[0], doc[1]))


"""
В качестве аргументов необходимо указать путь к папке с документами, 
запрос и максимальное число ответов для вывода, например:
    -path="data.jsonl" -query="Как дела?" -amount=5
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify the file, the query and the desired amount of answers.')
    parser.add_argument("-path", dest="path", required=False)
    parser.add_argument("-query", dest="query", required=True)
    parser.add_argument("-amount", dest="amount", required=True)
    args = parser.parse_args()

    main(path=args.path, query=args.query, amount=args.amount)
