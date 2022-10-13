import numpy as np
from scipy import sparse

def metrics(index, query):
    scores = np.dot(index, query.T).toarray()
    sorted_scores = np.argsort(scores, axis=0)[::-1]

    i = 0
    for index, row in enumerate(sorted_scores):
        output = row[:5]
        if index in output:
            i += 1
    score = i / len(sorted_scores)
    return score


def main():
    matrices = {'bert_questions_10000.npz': 'bert_answers_10000.npz',
                'bm25_questions_10000.npz': 'bm25_answers_10000.npz'}

    scores = {}
    for key, value in matrices.items():
        index = sparse.load_npz(key)
        query = sparse.load_npz(value)

        scores[key[:4]] = metrics(index, query)

    for mode, score in scores.items():
        print('{}: {}'.format(mode, score))


if __name__ == '__main__':
    main()
