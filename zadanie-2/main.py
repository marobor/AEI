import os, re, marshal, math
import numpy as np
from pprint import pprint
from collections import Counter
from clp3 import clp
from scipy import sparse

dir_1 = './teksty-na-dowolny-temat'
dir_2 = './teksty-o-świnkach-morskich'


def tf(text):
    words = text.split()
    tf_base = len(words)
    for i in range(len(words)):
        id = clp(words[i])
        if id:
            words[i] = clp.bform(id[0])
            # print(word)
    word_counter = Counter(words)
    t_f = dict(word_counter)

    for key, value in t_f.items():
        t_f[key] = value / tf_base
    # dodać w marshalu krotkę: pierwsza wartosć to ilość słów w tekście, druga to słownik t_f
    return t_f


def add_row(filename, tf_dict, matrix):
    matrix.append([0 for _ in range(len(matrix[0]))])
    matrix[len(matrix) - 1][0] = filename
    for key, value in tf_dict.items():
        if key in matrix[0]:
            i = matrix[0].index(key)
            matrix[len(matrix) - 1][i] = value
        else:
            matrix[0].append(key)
            matrix[len(matrix) - 1].append(value)

    return matrix


def idf_tfidf(matrix):
    bare_matrix = matrix.copy()
    idf_result = []
    del bare_matrix[0]

    for row in bare_matrix:
        del row[0]

    max_len = len(bare_matrix[len(bare_matrix) - 1])
    corpus_lenght = len(bare_matrix)
    padded_matrix = np.array([np.pad(row, (0, max_len - len(row)), 'constant') for row in bare_matrix])

    # Może być użyte zamiennie z pętlą usuwającą pierwsze indeksy w rzędach kilka linijek wyżej.
    # padded_matrix = np.delete(padded_matrix, 0, axis=1)

    for vector in range(max_len):
        word_counter = np.count_nonzero(padded_matrix[:, vector])
        word_idf = math.log(corpus_lenght / word_counter)
        idf_result.append(word_idf)

    idf_vector = np.array(idf_result)

    tfidf_result = padded_matrix * idf_vector

    return idf_result, tfidf_result


def get_result():
    pass


def read_files(directory):
    corpus_matrix = []
    all_texts = ''
    i = 0
    for filename in os.listdir(directory):
        # 0print(filename)
        if filename.endswith('.txt'):
            try:
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    text = re.sub(r'[\d\W_]', ' ', text)
                    t_f = tf(text)

                    if corpus_matrix:
                        corpus_matrix = add_row(filename, t_f, corpus_matrix)
                    else:
                        initial_row = [''] + list(t_f.keys())
                        row = [filename] + list(t_f.values())
                        corpus_matrix.append(initial_row)
                        corpus_matrix.append(row)

                    # ogranicznik
                    # if i == 30:
                    #     break
                    # i += 1

            except FileNotFoundError:
                print('Plik nie został znaleziony')

    word_list = corpus_matrix[0].copy()
    del word_list[0]

    idf, tfidf = idf_tfidf(corpus_matrix)

    result = get_result()




read_files(dir_2)
