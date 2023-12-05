import os
import re
import marshal
import math
import numpy as np
from collections import Counter
from clp3 import clp

dir_1 = './teksty-o-świnkach-morskich'


def pad_matrix(matrix):
    bare_matrix = matrix.copy()
    del bare_matrix[0]

    for row in bare_matrix:
        del row[0]

    max_len = len(bare_matrix[len(bare_matrix) - 1])
    padded_matrix = np.array([np.pad(row, (0, max_len - len(row)), 'constant') for row in bare_matrix])

    return padded_matrix


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


def tf(text):
    words = text.split()
    tf_base = len(words)

    for i in range(len(words)):
        id = clp(words[i])
        if id:
            words[i] = clp.bform(id[0])
    word_counter = Counter(words)
    t_f = dict(word_counter)

    for key, value in t_f.items():
        t_f[key] = value / tf_base

    return t_f


def idf(matrix):
    idf_result = []
    max_len = len(matrix[len(matrix) - 1])
    corpus_length = len(matrix)

    for vector in range(max_len):
        word_counter = np.count_nonzero(matrix[:, vector])
        word_idf = math.log(corpus_length / word_counter)
        idf_result.append(word_idf)

    idf_vector = np.array(idf_result)

    return idf_vector


def tfidf(matrix, idf_vector):
    tfidf_result = matrix * idf_vector

    return tfidf_result


def pack_marshal(text_directory, text_filename, path_to_marshal, cached_filename):
    try:
        with open(os.path.join(text_directory, text_filename), 'r', encoding='utf-8') as file:
            text = file.read()
            text = re.sub(r'[\d\W_]', ' ', text)
            t_f = tf(text)
    except FileNotFoundError:
        print('Plik nie został znaleziony')
        return False

    with open(os.path.join(path_to_marshal, cached_filename), 'wb') as cached_file:
        cached_data = marshal.dumps(t_f)
        cached_file.write(cached_data)

    return t_f


def unpack_marshal(path_to_marshal, cached_filename):
    try:
        with open(os.path.join(path_to_marshal, cached_filename), 'rb') as cached_file:
            cached_data = cached_file.read()
            t_f = marshal.loads(cached_data)
    except (FileNotFoundError, EOFError, ValueError):
        print('     wystąpił problem')
        return False

    return t_f


def get_result(words, matrix, text_names):
    words = np.array(words)
    result = []
    top_10_words = np.argsort(matrix, axis=1)[:, -10:]
    top_10_results = np.sort(matrix, axis=1)[:, -10:]

    for text in range(len(text_names)):
        row = [text_names[text]]

        for i in range(len(top_10_words[0])):
            row.append((words[top_10_words[text][i]], round(top_10_results[text][i], 3)))
        row = [row[0]] + row[1:][::-1]
        result.append(row)
    result = tuple(result)

    # print(result)
    return result


def run(directory):
    corpus_matrix = []
    filenames = []
    path_to_marshal_files = "./cache"

    if not os.path.exists(path_to_marshal_files):
        os.makedirs(path_to_marshal_files)

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):

            f_name = filename[:-4]

            cached_filename = f_name + '_cached.marshal'
            path_to_file = os.path.join(path_to_marshal_files, cached_filename)

            if os.path.exists(path_to_file):
                t_f = unpack_marshal(path_to_marshal_files, cached_filename)
                if not t_f:
                    return
            else:
                t_f = pack_marshal(directory, filename, path_to_marshal_files, cached_filename)
                if not t_f:
                    return

            filenames.append(filename)
            if corpus_matrix:
                corpus_matrix = add_row(filename, t_f, corpus_matrix)
            else:
                initial_row = [''] + list(t_f.keys())
                row = [filename] + list(t_f.values())
                corpus_matrix.append(initial_row)
                corpus_matrix.append(row)

    texts = tuple(filenames)
    word_list = corpus_matrix[0].copy()
    del word_list[0]

    p_matrix = pad_matrix(corpus_matrix)
    idf_result = idf(p_matrix)
    tfidf_result = tfidf(p_matrix, idf_result)
    results = get_result(word_list, tfidf_result, texts)

    for result in results:
        print(result)

    return results


run(dir_1)
