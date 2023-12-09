import os
import re
import marshal
import math
import locale
import numpy as np
from collections import Counter
from clp3 import clp
locale.setlocale(locale.LC_COLLATE, 'pl_PL.UTF-8')


dir_1 = './teksty-o-świnkach-morskich'


def split_text_to_sentences(text):
    sentences = []
    current_sentence = ''

    for character in text:
        current_sentence += character

        if character in ['.', '?', '!']:
            sentences.append((re.sub(r'[\d\W_]+', ' ', current_sentence.lower().strip())))
            current_sentence = ''

    if current_sentence:
        sentences.append((re.sub(r'[\d\W_]+', ' ', current_sentence.lower().strip())))

    return tuple(sentences)


def concordance(word_forms, sentence_list):
    result = []
    not_empty = False
    for sentence in sentence_list:
        words_in_sentence = sentence.split(' ')
        for word in word_forms:
            if word in words_in_sentence:
                word_index = words_in_sentence.index(word)
                strip_start = max(0, word_index - 4)
                strip_end = min(len(words_in_sentence), word_index + 5)

                strip = words_in_sentence[strip_start:strip_end]
                result.append((sentence_list.index(sentence), ' '.join(strip)))

    if not result:
        return not_empty
    else:
        return result


def print_result(word, filenames, results):
    x = ', '.join(filenames)
    print(f'\nSłowo {word} występuje w następujących plikach:\n {x}\n')
    for result in results:
        for strip in result[1]:
            print(f'{result[0]}: \nfragment zdania: {strip[1]} \nnumer zdania: {strip[0]} \n\n')


def run(directory):
    word = input('Wpisz szukane słowo: ')
    word_forms = clp.forms_all(word)
    filenames = set()
    result = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                processed_text = split_text_to_sentences(text)
                answer = concordance(word_forms, processed_text)
        if answer:
            result.append((filename, answer))
            filenames.add(filename)

    print_result(word, filenames, result)


run(dir_1)
