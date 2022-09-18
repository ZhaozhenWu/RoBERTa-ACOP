# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
import re
import tqdm

nlp = spacy.load('en_core_web_sm')


def tokenize_spacy(text):
    text = text.strip()
    text = re.sub(r' {2,}', ' ', text)
    document = nlp(text)
    return [token.text for token in document]


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    text = text.strip()
    text = re.sub(r' {2,}', ' ', text)
    tokens = nlp(text)
    matrix = np.zeros((len(tokens), len(tokens))).astype('float32')

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename + '.graph', 'wb')
    for i in range(0, len(lines), 4):
        text = lines[i].lower().strip()
        adj_matrix = dependency_adj_matrix(text)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()


if __name__ == '__main__':
    # process('./datasets/acl-14-short-data/train.raw.order')
    # process('./datasets/acl-14-short-data/test.raw.order')
    process('./datasets/semeval14/restaurant_train.raw.order')
    process('./datasets/semeval14/restaurant_test.raw.order')
    process('./datasets/semeval14/laptop_train.raw.order')
    process('./datasets/semeval14/laptop_test.raw.order')
    process('./datasets/semeval15/restaurant_train.raw.order')
    process('./datasets/semeval15/restaurant_test.raw.order')
    process('./datasets/semeval16/restaurant_train.raw.order')
    process('./datasets/semeval16/restaurant_test.raw.order')
