import copy
from collections import defaultdict

import numpy as np

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

TRAIN = '/home/ddragon/Desktop/CoreNLP/data/vlsp2016/corpus/train.txt'
TAGSET = ['START', 'N', 'V', 'CH', 'R', 'E', 'A', 'P', 'Np', 'M', 'C', 'Nc', 'L', 'T', 'Ny', 'Nu', 'X', 'B', 'S', 'I',
          'Y', 'Vy', 'FW', 'Z']



def load_train(corpus_path):
    with open(corpus_path, 'r') as handle:
        train = handle.read()
    handle.close()
    return train


def clean_corpus(corpus):
    '''

    :param corpus:
    :return: list of sentences. Each element in sentence is a pair of word and postag
    '''
    cur_sen = []
    res = []
    for line in corpus.split('\n'):
        if len(line) == 0:
            res.append(copy.deepcopy(cur_sen))
            cur_sen.clear()
        else:
            arr = line.split('\t')
            word, postag = arr[0].replace(' ', '_'), arr[1]
            cur_sen.append((word, postag))
    return res



def get_hmm_components(corpus):
    A = np.zeros((len(TAGSET), len(TAGSET)))
    B = defaultdict(lambda: defaultdict(int))
    for sent in corpus:
        tmp = [('', 'START')] + sent
        for i in range(len(tmp) - 1):
            pos_index_i = TAGSET.index(tmp[i][1])
            pos_index_j = TAGSET.index(tmp[i + 1][1])
            A[pos_index_i, pos_index_j] += 1.0
            B[tmp[i + 1][1]][tmp[i + 1][0].lower()] += 1
    return A, B


def ar2df(array, labels):
    df_cm = pd.DataFrame(array, index=[i for i in labels], columns=[i for i in labels])
    return df_cm


def confusion_matrix_visualize(array, lables):
    df_cm = ar2df(array, lables)
    plt.figure(figsize=(10, 7))
    plt.tight_layout()
    sn.heatmap(df_cm, annot=True, cmap=plt.get_cmap('Blues'))
    plt.show()


def word_tag_emission_matrix(corpus):
    pass


