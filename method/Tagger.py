import copy

import numpy as np

TRAIN = '/home/ddragon/Desktop/CoreNLP/data/vlsp2016/corpus/train.txt'
TAGSET = ['N', 'V', 'CH', 'R', 'E', 'A', 'P', 'Np', 'M', 'C', 'Nc', 'L', 'T', 'Ny', 'Nu', 'X', 'B', 'S', 'I', 'Y', 'Vy',
          'FW', 'Z', 'START', 'END']


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


def tag_transition_matrix(corpus):
    A = np.zeros((len(TAGSET), len(TAGSET)))
    for sent in corpus:
        tmp = [('', 'START')] + sent + [('', 'END')]
        for i in range(len(tmp)-1):
            print(tmp[i+1][1])
            pos_index_i = TAGSET.index(tmp[i][1])
            pos_index_j = TAGSET.index(tmp[i + 1][1])
            A[pos_index_i, pos_index_j] += 1.0
    return A


def word_tag_emission_matrix(corpus):
    pass


corpus = load_train(TRAIN)
res = clean_corpus(corpus)
A = tag_transition_matrix(res)
print(A)
