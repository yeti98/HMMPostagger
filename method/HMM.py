import sys

import numpy as np

from method import utils
from gensim.corpora import Dictionary


class HMMTagger:
    hmm_obj = None
    freq_obj = None
    tag_set = None
    tag2idx = None
    idx2tag = None

    def __init__(self, tagset):
        tagset = tagset + ['START']
        HMMTagger.tag_set = tagset
        HMMTagger.tag2idx = {tagset[i]: i for i in range(len(tagset))}
        HMMTagger.idx2tag = {i: tagset[i] for i in range(len(tagset))}

        pass

    def fit(self, corpus):
        # TODO: smoothing
        clean_corpus = utils.clean_corpus(utils.load_train(corpus))
        A, B = utils.corpus_analyzer(clean_corpus, tagset=HMMTagger.tag_set)
        self.freq_obj = Freq(A, B)
        self.hmm_obj = HMM()
        # print('INITIAL_DIST:\n', self.freq_obj.get_initial_distribution())
        # print('TRANSITION_MATRIX:\n', self.freq_obj.get_transition_matrix())

    def postagging(self, word_seq):
        N = self.freq_obj.get_hidden_stage()
        T = len(word_seq)
        word_seq = [_.lower() for _ in word_seq]
        dct = Dictionary([word_seq])
        word2idx = dct.token2id
        id2word = {v: k for k, v in word2idx.items()}

        observed = np.array(dct.doc2idx(word_seq))
        emission_prob_matrix = np.zeros((N, T))
        for word in word2idx.keys():
            for tag in self.tag_set:
                if tag != 'START':
                    emission_prob_matrix[HMMTagger.get_tagid(tag), word2idx[word]] = self.freq_obj.get_emission_prob(
                        word.lower(), tag)

        path = self.hmm_obj.viterbi(observed, self.freq_obj.get_transition_matrix(), emission_prob_matrix,
                                    self.freq_obj.get_initial_distribution())
        return [(id2word[observed[i]], HMMTagger.idx2tag[path[0][i]]) for i in range(len(observed))]

    @staticmethod
    def get_tagid(postag):
        return HMMTagger.tag2idx[postag]

    def eval(self, test):
        test_set = utils.clean_corpus(utils.load_train(test))
        token_accuracy = 0.0
        sent_accuracy = 0.0
        total_tag = 0
        total_sent = len(test_set)
        size = len(HMMTagger.tag_set) - 1
        cf_matrix = np.zeros((size, size))
        for sent in test_set:
            word_seq = [tp[0] for tp in sent]
            total_tag += len(word_seq)
            tag_true = [tp[1] for tp in sent]
            tag_pred = [t[1] for t in self.postagging(word_seq)]
            # print(tag_pred)
            flag = True
            for i in range(len(word_seq)):
                if tag_pred[i] == tag_true[i]:
                    token_accuracy += 1
                else:
                    flag = False
                id_true = HMMTagger.tag2idx[tag_true[i]]
                id_pred = HMMTagger.tag2idx[tag_pred[i]]
                cf_matrix[id_true, id_pred] += 1.0
            if flag:
                sent_accuracy += 1.0
            else:
                print()
                for _ in range(len(word_seq)):
                    print('\t'.join([word_seq[_], tag_pred[_], tag_true[_]]))

        token_accuracy /= total_tag
        sent_accuracy /= total_sent
        return token_accuracy, sent_accuracy, cf_matrix


class HMM:

    def __init__(self):
        pass

    def viterbi(self, observed, a, b, initial_distribution):
        '''
        :param observed:
        :param a: transition prob matrix
        :param b: emission prob matrix
        :param initial_distribution:
        :return:
        '''
        N = b.shape[0]
        T = observed.shape[0]
        # print(T)
        viterbi_matrix = np.zeros((N, T))
        back_pointer = np.zeros((N, T))
        # initialization step
        # using log scale trick if you don't want to get underflow errors
        viterbi_matrix[:, 0] = np.log(initial_distribution * b[:, observed[0]])

        for t in range(1, T):
            for stage in range(N):
                probs = viterbi_matrix[:, t - 1] + np.log(a[:, stage]) + np.log(b[stage, observed[t]])
                viterbi_matrix[stage, t] = np.max(probs)
                back_pointer[stage, t] = np.argmax(probs)

        best_path = np.max(viterbi_matrix[:, T - 1])
        best_path_pointer = np.argmax(viterbi_matrix[:, T - 1])
        # print(viterbi_matrix)
        # print(back_pointer)
        # print(best_path)

        S = np.zeros(T)
        S[0] = best_path_pointer
        i = 1
        while i < T:
            S[i] = back_pointer[int(S[i - 1]), T - i]
            i += 1
        S = np.flip(S, axis=0)
        return S, best_path


class Freq:
    '''
    '''

    def __init__(self, A, B):
        '''
        :param A: Transition matrix
        :param B: Emission matrix
        '''
        self.A = A
        self.B = B
        self.tag_counter = np.sum(self.A, axis=1)
        start_idx = HMMTagger.get_tagid('START')
        tmp = np.delete(self.A, start_idx, axis=1)
        tmp = tmp / np.sum(tmp, axis=1).reshape(tmp.shape[0], 1)
        self.initial_distribution = tmp[start_idx]
        self.transition_prob_matrix = np.delete(tmp, start_idx, axis=0)
        self.N = self.transition_prob_matrix.shape[0]

    def get_transition_prob(self, tag1, tag2):
        '''
        P(ti|ti−1) = =C(ti−1,ti) / C(ti−1)
        :param tag1: ti
        :param tag2: ti-1
        :return: P(tag1 | tag2)
        '''
        idx1 = HMMTagger.get_tagid(tag1)
        idx2 = HMMTagger.get_tagid(tag2)
        return self.A[idx2, idx1] / self.tag_counter[idx2]

    def get_emission_prob(self, word, tag):
        '''
        P(wi|ti) =C(ti,wi)/C(ti)
        :param word:
        :param tag:
        :return: P(word | tag)
        '''
        idx = HMMTagger.get_tagid(tag)
        if word not in self.B[tag]:
            # TODO: Smooth
            return 1 / self.tag_counter[idx]
        # print('{} occured {} time in category {}'.format(word, self.B[tag][word], self.tag_counter[idx]))
        return self.B[tag][word] / self.tag_counter[idx]

    def get_initial_distribution(self):
        return self.initial_distribution

    def get_transition_matrix(self):
        return self.transition_prob_matrix

    def get_hidden_stage(self):
        return self.N


