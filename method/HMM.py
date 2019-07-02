import numpy as np

from method import utils


class HMMTagger:
    hmm_obj = None
    freq_obj = None
    tag_set = None

    def __init__(self):
        pass

    def fit(self, corpus):
        # TODO: smoothing
        A, B = utils.get_hmm_components(corpus)
        self.freq_obj = Freq(A, B, self.tag_set)
        pass

    def postag(self, word_seq):
        transition_maxtrix, emission_matrix = self.freq_obj.create_matrices(word_seq)


class HMM:
    def __init__(self, hidden_stage=3):
        self.NSTAGE = hidden_stage
        pass

    def forward(self, V, a, b, initial_distribution):
        alpha = np.zeros((V.shape[0], a.shape[0]))
        alpha[0, :] = initial_distribution * b[:, V[0]]

        for t in range(1, V.shape[0]):
            for j in range(a.shape[0]):
                # Matrix Computation Steps
                #                  ((1x2) . (1x2))      *     (1)
                #                        (1)            *     (1)
                alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]

        return alpha

    def backward(self, V, a, b):
        beta = np.zeros((V.shape[0], a.shape[0]))

        # setting beta(T) = 1
        beta[V.shape[0] - 1] = np.ones((a.shape[0]))

        # Loop in backward way from T-1 to
        # Due to python indexing the actual loop will be T-2 to 0
        for t in range(V.shape[0] - 2, -1, -1):
            for j in range(a.shape[0]):
                beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])

        return beta

    def baum_welch(self, V, a, b, initial_distribution, n_iter=100):
        M = a.shape[0]
        T = len(V)

        for n in range(n_iter):
            alpha = self.forward(V, a, b, initial_distribution)
            beta = self.backward(V, a, b)

            xi = np.zeros((M, M, T - 1))
            for t in range(T - 1):
                denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
                for i in range(M):
                    numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                    xi[i, :, t] = numerator / denominator

            gamma = np.sum(xi, axis=1)
            a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

            # Add additional T'th element in gamma
            gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

            K = b.shape[1]
            denominator = np.sum(gamma, axis=1)
            for l in range(K):
                b[:, l] = np.sum(gamma[:, V == l], axis=1)

            b = np.divide(b, denominator.reshape((-1, 1)))

        return a, b

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


class Freq:
    '''
    '''
    TAGSET = None

    def __init__(self, A, B, tagset):
        '''
        :param A: Transition matrix
        :param B: Emission matrix
        '''
        self.A = A
        self.B = B
        self.TAGSET = tagset

    def get_transition_prob(self, tag1, tag2):
        '''
        P(ti|ti−1) = =C(ti−1,ti) / C(ti−1)
        :param tag1: ti
        :param tag2: ti-1
        :return: P(tag1 | tag2)
        '''
        assert tag1 in self.TAGSET
        assert tag2 in self.TAGSET
        return self.A[self.TAGSET.index(tag2), self.TAGSET.index(tag1)] / self.count_tag(tag2)

    def count_tag(self, postag):
        '''
        :param postag:
        :return: number postag occurrence in corpus
        '''
        return sum(self.B[postag].values())

    def get_emission_prob(self, word, tag):
        '''
        P(wi|ti) =C(ti,wi)/C(ti)
        :param word:
        :param tag:
        :return: P(word | tag)
        '''
        assert tag in self.TAGSET
        return self.B[tag][word] / self.count_tag(tag)

    def get_initial_distribution(self):
        start_row_index = self.TAGSET.index('START')
        initial_distribution = self.A[start_row_index, :] / self.A.sum(start_row_index)

        return initial_distribution

    def create_matrices(self, observed):
        N = self.A.shape[0]
        T = len(observed)
        emission_prob_matrix = np.zeros((N, T))

        row_sums = np.sum(self.A, axis=1).reshape(N, 1)
        transition_prob_matrix = self.A / row_sums

        emission_prob_matrix

        return transition_prob_matrix, emission_prob_matrix

