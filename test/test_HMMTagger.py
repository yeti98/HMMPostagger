import unittest

import numpy as np

import settings
from method import utils
from method.HMM import HMMTagger, Freq


class test_HMMTagger(unittest.TestCase):
    def test_freq(self):
        cor = utils.load_train(settings.TEST_DATA + '/test_corpus')
        clean_corpus = utils.clean_corpus(cor)
        A, B = utils.corpus_analyzer(clean_corpus, tagset=['A', 'B', 'START'])
        hmm_obj = HMMTagger(['A', 'B'])
        freq_obj = Freq(A, B)
        initial_dis = freq_obj.get_initial_distribution()
        transition_mt = freq_obj.get_transition_matrix()
        assert np.all(initial_dis == [.6, .4]) == True
        assert np.all(np.round(transition_mt, 4) == [[0.4286, 0.5714], [0.5, 0.5]]) == True
        assert freq_obj.get_hidden_stage() == 2


if __name__ == '__main__':
    unittest.main()
