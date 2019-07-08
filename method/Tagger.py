from method import utils
from method.HMM import HMMTagger

TRAIN = '/home/ddragon/Desktop/CoreNLP/data/vlsp2016/corpus/train.txt'
hmm_obj = HMMTagger(tagset=['A', 'P', 'Np', 'M', 'C', 'Nc', 'L', 'T', 'Ny', 'Nu', 'X', 'I', 'FW', 'Z',
                            'Vy', 'N', 'V', 'CH', 'R', 'E'])
hmm_obj.fit(TRAIN)

# TAGGING SEQUENCE
# sent = ['Chị', 'Minh', 'ôm', 'đứa', 'con_gái', 'mới', 'hơn', 'hai',
#         'tháng', 'rưỡi', 'tuổi', 'nấc', 'lên', 'từng', 'tiếng', 'thảm_thiết',
#         'khi', 'kể', 'lại', 'cho', 'chúng_tôi', 'nghe',
#         'về', 'cái', 'chết', 'của', 'chồng', '.']
# hmm_obj.postagging(sent)

# EVALUATION
pertag_accur, sent_accur, cf_matrix = hmm_obj.eval('/home/ddragon/Desktop/CoreNLP/data/vlsp2016/corpus/test.txt')
print('PERTAG ACCURACY: {}'.format(pertag_accur))
print('SENT_ACCURACY: {}'.format(sent_accur))
utils.confusion_matrix_visualize(cf_matrix, ['A', 'P', 'Np', 'M', 'C', 'Nc', 'L', 'T', 'Ny', 'Nu', 'X', 'I', 'FW', 'Z',
                                             'Vy', 'N', 'V', 'CH', 'R', 'E'])
