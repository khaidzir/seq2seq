import torch

from model_bidirectional_v1_lstm import WordEncoderBiRNN, PreTrainedEmbeddingEncoderBiRNN, PreTrainedEmbeddingWordCharEncoderBiRNN, AttnDecoderRNN
from preprocess import prepareData
from train_bidirectional import Trainer
from preprocess import buildPairs
from gensim.models import KeyedVectors
import params
from util import load_wordvector_text
from lang import Lang

# train_file = '/home/prosa/Works/Text/korpus/chatbot_dataset/plain/preprocessed/split-augmented/combine/nontask/train-nontask.aug.shuffle.pre'
# train_file = '/home/prosa/Works/Text/korpus/chatbot_dataset/plain/preprocessed/split-augmented/combine/nontask/train.test'
# src_lang, tgt_lang, pairs = prepareData('dataset/chatbot/input-output.txt', reverse=False)

# train_file = '/home/prosa/Works/Text/mt/dataset/filter-en-id/lenlim80/sorted/train.dummy'
# train_file = '/home/prosa/Works/Text/mt/dataset/filter-en-id/lenlim80/sorted/limit-en-id.sorted.01.txt'
# train_file = '/home/prosa/Works/Text/korpus/asr_dataset/dataset_pruned/word/dummy'
# train_file = '/home/prosa/Works/Text/korpus/dialogue/dataset_filtered/gabung.shuffle'
# train_file = '/home/prosa/Works/Text/korpus/dialogue/misc.txt'

train_file = '/home/prosa/Works/Text/seq2seq/dataset/en-id-10k-v2.txt'

src_lang, tgt_lang, pairs = prepareData(train_file, reverse=False)

# Word vector
# word_vectors = KeyedVectors.load_word2vec_format(params.WORD_VECTORS_FILE, binary=True)
# word_vectors = KeyedVectors.load(params.WORD_VECTORS_FILE)
word_vectors = load_wordvector_text(params.WORD_VECTORS_FILE)


############
# folder_model = 'model/dialogue/fix/oovchar_rnn/'

# folder_model = 'model/dialogue/dummy/wordchar_cnn/'
# folder_model = 'model/dialogue/dummy/oovchar_rnn/'
folder_model = 'model/dialogue/dummy/wordchar_rnn/'
# folder_model = 'model/dialogue/dummy/word/'

# folder_model = 'model/dialogue/fix/oovchar_rnn/'
# folder_model = 'model/mt/tesis/oovchar_rnn/'

'''
# Load and continue train
encoder_file = folder_model + 'encoder-e50.pt'
decoder_file = folder_model + 'decoder-e50.pt'
encoder_dict = torch.load(encoder_file)
decoder_dict = torch.load(decoder_file)
decoder_lang = Lang()
decoder_lang.load_dict(decoder_dict['lang'])
encoder = PreTrainedEmbeddingEncoderBiRNN(word_vectors, encoder_dict['hidden_size'], encoder_dict['max_length'], char_embed=encoder_dict['char_embed'], seeder=params.SEEDER)
# encoder = PreTrainedEmbeddingWordCharEncoderBiRNN(word_vectors, encoder_dict['input_size'], encoder_dict['max_length'], char_feature='cnn', seeder=params.SEEDER)
attn_decoder = AttnDecoderRNN(decoder_dict['input_size'], decoder_dict['hidden_size'], decoder_dict['max_length'], decoder_lang, seeder=params.SEEDER)
encoder.loadAttributes(encoder_dict)
attn_decoder.loadAttributes(decoder_dict)

'''

# New model
input_size = word_vectors.vector_size
hidden_size = 256
max_length = 50
dropout_p = 0.0
char_feature = 'rnn'
# encoder = WordEncoderBiRNN(hidden_size, max_length, src_lang, seeder=params.SEEDER)
encoder = PreTrainedEmbeddingEncoderBiRNN(word_vectors, hidden_size, max_length, char_embed=False, dropout_p=dropout_p, seeder=params.SEEDER)
# encoder = PreTrainedEmbeddingWordCharEncoderBiRNN(word_vectors, hidden_size, max_length, char_feature=char_feature, dropout_p=dropout_p, seeder=params.SEEDER)
attn_decoder = AttnDecoderRNN(input_size, hidden_size*2, max_length, tgt_lang, dropout_p=dropout_p, seeder=params.SEEDER)


folder_model_2 = folder_model
num_iter = len(pairs)
epoch = 50
lr = 0.001
batch_size = 4
save_every = 5
trainer = Trainer(pairs, encoder, attn_decoder)
trainer.train_batch(learning_rate=lr, print_every=17, epoch=epoch, batch_size=batch_size, save_every=save_every, folder_model=folder_model_2)
trainer.evaluateRandomly(n=100)
# trainer.evaluateTrainSet()

# torch.save(encoder.getAttrDict(), folder_model + 'encoder-final-d' + str(hidden_size) + '-e' + str(epoch) + '.pt')
# torch.save(attn_decoder.getAttrDict(), folder_model + 'decoder-final-d' + str(hidden_size) + '-e' + str(epoch) + '.pt')
#################
