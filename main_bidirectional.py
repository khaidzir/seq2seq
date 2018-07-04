import torch

from model_bidirectional_v1_lstm import WordEncoderBiRNN, PreTrainedEmbeddingEncoderBiRNN, AttnDecoderRNN
from preprocess import prepareData
from train_bidirectional import Trainer
from preprocess import buildPairs
from gensim.models import KeyedVectors
import params
from util import load_wordvector_text

# train_file = '/home/prosa/Works/Text/korpus/chatbot_dataset/plain/preprocessed/split-augmented/combine/nontask/train-nontask.aug.shuffle.pre'
# train_file = '/home/prosa/Works/Text/korpus/chatbot_dataset/plain/preprocessed/split-augmented/combine/nontask/train.test'
# src_lang, tgt_lang, pairs = prepareData('dataset/chatbot/input-output.txt', reverse=False)

# train_file = '/home/prosa/Works/Text/mt/dataset/filter-en-id/lenlim80/sorted/train.dummy'
# train_file = '/home/prosa/Works/Text/mt/dataset/filter-en-id/lenlim80/sorted/limit-en-id.sorted.01.txt'
# train_file = '/home/prosa/Works/Text/korpus/asr_dataset/dataset_pruned/word/dummy'
train_file = '/home/prosa/Works/Text/korpus/dialogue/misc.txt'

src_lang, tgt_lang, pairs = prepareData(train_file, reverse=False)

# Word vector
# word_vectors = KeyedVectors.load_word2vec_format(params.WORD_VECTORS_FILE, binary=True)
# word_vectors = KeyedVectors.load(params.WORD_VECTORS_FILE)
word_vectors = load_wordvector_text(params.WORD_VECTORS_FILE)

'''
### CHATBOT ###

hidden_size = word_vectors.vector_size
max_length = 32
# encoder = WordEncoderBiRNN(hidden_size, max_length, src_lang)
encoder = PreTrainedEmbeddingEncoderBiRNN(word_vectors, max_length, char_embed=True, seeder=params.SEEDER)
attn_decoder = AttnDecoderRNN(hidden_size, max_length, tgt_lang, seeder=params.SEEDER)

# Load and continue train
# encoder_file = 'model/mt/encoder-word-en-id-d512-i10k.pt'
# decoder_file = 'model/mt/decoder-word-en-id-d512-i10k.pt'
# encoder_dict = torch.load(encoder_file)
# decoder_dict = torch.load(decoder_file)
# encoder.loadAttributes(encoder_dict)
# attn_decoder.loadAttributes(decoder_dict)

num_iter = len(pairs)
# num_iter = 10
epoch=5
trainer = Trainer(pairs, encoder, attn_decoder)
trainer.train(num_iter, print_every=num_iter//100, epoch=epoch)
# trainer.train(num_iter, print_every=1, epoch=epoch)
trainer.evaluateRandomly()

# str_iter = str(num_iter//1000) + 'k'
torch.save(encoder.getAttrDict(), 'model/chatbot/augmented_data/word2vec/cbow/codot_cbow/charembed_encoder-d' + str(hidden_size) + '-e' + str(epoch) + '-v2.pt')
torch.save(attn_decoder.getAttrDict(), 'model/chatbot/augmented_data/word2vec/cbow/codot_cbow/charembed_decoder-d' + str(hidden_size) + '-e' + str(epoch) + '-v2.pt')
# torch.save(encoder.getAttrDict(), 'model/chatbot/encoder-d' + str(hidden_size) + '-i' + str_iter + '.pt')
# torch.save(attn_decoder.getAttrDict(), 'model/chatbot/decoder-d' + str(hidden_size) + '-i' + str_iter + '.pt')
#################
'''

'''
### MT ###

hidden_size = word_vectors.vector_size
max_length = 80
# encoder = WordEncoderBiRNN(hidden_size, max_length, src_lang)
encoder = PreTrainedEmbeddingEncoderBiRNN(word_vectors, max_length, char_embed=False, seeder=params.SEEDER)
attn_decoder = AttnDecoderRNN(hidden_size, max_length, tgt_lang, seeder=params.SEEDER)

# Load and continue train
# encoder_file = 'model/mt/encoder-word-en-id-d512-i10k.pt'
# decoder_file = 'model/mt/decoder-word-en-id-d512-i10k.pt'
# encoder_dict = torch.load(encoder_file)
# decoder_dict = torch.load(decoder_file)
# encoder.loadAttributes(encoder_dict)
# attn_decoder.loadAttributes(decoder_dict)

num_iter = len(pairs)
epoch=1
trainer = Trainer(pairs, encoder, attn_decoder)
trainer.train(num_iter, print_every=num_iter//1000, epoch=epoch)
# trainer.train(num_iter, print_every=1, epoch=epoch)
trainer.evaluateRandomly()

torch.save(encoder.getAttrDict(), 'model/mt/1m/encoder-d' + str(hidden_size) + '-e' + str(epoch) + '.pt')
torch.save(attn_decoder.getAttrDict(), 'model/mt/1m/decoder-d' + str(hidden_size) + '-e' + str(epoch) + '.pt')
#################
'''


### ASR Correction ###

hidden_size = word_vectors.vector_size
# hidden_size = 128
max_length = 80
# encoder = WordEncoderBiRNN(hidden_size, max_length, src_lang, seeder=params.SEEDER)
encoder = PreTrainedEmbeddingEncoderBiRNN(word_vectors, max_length, char_embed=True, seeder=params.SEEDER)
attn_decoder = AttnDecoderRNN(hidden_size*2, max_length, tgt_lang, seeder=params.SEEDER)

# Load and continue train
# encoder_file = 'model/mt/encoder-word-en-id-d512-i10k.pt'
# decoder_file = 'model/mt/decoder-word-en-id-d512-i10k.pt'
# encoder_dict = torch.load(encoder_file)
# decoder_dict = torch.load(decoder_file)
# encoder.loadAttributes(encoder_dict)
# attn_decoder.loadAttributes(decoder_dict)

num_iter = len(pairs)
epoch=100
batch_size = 32
trainer = Trainer(pairs, encoder, attn_decoder)
# trainer.train(num_iter, print_every=num_iter//1000, epoch=epoch)
trainer.train_batch(print_every=32, epoch=epoch, batch_size=batch_size)
# trainer.evaluateRandomly()
trainer.evaluateTrainSet()

torch.save(encoder.getAttrDict(), 'model/dialogue/encoder-charembed-d' + str(hidden_size) + '-e' + str(epoch) + '.pt')
torch.save(attn_decoder.getAttrDict(), 'model/dialogue/decoder-charembed-d' + str(hidden_size) + '-e' + str(epoch) + '.pt')
#################
