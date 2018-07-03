import torch

from model import EncoderRNN, EncoderEmbeddingInputRNN, AttnDecoderRNN
from preprocess import prepareData
from train_wordembed2 import Trainer
from preprocess import buildPairs
from gensim.models import KeyedVectors

use_cuda = torch.cuda.is_available()

src_lang, tgt_lang, pairs = prepareData('dataset/input-output.txt', reverse=False)

# Word vector
word_vector = KeyedVectors.load_word2vec_format("word_vector/koran.vec", binary=True)

hidden_size = 64
max_len = 50
encoder = EncoderEmbeddingInputRNN(src_lang.n_words, hidden_size, word_vector)
attn_decoder = AttnDecoderRNN(hidden_size, tgt_lang.n_words, dropout_p=0.1, max_length=max_len)

if use_cuda:
    encoder = encoder.cuda()
    attn_decoder = attn_decoder.cuda()

num_iter = 100000
trainer = Trainer(src_lang, tgt_lang, pairs)
trainer.train(encoder, attn_decoder, num_iter, print_every=num_iter//100, max_len=max_len)
trainer.evaluateRandomly(encoder, attn_decoder, max_len=max_len)
# trainer.evaluateAll(encoder, attn_decoder)

encoder.saveState('model/encoder-embedding2-h64' + str(num_iter) + '.pt')
attn_decoder.saveState('model/decoder-embedding2-h64' + str(num_iter) + '.pt')

# Open testfile as test and build pairs from it
# test_pairs = buildPairs("corpus/test-ind-eng.txt")

# Test using test data
# trainer.evaluateFromTest(test_pairs, encoder, attn_decoder, max_len=max_len)

# encoder.saveState('checkpoint/encoder-ind-eng-' + str(num_iter*2) + '.pt')
# attn_decoder.saveState('checkpoint/decoder-ind-eng-' + str(num_iter*2) + '.pt')
