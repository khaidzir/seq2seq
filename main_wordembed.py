import torch

from model2 import PreTrainedEmbeddingEncoderRNN, AttnDecoderRNN
from preprocess import prepareData
from train_bidirectional import Trainer
from preprocess import buildPairs
from gensim.models import KeyedVectors
import params

use_cuda = torch.cuda.is_available()

trainfile = '/home/prosa/Works/Text/mt/dataset/filter-en-id/lenlim80/sorted/train.dummy'
src_lang, tgt_lang, pairs = prepareData(trainfile, reverse=False)

# Word vector
word_vectors = KeyedVectors.load(params.WORD_VECTORS_FILE)

hidden_size = word_vectors.vector_size
max_len = 8
encoder = PreTrainedEmbeddingEncoderRNN(word_vectors, max_len)
attn_decoder = AttnDecoderRNN(hidden_size, tgt_lang, dropout_p=0.1, max_length=max_len)

if use_cuda:
    encoder = encoder.cuda()
    attn_decoder = attn_decoder.cuda()

epoch = 100
num_iter = len(pairs)
trainer = Trainer(pairs, encoder, attn_decoder)
# trainer.train(encoder, attn_decoder, num_iter, print_every=num_iter//10, max_len=max_len, epoch=epoch)
trainer.train(num_iter, print_every=num_iter//10, epoch=epoch)
trainer.evaluateRandomly()
# trainer.evaluateAll(encoder, attn_decoder)

# str_iter = str(num_iter//1000) + 'k'
# torch.save(encoder.getAttrDict(), 'model/chatbot/encoder-uni-d' + str(hidden_size) + '-i' + str_iter + '.pt')
# torch.save(attn_decoder.getAttrDict(), 'model/chatbot/decoder-uni-d' + str(hidden_size) + '-i' + str_iter + '.pt')

torch.save(encoder.getAttrDict(), 'model/mt/dummy/encoder-d' + str(hidden_size) + '-e' + str(epoch) + '.pt')
torch.save(attn_decoder.getAttrDict(), 'model/mt/dummy/decoder-d' + str(hidden_size) + '-e' + str(epoch) + '.pt')

# Open testfile as test and build pairs from it
# test_pairs = buildPairs("corpus/test-ind-eng.txt")

# Test using test data
# trainer.evaluateFromTest(test_pairs, encoder, attn_decoder, max_len=max_len)

# encoder.saveState('checkpoint/encoder-ind-eng-' + str(num_iter*2) + '.pt')
# attn_decoder.saveState('checkpoint/decoder-ind-eng-' + str(num_iter*2) + '.pt')
