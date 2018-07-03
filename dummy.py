import torch
import torch.nn as nn
from preprocess import prepareData
from torch.autograd import Variable
from train_bidirectional import Trainer
import numpy as np
from lang import Lang
from model_bidirectional import WordEncoderBiRNN,PreTrainedEmbeddingEncoderBiRNN,AttnDecoderRNN
from gensim.models import KeyedVectors
import params

# train_file = '/home/prosa/Works/Text/korpus/chatbot_dataset/plain/preprocessed/split-augmented/combine/nontask/train.test'
# src_lang, tgt_lang, pairs = prepareData(train_file, reverse=False)

# Word vector
# word_vector = KeyedVectors.load_word2vec_format("word_vector/koran.vec", binary=True)
word_vectors = KeyedVectors.load(params.WORD_VECTORS_FILE)

encoder_file = 'encoder_dummy.pt'
decoder_file = 'decoder_dummy.pt'
encoder_dict = torch.load(encoder_file)
decoder_dict = torch.load(decoder_file)

decoder_lang = Lang()
decoder_lang.load_dict(decoder_dict['lang'])

hidden_size = word_vectors.vector_size
max_length = 32
# encoder = WordEncoderBiRNN(hidden_size, max_length, src_lang)
encoder = PreTrainedEmbeddingEncoderBiRNN(word_vectors, max_length, char_embed=True, seeder=params.SEEDER)
attn_decoder = AttnDecoderRNN(2*hidden_size, max_length, decoder_lang, seeder=params.SEEDER)

# Load and continue train
encoder.loadAttributes(encoder_dict)
attn_decoder.loadAttributes(decoder_dict)

trainer = Trainer([], encoder, attn_decoder)
sentences = [
    'saya muahdg makan',
    'halloooooo buooott awkwkwk',
    'zehahahaha nuasf bisa geloooo'
]
i = 1
for sent in sentences :
    decoded_words,_ = trainer.evaluate(sent)
    print(str(i) + ' : ' + ' '.join(decoded_words))
    i += 1

# num_iter = 9769
# num_iter = 10
# epoch=5
# trainer = Trainer(pairs, encoder, attn_decoder)
# trainer.train(num_iter, print_every=num_iter//100, epoch=epoch)
# trainer.train(num_iter, 1, epoch=epoch)

# torch.save(encoder.getAttrDict(), 'encoder_dummy.pt')
# torch.save(attn_decoder.getAttrDict(), 'decoder_dummy.pt')
