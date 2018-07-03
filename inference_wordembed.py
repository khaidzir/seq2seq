import torch

from train_bidirectional import Trainer
from model2 import PreTrainedEmbeddingEncoderRNN, AttnDecoderRNN
from preprocess import prepareData, unicodeToAscii, normalizeString
from gensim.models import KeyedVectors
from lang import Lang
import params
from gensim.models import KeyedVectors

def preprocessSentence(sentence, max_len) :
    sentence = normalizeString(unicodeToAscii(sentence))
    split = sentence.split()
    if len(split) >= max_len :
        split = split[:max_len-1]
    return ' '.join(split)

# Config variables
encoder_file = 'model/chatbot/augmented_data/word2vec/skipgram/twitter_sgram/encoder-d100-e5.pt'
decoder_file = 'model/chatbot/augmented_data/word2vec/skipgram/twitter_sgram/decoder-d100-e5.pt'
encoder_attr_dict = torch.load(encoder_file)
decoder_attr_dict = torch.load(decoder_file)

# Dataset (for build dictionary)
# src_lang, tgt_lang, pairs = prepareData('dataset/input-output.txt', reverse=False)

# Lang
decoder_lang = Lang()
decoder_lang.load_dict(decoder_attr_dict['lang'])

# Word vector
# word_vector = KeyedVectors.load_word2vec_format("word_vector/koran.vec", binary=True)
word_vectors = KeyedVectors.load(params.WORD_VECTORS_FILE)

# Params
use_cuda = params.USE_CUDA
hidden_size = word_vectors.vector_size

# Encoder & Decoder
# encoder = EncoderEmbeddingRNN(src_lang.n_words, hidden_size, word_vector)
# attn_decoder = AttnDecoderRNN(hidden_size, tgt_lang.n_words, dropout_p=0.1, max_length=max_len)
# encoder.loadState(ENCODER_MODEL)
# attn_decoder.loadState(DECODER_MODEL)
encoder = PreTrainedEmbeddingEncoderRNN(word_vectors, encoder_attr_dict['max_length'])
encoder.loadAttributes(encoder_attr_dict)
attn_decoder = AttnDecoderRNN(decoder_attr_dict['hidden_size'], decoder_lang, max_length=decoder_attr_dict['max_length'])
encoder.loadAttributes(encoder_attr_dict)
attn_decoder.loadAttributes(decoder_attr_dict)

if use_cuda:
    encoder = encoder.cuda()
    attn_decoder = attn_decoder.cuda()

trainer = Trainer([], encoder, attn_decoder)

sentence = input("Input : ")
while (sentence != "<end>") :
    sentence = preprocessSentence(sentence, attn_decoder.max_length)
    output_words, attentions = trainer.evaluate(sentence)
    output = ' '.join(output_words[:-1])
    print(output)
    sentence = input("Input : ")

# file_test = "test/test.txt"
# results = []
# with open(file_test, "r", encoding="utf-8") as f :
#     for line in f :
#         line = line.strip()
#         output_words, attentions = trainer.evaluate(encoder, attn_decoder, line, max_len=max_len)
#         output = ' '.join(output_words[:-1])
#         results.append(output)

# file_out = "test/result.txt"
# fout = open(file_out, "w", encoding="utf-8")
# for result in results :
#     fout.write("%s\n"%(result))
# fout.close()