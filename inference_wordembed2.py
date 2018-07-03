import torch

from train_wordembed import Trainer
from model import EncoderRNN, EncoderEmbeddingInputRNN, AttnDecoderRNN
from preprocess import prepareData, unicodeToAscii, normalizeString
from gensim.models import KeyedVectors
import configparser

def preprocessSentence(sentence, max_len) :
    sentence = normalizeString(unicodeToAscii(sentence))
    split = sentence.split()
    if len(split) >= max_len :
        split = split[:max_len-1]
    return ' '.join(split)

# Config variables
config = configparser.ConfigParser()
config.read('config.ini')
DATASET = config['DATA']['Dataset']
ENCODER_MODEL = config['MODELS']['EncoderModel']
DECODER_MODEL = config['MODELS']['DecoderModel']
WORD_VECTOR = config['MODELS']['WordVector']
MAX_LEN = int(config['PARAMS']['MaxLength'])

# Dataset (for build dictionary)
src_lang, tgt_lang, pairs = prepareData('dataset/input-output.txt', reverse=False)

# Word vector
word_vector = KeyedVectors.load_word2vec_format("word_vector/koran.vec", binary=True)

# Params
use_cuda = torch.cuda.is_available()
hidden_size = 64
max_len = MAX_LEN

# Encoder & Decoder
encoder = EncoderEmbeddingInputRNN(src_lang.n_words, hidden_size, word_vector)
attn_decoder = AttnDecoderRNN(hidden_size, tgt_lang.n_words, dropout_p=0.1, max_length=max_len)
encoder.loadState(ENCODER_MODEL)
attn_decoder.loadState(DECODER_MODEL)

if use_cuda:
    encoder = encoder.cuda()
    attn_decoder = attn_decoder.cuda()

trainer = Trainer(src_lang, tgt_lang, pairs)

sentence = input("Input : ")
while (sentence != "<end>") :
    sentence = preprocessSentence(sentence, max_len)
    output_words, attentions = trainer.evaluate(encoder, attn_decoder, sentence, max_len=max_len)
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

# file_out = "test/resultv2-h64.txt"
# fout = open(file_out, "w", encoding="utf-8")
# for result in results :
#     fout.write("%s\n"%(result))
# fout.close()
