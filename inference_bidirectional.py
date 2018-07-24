import torch
from train_bidirectional import Trainer
# from model_bidirectional import WordEncoderBiRNN, PreTrainedEmbeddingEncoderBiRNN, AttnDecoderRNN
from model_bidirectional_v1_lstm import WordEncoderBiRNN, PreTrainedEmbeddingEncoderBiRNN, PreTrainedEmbeddingWordCharEncoderBiRNN, AttnDecoderRNN
from preprocess import prepareData, unicodeToAscii, normalizeString
import util
from lang import Lang
import params
from gensim.models import KeyedVectors
from gensim.models import FastText

def preprocessSentence(sentence, max_len) :
    # sentence = normalizeString(unicodeToAscii(sentence))
    # sentence = util.normalizeString(sentence)
    # sentence = util.normalize_no_punc(sentence)
    split = sentence.split()
    if len(split) >= max_len :
        split = split[:max_len-1]
    return ' '.join(split)

# Model files
# encoder_file = 'model/mt/encoder-word-en-id-d256-i20k.pt'
# decoder_file = 'model/mt/decoder-word-en-id-d256-i20k.pt'
# encoder_file = 'model/chatbot/fasttext/twitter_cbow/encoder-d100-e5.pt'
# decoder_file = 'model/chatbot/fasttext/twitter_cbow/decoder-d100-e5.pt'
# encoder_file = 'model/chatbot/augmented_data/word2vec/cbow/combined_cbow/charembed_encoder-d100-e3-v2.pt'
# decoder_file = 'model/chatbot/augmented_data/word2vec/cbow/combined_cbow/charembed_decoder-d100-e3-v2.pt'
# encoder_file = 'model/dialogue/encoder-charembed-d50-e100.pt'
# decoder_file = 'model/dialogue/decoder-charembed-d50-e100.pt'

# encoder_file = '/home/prosa/Works/Text/seq2seq/model/dialogue/fix/word/encoder-e15.pt'
# decoder_file = '/home/prosa/Works/Text/seq2seq/model/dialogue/fix/word/decoder-e15.pt'

# encoder_file = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/word/lr10-3/encoder-e50.pt'
# decoder_file = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/word/lr10-3/decoder-e50.pt'

# encoder_file = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/wordchar_rnn/encoder-e50.pt'
# decoder_file = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/wordchar_rnn/decoder-e50.pt'

# encoder_file = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/wordchar_cnn/encoder-e25.pt'
# decoder_file = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/wordchar_cnn/decoder-e25.pt'

# encoder_file = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/word/encoder-e50.pt'
# decoder_file = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/word/decoder-e50.pt'

# encoder_file = '/home/prosa/Works/Text/seq2seq/model/mt/tesis/oovchar_rnn/encoder-e100.pt'
# decoder_file = '/home/prosa/Works/Text/seq2seq/model/mt/tesis/oovchar_rnn/decoder-e100.pt'

encoder_file = 'model/mt/tesis/wordchar_rnn/do_0_5/encoder-e100.pt'
decoder_file = 'model/mt/tesis/wordchar_rnn/do_0_5/decoder-e100.pt'

# encoder_file = 'model/mt/tesis/wordchar_cnn/do_0_5/encoder-e100.pt' 
# decoder_file = 'model/mt/tesis/wordchar_cnn/do_0_5/decoder-e100.pt' 

# encoder_file = 'model/mt/tesis/word/do_0_5/encoder-e100.pt'
# decoder_file = 'model/mt/tesis/word/do_0_5/decoder-e100.pt'

# encoder_file = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/wordchar_cnn_rnn/avg/encoder-e50.pt'
# decoder_file = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/wordchar_cnn_rnn/avg/decoder-e50.pt'

encoder_attr_dict = torch.load(encoder_file)
decoder_attr_dict = torch.load(decoder_file)

# Lang
# encoder_lang = Lang()
decoder_lang = Lang()
# encoder_lang.load_dict(encoder_attr_dict['lang'])
decoder_lang.load_dict(decoder_attr_dict['lang'])

# Word vectors
# word_vectors = KeyedVectors.load_word2vec_format(params.WORD_VECTORS_FILE, binary=True)
# word_vectors = KeyedVectors.load(params.WORD_VECTORS_FILE)
# word_vectors = FastText.load_fasttext_format(params.WORD_VECTORS_FILE)
word_vectors = util.load_wordvector_text(params.WORD_VECTORS_FILE)

# encoder_attr_dict['dropout_p'] = 0.0

# Encoder & Decoder
# encoder = WordEncoderBiRNN(encoder_attr_dict['hidden_size'], encoder_attr_dict['max_length'], encoder_lang)
# encoder = PreTrainedEmbeddingEncoderBiRNN(word_vectors, encoder_attr_dict['hidden_size'] , encoder_attr_dict['max_length'], dropout_p=encoder_attr_dict['dropout_p'], char_embed=encoder_attr_dict['char_embed'])
encoder = PreTrainedEmbeddingWordCharEncoderBiRNN(word_vectors, encoder_attr_dict['hidden_size'], encoder_attr_dict['max_length'], char_feature=encoder_attr_dict['char_feature'], dropout_p=encoder_attr_dict['dropout_p'], seeder=params.SEEDER)
encoder.loadAttributes(encoder_attr_dict)
attn_decoder = AttnDecoderRNN(decoder_attr_dict['input_size'], decoder_attr_dict['hidden_size'], decoder_attr_dict['max_length'], decoder_lang, dropout_p=decoder_attr_dict['dropout_p'])
encoder.loadAttributes(encoder_attr_dict)
attn_decoder.loadAttributes(decoder_attr_dict)

# Trainer
trainer = Trainer([], encoder, attn_decoder)

'''
sentence = input("Input : ")
while (sentence != "<end>") :
    sentence = preprocessSentence(sentence, attn_decoder.max_length)
    output_words, attentions = trainer.evaluate(sentence)
    output = ' '.join(output_words[:-1])
    print(output)
    # output = trainer.evaluate_beam_search(sentence, 5)
    # output = [ sent[1:-1] for sent in output ]
    # output = [ ' '.join(item) for item in output ]
    # print(output)
    sentence = input("Input : ")

'''

# file_test = "/home/prosa/Works/Text/korpus/dialogue/dataset/testset/testset1k.txt"
# file_test = '/home/prosa/Works/Text/korpus/dialogue/misc.txt'
# file_test = '/home/prosa/Works/Text/seq2seq/dataset/en-id-10k-v2.txt'
file_test = '/home/prosa/Works/Text/seq2seq/test/mt/test-en-id-1k.v2.txt'
results = []
hit = 0
n_test = 1
beam_search = True
beam_width = 5
with open(file_test, "r", encoding="utf-8") as f :
    for line in f :
        line = line.strip()
        split = line.split('\t')
        if (beam_search) :
            outputs = trainer.evaluate_beam_search(split[0], beam_width)
            output = outputs[0][1:-1]
            output = ' '.join(output)
        else :
            output_words, attentions = trainer.evaluate(split[0])
            output = ' '.join(output_words[:-1])
        results.append(output)


# file_out = "/home/prosa/Works/Text/seq2seq/test/dialogue/fix/word/output1k.txt"
# file_out = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/word/hyp.txt'
# file_out = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/wordchar_rnn/hyp-beam-3.txt'
# file_out = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/wordchar_cnn/hyp.txt'
# file_out = '/home/prosa/Works/Text/seq2seq/model/dialogue/dummy/wordchar_cnn_rnn/avg/hyp.txt'
# file_out = '/home/prosa/Works/Text/seq2seq/model/mt/tesis/oovchar_rnn/hyp.txt'
file_out = '/home/prosa/Works/Text/seq2seq/model/mt/tesis/wordchar_rnn/do_0_5/hyp-beam-5.txt'
# file_out = '/home/prosa/Works/Text/seq2seq/model/mt/tesis/word/do_0_5/hyp-beam-5.txt'
# file_out = '/home/prosa/Works/Text/seq2seq/model/mt/tesis/wordchar_cnn/do_0_5/hyp-beam-3.txt'
fout = open(file_out, "w", encoding="utf-8")
for result in results :
    fout.write("%s\n"%(result))
# fout.write("Akurasi : %.4f"%(hit/n_test))
fout.close()

