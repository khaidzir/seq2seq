import torch

from gensim.models import FastText
from gensim.models import KeyedVectors
from preprocess import prepareData
from model_bidirectional import WordEncoderBiRNN, PreTrainedEmbeddingEncoderBiRNN, AttnDecoderRNN
from train_bidirectional import Trainer
import params

root_folder = '/home/prosa/Works/Text/'
dataset_folder = root_folder + 'korpus/chatbot_dataset/'

src_lang, tgt_lang, pairs = prepareData(dataset_folder + 'input-response.shuffle', reverse=False)
word_vectors = KeyedVectors.load(params.WORD_VECTORS_FILE)
# word_vectors = FastText.load_fasttext_format(params.WORD_VECTORS_FILE)

hidden_size = word_vectors.vector_size
max_length = 50
k_validation = 10
n_data = len(pairs)
last_idx = 0
epoch = 5

tests = []
outputs = []
refs = []

for k in range(k_validation) :
    # Prepare dataset for kth-fold
    ntest = n_data//k_validation
    if k < n_data%k_validation :
        ntest += 1
    test_set = pairs[last_idx:last_idx+ntest]
    train_set = pairs[:last_idx] + pairs[last_idx+ntest:]
    last_idx += ntest

    # Train
    num_iter = len(train_set)
    encoder = PreTrainedEmbeddingEncoderBiRNN(word_vectors, max_length)
    decoder = AttnDecoderRNN(2*hidden_size, max_length, tgt_lang)
    trainer = Trainer(train_set, encoder, decoder)
    print("Training fold-%d (%d data)..."%(k+1, len(train_set)))
    trainer.train(num_iter, print_every=num_iter//100, epoch=epoch)

    # Validation
    print("Validation fold-%d (%d data)..."%(k+1, len(test_set)))
    for pair in test_set :
        tests.append(pair[0])
        refs.append(pair[1])
        decoded_words, _ = trainer.evaluate(pair[0])
        if decoded_words[-1] == '<EOS>' :
            decoded_words = decoded_words[:-1]
        outputs.append(' '.join(decoded_words))

# Write results to file
# test_file = 'test/chatbot/fasttext/combined_cbow/test-d' + str(hidden_size) + '-e' + str(epoch) + '.txt'
# output_file = 'test/chatbot/fasttext/combined_cbow/output-d' + str(hidden_size) + '-e' + str(epoch) + '.txt'
# ref_file = 'test/chatbot/fasttext/combined_cbow/ref-d' + str(hidden_size) + '-e' + str(epoch) + '.txt'

# Word2vec
test_file = 'test/chatbot/word2vec/codot_cbow/test-d' + str(hidden_size) + '-e' + str(epoch) + '.txt'
output_file = 'test/chatbot/word2vec/codot_cbow/output-d' + str(hidden_size) + '-e' + str(epoch) + '.txt'
ref_file = 'test/chatbot/word2vec/codot_cbow/ref-d' + str(hidden_size) + '-e' + str(epoch) + '.txt'

fileouts = [test_file, output_file, ref_file]
dataouts = [tests, outputs, refs]

for i in range(len(dataouts)) :
    fout = open(fileouts[i], 'w', encoding='utf-8')
    for sent in dataouts[i] :
        fout.write("%s\n"%(sent))
    fout.close()

# Build final model
print('\nBuild final model...')
num_iter = len(pairs)
encoder = PreTrainedEmbeddingEncoderBiRNN(word_vectors, max_length)
decoder = AttnDecoderRNN(2*hidden_size, max_length, tgt_lang)
trainer = Trainer(pairs, encoder, decoder)
trainer.train(num_iter, print_every=num_iter//100, epoch=epoch)

# Save model
# FastText
# torch.save(encoder.getAttrDict(), 'model/chatbot/fasttext/combined_cbow/encoder-d' + str(hidden_size) + '-e' + str(epoch) + '.pt')
# torch.save(decoder.getAttrDict(), 'model/chatbot/fasttext/combined_cbow/decoder-d' + str(hidden_size) + '-e' + str(epoch) + '.pt')

# Word2vec
torch.save(encoder.getAttrDict(), 'model/chatbot/word2vec/codot_cbow/encoder-d' + str(hidden_size) + '-e' + str(epoch) + '.pt')
torch.save(decoder.getAttrDict(), 'model/chatbot/word2vec/codot_cbow/decoder-d' + str(hidden_size) + '-e' + str(epoch) + '.pt')
