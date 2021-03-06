import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from gensim.models import KeyedVectors
from lang import Lang
import params
import random
import time

def build_char_lang() :
    lang = Lang()
    lang.word2index = dict()
    lang.index2word = dict()
    lang.n_words = 0
    chars = "!\"$%&'()*+,-./0123456789:;<>?[]abcdefghijklmnopqrstuvwxyz"
    for c in chars :
        lang.addWord(c)
    return lang

# Model for word feature extraction based on character embedding using CNN
class CNNWordFeature(nn.Module) :
    def __init__(self, embedding_size, feature_size, max_length, seeder=int(time.time()) ) :
        super(CNNWordFeature, self).__init__()
        random.seed(seeder)
        torch.manual_seed(seeder)
        if params.USE_CUDA :
            torch.cuda.manual_seed_all(seeder)
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.max_length = max_length
        self.lang = build_char_lang()
        self.vocab_size = self.lang.n_words

        # embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        # convolutional layers for 2,3,4 window
        self.conv2 = nn.Conv2d(1, (self.feature_size//3)+(self.feature_size%3), (2,self.embedding_size))
        self.conv3 = nn.Conv2d(1, self.feature_size//3, (3,self.feature_size))
        self.conv4 = nn.Conv2d(1, self.feature_size//3, (4,self.feature_size))

        # maxpool layers
        self.maxpool2 = nn.MaxPool2d(kernel_size=(self.max_length-1,1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(self.max_length-2,1))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(self.max_length-3,1))

        # linear layer
        # self.linear = nn.Linear(2*(feature_size//2), feature_size)

        if params.USE_CUDA :
            self.cuda()

    # char_idxs is a list of char index (list of torch.autograd.Variable)
    def forward(self, char_idxs) :
        # Get embedding for every chars
        embeddings = Variable(torch.zeros(self.max_length, self.feature_size))
        if params.USE_CUDA :
            embeddings = embeddings.cuda()
        for i in range(len(char_idxs)) :
            c_embed = self.embedding(char_idxs[i])
            embeddings[i] = c_embed.view(1,1,-1)
        embeddings = embeddings.view(1, 1, self.max_length, -1)

        # Pass to cnn
        relu2 = F.relu(self.conv2(embeddings))
        relu3 = F.relu(self.conv3(embeddings))
        relu4 = F.relu(self.conv4(embeddings))

        # Max pooling
        pool2 = self.maxpool2(relu2).view(-1)
        pool3 = self.maxpool3(relu3).view(-1)
        pool4 = self.maxpool4(relu4).view(-1)
        
        # Concat
        concat = torch.cat((pool2,pool3,pool4))
        # concat = pool2.view(1,-1)
        # concat = torch.cat((pool2,pool3)).view(1,-1)

        # Pass to linear layer
        # output = self.linear(concat).view(-1)
        # output = pool2.view(-1)
        output = concat

        return output


# Encoder base class, only contains hidden_size, lstm layer, and empty vector
class BaseEncoderBiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, max_length, dropout_p=0.0, seeder=int(time.time()) ):
        super(BaseEncoderBiRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.model_type = 'base'
        self.dropout_p = dropout_p
        random.seed(seeder)
        torch.manual_seed(seeder)
        if params.USE_CUDA :
            torch.cuda.manual_seed_all(seeder)

        # Dropout layer
        self.dropout = nn.Dropout(p=self.dropout_p)

        # Forward and backward RNN
        self.fwd_lstm = nn.LSTM(input_size, hidden_size)
        self.rev_lstm = nn.LSTM(input_size, hidden_size)

        # define empty word vector (oov)
        # self.empty_vector = np.array([0. for _ in range(hidden_size)])
        self.empty_vector = np.array([0. for _ in range(input_size)])

        # define initial cell and hidden vector
        self.h0 = Variable(torch.zeros(1, 1, self.hidden_size))
        self.c0 = Variable(torch.zeros(1, 1, self.hidden_size))

        if params.USE_CUDA :
            self.cuda()
            self.h0 = self.h0.cuda()
            self.c0 = self.c0.cuda()

    # Input is list of embedding
    def forward(self, input):
        embedding_inputs = input

        # Forward to fwd_lstm unit
        # (fwd_hidden, fwd_cell) = self.initHidden()
        fwd_hidden, fwd_cell = self.h0, self.c0
        fwd_outputs = Variable(torch.zeros(self.max_length, self.hidden_size))
        if params.USE_CUDA :
            fwd_outputs = fwd_outputs.cuda()
        for k,embed in enumerate(embedding_inputs) :
            embed = self.dropout(embed)
            fwd_output,(fwd_hidden, fwd_cell) = self.fwd_lstm(embed, (fwd_hidden, fwd_cell))
            fwd_outputs[k] = fwd_output[0][0]

        # Forward to rev_lstm unit
        (rev_hidden, rev_cell) = self.initHidden()
        rev_outputs = Variable(torch.zeros(self.max_length, self.hidden_size))
        if params.USE_CUDA :
            rev_outputs = rev_outputs.cuda()
        n = len(embedding_inputs)-1
        for i in range(n,-1,-1) :
            rev_output,(rev_hidden, rev_cell) = self.rev_lstm(embedding_inputs[i], (rev_hidden, rev_cell))
            rev_outputs[i] = rev_output[0][0]
        
        # Concatenate fwd_output and rev_output
        outputs = torch.cat( (fwd_outputs, rev_outputs), 1 )
        hidden = torch.cat( (fwd_hidden, rev_hidden), 2 )
        cell = torch.cat( (fwd_cell, rev_cell), 2 )
        
        if params.USE_CUDA :
            # return outputs.cuda(), hidden.cuda()
            outputs = outputs.cuda()
            hidden = hidden.cuda()

        # projected_output = self.projection(hidden)
        projected_output = (hidden, cell)

        return outputs, (hidden, cell), projected_output

    def initHidden(self):
        h0 = Variable(torch.zeros(1, 1, self.hidden_size))
        c0 = Variable(torch.zeros(1, 1, self.hidden_size))
        if params.USE_CUDA:
            return (h0.cuda(), c0.cuda())
        else:
            return (h0, c0)

    def getCpuStateDict(self) :
        state_dict = self.state_dict()
        if params.USE_CUDA :
            for key in state_dict :
                state_dict[key] = state_dict[key].cpu()
        return state_dict

    def getAttrDict(self):
        return None

    def loadAttributes(self, attr_dict):
        self.input_size = attr_dict['input_size']
        self.hidden_size = attr_dict['hidden_size']
        self.max_length = attr_dict['max_length']
        self.model_type = attr_dict['model_type']
        self.dropout_p = attr_dict['dropout_p']

# Encoder ...
class WordCharEncoderBiRNN(BaseEncoderBiRNN) :
    def __init__(self, input_size, hidden_size, max_length, char_feature='cnn', dropout_p=0.0, seeder=int(time.time())) :
        super(WordCharEncoderBiRNN, self).__init__(input_size*2, hidden_size, max_length, dropout_p=dropout_p, seeder=seeder)
        assert (char_feature == 'rnn' or char_feature == 'cnn' or char_feature == 'cnn_rnn')
        if char_feature == 'rnn' :
            self.charbased_model = self.build_rnn(seeder)
        elif char_feature == 'cnn' :
            self.charbased_model = self.build_cnn(seeder)
        elif char_feature == 'cnn_rnn' :
            self.charbased_rnn = self.build_rnn(seeder)
            self.charbased_cnn = self.build_cnn(seeder)
        self.char_feature = char_feature
        self.model_type = ''

    def build_cnn(self, seeder=int(time.time()) ) :
        return CNNWordFeature(self.input_size//2, self.input_size//2, params.CHAR_LENGTH, seeder=seeder)

    def build_rnn(self, seeder=int(time.time()) ) :
        lang = build_char_lang()
        return WordEncoderBiRNN(self.input_size//4, self.input_size//4, params.CHAR_LENGTH, lang, seeder=seeder)

    # Word_embeddings is word_vector of sentence, words is list of word
    def forward(self, word_embeddings, words) :
        assert(len(word_embeddings) == len(words))

        # Get word embeddings extracted from its character
        char_embeddings = []
        for word in words :
            # Get character indexes
            if self.char_feature == 'cnn_rnn' :
                inputs = [self.charbased_cnn.lang.word2index[c] for c in word]
            else :
                inputs = [self.charbased_model.lang.word2index[c] for c in word]
            inputs = Variable(torch.LongTensor(inputs))
            if params.USE_CUDA :
                inputs = inputs.cuda()
            
            # Get vector rep of word (pass to charbased_model)
            if self.char_feature == 'cnn' :
                vec = self.charbased_model(inputs)
            elif self.char_feature == 'rnn' :
                _, _, (vec, cell)  = self.charbased_model(inputs)
            elif self.char_feature == 'cnn_rnn' :
                cnn_vec = self.charbased_cnn(inputs)
                _, _, (rnn_vec, cell) = self.charbased_rnn(inputs)
                # Addition
                # vec = cnn_vec + rnn_vec
                # Average
                # vec = (cnn_vec + rnn_vec) / 2
                # Hadamard product
                vec = cnn_vec * rnn_vec

            # Add to list of word embeddings based on char
            char_embeddings.append(vec.view(1,1,-1))

        # concat word_embeddings with char_embeddings
        embeddings = []
        for i in range(len(word_embeddings)) :
            embeddings.append(torch.cat((word_embeddings[i],char_embeddings[i]),2) )

        # print('word embedding :')
        # print(word_embeddings)
        # print('word-char embedding :')
        # print(char_embeddings)
        # print('embedding :')
        # print(embeddings)

        # Forward to rnn
        return super(WordCharEncoderBiRNN, self).forward(embeddings)

    def loadAttributes(self, attr_dict) :
        super(WordCharEncoderBiRNN, self).loadAttributes(attr_dict)
        self.load_state_dict(attr_dict['state_dict'])

    def getAttrDict(self) :
        return {
            'model_type' : self.model_type,
            'char_feature' : self.char_feature,
            'input_size' : self.input_size,
            'hidden_size' : self.hidden_size,
            'max_length' : self.max_length,
            'dropout_p' : self.dropout_p,
            'state_dict' : self.getCpuStateDict(),
        }

class PreTrainedEmbeddingWordCharEncoderBiRNN(WordCharEncoderBiRNN) :
    def __init__(self, word_vectors, hidden_size, max_length, char_feature='cnn', dropout_p=0.0, seeder=int(time.time())) :
        super(PreTrainedEmbeddingWordCharEncoderBiRNN, self).__init__(word_vectors.vector_size, hidden_size, max_length, char_feature, dropout_p=dropout_p, seeder=seeder)
        empty_vector = np.array([0. for _ in range(word_vectors.vector_size)])
        self.empty_vector = Variable(torch.Tensor(empty_vector).view(1, 1, -1))
        self.cache_dict = dict()
        self.word_vectors = word_vectors
        self.model_type = 'pre_trained_embedding_wordchar'
        if params.USE_CUDA :
            self.cuda()
            self.empty_vector = self.empty_vector.cuda()

    def get_word_vector(self, word_input) :
        if word_input in self.cache_dict :
            return self.cache_dict[word_input]
        else :
            if word_input in self.word_vectors :
                # If word is not oov, take embedding vector of it
                word_embed = self.word_vectors[word_input]
                word_vector = Variable(torch.Tensor(word_embed)).view(1, 1, -1)
                if params.USE_CUDA:
                    word_vector = word_vector.cuda()
            else :
                # Word is oov, take [0, 0, 0, ...] as embedding vector
                word_vector = self.empty_vector
            self.cache_dict[word_input] = word_vector
            return word_vector

    # Feed forward method, input is list of word
    def forward(self, input) :
        word_embeddings = [self.get_word_vector(word) for word in input]
        return super(PreTrainedEmbeddingWordCharEncoderBiRNN, self).forward(word_embeddings, input)

    def getAttrDict(self):
        return {
            'model_type' : self.model_type,
            'char_feature' : self.char_feature,
            'input_size' : self.input_size,
            'hidden_size' : self.hidden_size,
            'max_length' : self.max_length,
            'dropout_p' : self.dropout_p,
            'state_dict' : self.getCpuStateDict(),
        }

    def loadAttributes(self, attr_dict) :
        super(PreTrainedEmbeddingWordCharEncoderBiRNN, self).loadAttributes(attr_dict)

# Encoder word based
class WordEncoderBiRNN(BaseEncoderBiRNN):
    def __init__(self, input_size, hidden_size, max_length, lang, dropout_p=0.0, seeder=int(time.time())):
        super(WordEncoderBiRNN, self).__init__(input_size, hidden_size, max_length, dropout_p=dropout_p, seeder=seeder)
        self.model_type = 'word_based'

        # define parameters
        self.lang = lang
        self.vocab_size = lang.n_words

        # define layers
        self.embedding = nn.Embedding(self.vocab_size, self.input_size)

        # empty vector for oov
        self.empty_vector = Variable(torch.Tensor(self.empty_vector)).view(1, 1, -1)

        if params.USE_CUDA :
            self.cuda()
            self.empty_vector = self.empty_vector.cuda()

    def loadAttributes(self, attr_dict) :
        super(WordEncoderBiRNN, self).loadAttributes(attr_dict)
        self.lang.load_dict(attr_dict['lang'])
        self.vocab_size = self.lang.n_words
        self.load_state_dict(attr_dict['state_dict'])
        if params.USE_CUDA :
            self.cuda()

    # Feed forward method, input is a list of word index (list of torch.autograd.Variable)
    def forward(self, input):
        # Get embedding vector
        embedding_inputs = []
        for idx in input :
            if idx.data.item() != params.OOV_INDEX :
                embed = self.embedding(idx).view(1,1,-1)
            else :
                embed = self.empty_vector
            embedding_inputs.append(embed)
            
        return super(WordEncoderBiRNN, self).forward(embedding_inputs)

    # Get word index of every word in sentence
    def get_indexes(self, sentence, reverse_direction=False) :
        if not reverse_direction :
            arr = [self.lang.word2index[word] if word in self.lang.word2index else -1 for word in sentence]
        else :
            arr = [self.lang.word2index[sentence[i]] if sentence[i] in self.lang.word2index else -1 for i in range(len(sentence)-1,-1,-1)]
        retval = Variable(torch.LongTensor(arr))
        if params.USE_CUDA :
            return retval.cuda()
        return retval

    # Get dict representation of attributes
    def getAttrDict(self):
        return {
            'model_type' : self.model_type,
            'input_size' : self.input_size,
            'hidden_size' : self.hidden_size,
            'max_length' : self.max_length,
            'lang' : self.lang.getAttrDict(),
            'state_dict' : self.getCpuStateDict(),
        }

# Encoder using pre trained word embedding
class PreTrainedEmbeddingEncoderBiRNN(BaseEncoderBiRNN) :
    def __init__(self, word_vectors, hidden_size, max_length, dropout_p=0.0, char_embed=False, seeder=int(time.time())):
        super(PreTrainedEmbeddingEncoderBiRNN, self).__init__(word_vectors.vector_size, hidden_size, max_length, dropout_p=dropout_p, seeder=seeder)
        self.model_type = 'pre_trained_embedding'

        # define word vector embedding
        self.word_vectors = word_vectors

        # empty vector for oov
        self.empty_vector = Variable(torch.Tensor(self.empty_vector)).view(1, 1, -1)

        # char embed
        self.char_embed = char_embed
        if self.char_embed :
            lang = build_char_lang()
            self.charbased_model = WordEncoderBiRNN(self.hidden_size//2, params.CHAR_LENGTH, lang, seeder=seeder)

        if params.USE_CUDA :
            self.cuda()
            self.empty_vector = self.empty_vector.cuda()

        self.cache_dict = dict()

    def get_word_vector(self, word_input) :
        if word_input in self.cache_dict :
            return self.cache_dict[word_input]
        else :
            if word_input in self.word_vectors :
                # If word is not oov, take embedding vector of it
                word_embed = self.word_vectors[word_input]
                word_vector = Variable(torch.Tensor(word_embed)).view(1, 1, -1)
                if params.USE_CUDA:
                    word_vector = word_vector.cuda()
            else :
                # Word is oov, take [0, 0, 0, ...] as embedding vectors
                word_vector = self.empty_vector
            self.cache_dict[word_input] = word_vector
            return word_vector

    # Feed forward method, input is list of word
    def forward(self, input):
        embedding_inputs = []
        for word in input :
            if (word not in self.word_vectors) and (self.char_embed) :
                inputs = [self.charbased_model.lang.word2index[c] for c in word]
                inputs = Variable(torch.LongTensor(inputs))
                if params.USE_CUDA :
                    inputs = inputs.cuda()
                _, _, (c_hidden, c_cell)  = self.charbased_model(inputs)
                embedding_inputs.append(c_hidden)
            else :
                embedding_inputs.append(self.get_word_vector(word))
        return super(PreTrainedEmbeddingEncoderBiRNN, self).forward(embedding_inputs)

    def loadAttributes(self, attr_dict) :
        super(PreTrainedEmbeddingEncoderBiRNN, self).loadAttributes(attr_dict)
        self.load_state_dict(attr_dict['state_dict'])

    def getAttrDict(self):
        return {
            'model_type' : self.model_type,
            'input_size' : self.input_size,
            'hidden_size' : self.hidden_size,
            'max_length' : self.max_length,
            'char_embed' : self.char_embed,
            'dropout_p' : self.dropout_p,
            'state_dict' : self.getCpuStateDict(),
        }
'''
# Encoder using word embedding vector as input
class EmbeddingEncoderInputBiRNN(nn.Module):
    def __init__(self, hidden_size, word_vector):
        super(EmbeddingEncoderInputBiRNN, self).__init__(hidden_size)
        self.model_type = 'word_vector_based'

        # define word vector embedding
        self.word_vector = word_vector

        # define layers
        self.linear = nn.Linear(self.word_vector.vector_size, hidden_size)

        # define empty word vector (oov)
        self.empty_vector = np.array([0. for _ in range(self.word_vector.vector_size)])

    def loadAttributes(self, attr_dict) :
        super(EmbeddingEncoderInputBiRNN, self).loadAttributes(attr_dict)
        self.max_length = attr_dict['max_length']
        self.load_state_dict(attr_dict['state_dict'])

    # Feed forward method, input is a word
    def forward(self, input, hidden):
        if input in self.word_vector :
            # If word is not oov, take embedding vector of it
            word_embed = self.word_vector[input]
        else :
            # Word is oov, take [0, 0, 0, ...] as embedding vectors
            word_embed = self.empty_vector
        input = Variable(torch.Tensor(word_embed)).view(1, 1, -1)
        if params.USE_CUDA:
            input = input.cuda()

        # Feed forward to linear unit
        input = self.linear(input)

        # Feed forward to gru unit
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if params.USE_CUDA:
            return result.cuda()
        else:
            return result

    def saveState(self, filepath):
        torch.save(self.state_dict(), filepath)

    def getAttrDict(self):
        return {
            'model_type' : self.model_type,
            'hidden_size' : self.hidden_size,
            'max_length' : self.max_length,
            'state_dict' : self.getCpuStateDict(),
        }
'''
# Decoder
class AttnDecoderRNN(nn.Module):
    def __init__( self, input_size, hidden_size, max_length, lang, dropout_p=0.5, seeder=int(time.time()) ):
        super(AttnDecoderRNN, self).__init__()
        random.seed(seeder)
        torch.manual_seed(seeder)
        if params.USE_CUDA :
            torch.cuda.manual_seed_all(seeder)

        # define parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = lang.n_words
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.lang = lang

        # define layers
        self.embedding = nn.Embedding(self.output_size, self.input_size)
        self.attn = nn.Linear(self.input_size+self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.input_size+self.hidden_size, self.input_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        if params.USE_CUDA :
            self.cuda()

    def loadAttributes(self, attr_dict) :
        self.input_size = attr_dict['input_size']
        self.hidden_size = attr_dict['hidden_size']
        self.dropout_p = attr_dict['dropout_p']
        self.max_length = attr_dict['max_length']
        self.lang.load_dict(attr_dict['lang'])
        self.output_size = self.lang.n_words
        self.load_state_dict(attr_dict['state_dict'])

    # Feed forward method, input is index of word (Variable)
    def forward(self, input, hidden, encoder_outputs):
        hidden,cell = hidden

        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_input = torch.cat((embedded[0], hidden[0]), 1)
        attn_weights = F.softmax(self.attn(attn_input), dim=1)
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, (hidden,cell) = self.lstm(output, (hidden,cell) )
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, (hidden,cell), attn_weights

    def initHidden(self):
        h0 = Variable(torch.zeros(1, 1, self.hidden_size))
        c0 = Variable(torch.zeros(1, 1, self.hidden_size))
        if params.USE_CUDA:
            return (h0.cuda(), c0.cuda())
        else:
            return (h0, c0)

    def getCpuStateDict(self) :
        state_dict = self.state_dict()
        if params.USE_CUDA :
            for key in state_dict :
                state_dict[key] = state_dict[key].cpu()
        return state_dict

    def getAttrDict(self):
        return {
            'input_size' : self.input_size,
            'hidden_size' : self.hidden_size,
            'dropout_p' : self.dropout_p,
            'max_length' : self.max_length,
            'lang' : self.lang.getAttrDict(),
            'state_dict' : self.getCpuStateDict(),
        }

