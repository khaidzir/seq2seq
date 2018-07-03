import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from gensim.models import KeyedVectors
import params

OOV_INDEX = -1  # Word index to represent oov word

# Encoder base class, only contains hidden_size, gru layer, and empty vector
class BaseEncoderRNN(nn.Module):
    def __init__(self, hidden_size, max_length):
        super(BaseEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.gru = nn.GRU(hidden_size, hidden_size)

        self.model_type = 'base'

        # define empty word vector (oov)
        self.empty_vector = np.array([0. for _ in range(hidden_size)])

    # Feed forward method, input is word
    def forward(self, input):
        embedding_inputs = input

        # Forward to fwd_gru unit
        hidden = self.initHidden()
        outputs = Variable(torch.zeros(self.max_length, self.hidden_size))
        if params.USE_CUDA :
            outputs = outputs.cuda()
        for k,embed in enumerate(embedding_inputs) :
            output,hidden = self.gru(embed, hidden)
            outputs[k] = output[0][0]

        return outputs, hidden, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if torch.cuda.is_available():
            return result.cuda()
        else:
            return result

    def getCpuStateDict(self) :
        state_dict = self.state_dict()
        if torch.cuda.is_available() :
            for key in state_dict :
                state_dict[key] = state_dict[key].cpu()
        return state_dict

    def getAttrDict(self):
        return None

    def loadAttributes(self, attrDict):
        return None

'''
# Encoder word based
class WordEncoderRNN(BaseEncoderRNN):
    def __init__(self, hidden_size, lang):
        super(WordEncoderRNN, self).__init__(hidden_size)
        self.model_type = 'word_based'

        # define parameters
        self.lang = lang
        self.input_size = lang.n_words

        # define layers
        self.embedding = nn.Embedding(self.input_size, hidden_size)

        self.empty_vector = Variable(torch.Tensor(self.empty_vector)).view(1, 1, -1)
        if torch.cuda.is_available() :
            self.empty_vector = self.empty_vector.cuda()

    def loadAttributes(self, attr_dict) :
        self.hidden_size = attr_dict['hidden_size']
        self.lang.load_dict(attr_dict['lang'])
        self.input_size = lang.n_words
        self.max_length = attr_dict['max_length']
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.empty_vector = Variable(torch.Tensor(self.empty_vector)).view(1, 1, -1)
        if torch.cuda.is_available() :
            self.empty_vector = self.empty_vector.cuda()
        self.load_state_dict(attr_dict['state_dict'])

    # Feed forward method, input is word
    def forward(self, input, hidden):
        # If word is not oov, take embedding vector of it
        if input in lang.word2index :
            embedded = self.embedding(lang.word2index[input]).view(1,1,-1)
        else :
            # Word is oov, take [0, 0, 0, ...] as embedding vectors
            embedded = self.empty_vector
        output = embedded
        # Forward to GRU unit
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def getAttrDict(self):
        state_dict= self.state_dict()
        if torch.cuda.is_available() :
            return {
                'model_type' : self.model_type,
                'hidden_size' : self.hidden_size,
                'max_length' : self.max_length,
                'lang' : self.lang.getAttrDict(),
                'state_dict' : self.getCpuStateDict(),
            }
'''

# Encoder using pre trained word embedding
class PreTrainedEmbeddingEncoderRNN(BaseEncoderRNN) :
    def __init__(self, word_vector, max_length, char_embed=False):
        super(PreTrainedEmbeddingEncoderRNN, self).__init__(word_vector.vector_size, max_length)
        self.model_type = 'pre_trained_embedding'
        self.max_length = max_length

        # define word vector embedding
        self.word_vectors = word_vector

        # empty vector for oov
        self.empty_vector = Variable(torch.Tensor(self.empty_vector)).view(1, 1, -1)

        # char embed
        self.char_embed = char_embed

        # word vector for start of string
        sos = torch.ones(self.hidden_size)
        self.sos_vector = Variable(sos).view(1, 1, -1)

        # word vector for end of string
        eos = torch.ones(self.hidden_size) * -1
        self.eos_vector = Variable(eos).view(1, 1, -1)

        if params.USE_CUDA :
            self.cuda()
            self.empty_vector = self.empty_vector.cuda()
            self.sos_vector = self.sos_vector.cuda()
            self.eos_vector = self.eos_vector.cuda()

        self.cache_dict = dict()
        self.cache_dict[params.SOS_TOKEN] = self.sos_vector
        self.cache_dict[params.EOS_TOKEN] = self.eos_vector

    def loadAttributes(self, attr_dict) :
        self.max_length = attr_dict['max_length']
        self.hidden_size = attr_dict['hidden_size']
        self.load_state_dict(attr_dict['state_dict'])

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

    '''
    # Feed forward method, input is a word
    def forward(self, input, hidden):
        if input in self.word_vector :
            # If word is not oov, take embedding vector of it
            word_embed = self.word_vector[input]
        else :
            # Word is oov, take [0, 0, 0, ...] as embedding vectors
            word_embed = self.empty_vector
        input = Variable(torch.Tensor(word_embed)).view(1, 1, -1)
        if torch.cuda.is_available():
            input = input.cuda()
        # Feed forward to gru unit
        output, hidden = self.gru(input, hidden)
        return output, hidden
    '''

    # Feed forward method, input is list of word
    def forward(self, input):
        embedding_inputs = []
        for word in input :
            if (word not in self.word_vectors) and (self.char_embed) :
                inputs = [self.charbased_model.lang.word2index[c] for c in word]
                inputs = Variable(torch.LongTensor(inputs))
                if params.USE_CUDA :
                    inputs = inputs.cuda()
                _, char_vector = self.charbased_model(inputs)
                embedding_inputs.append(char_vector)
            else :
                embedding_inputs.append(self.get_word_vector(word))
        return super(PreTrainedEmbeddingEncoderRNN, self).forward(embedding_inputs)

    def getAttrDict(self):
        return {
            'model_type' : self.model_type,
            'hidden_size' : self.hidden_size,
            'max_length' : self.max_length,
            'state_dict' : self.getCpuStateDict(),
        }

'''
# Encoder using word embedding vector as input
class EmbeddingEncoderInputRNN(nn.Module):
    def __init__(self, hidden_size, word_vector):
        super(EmbeddingEncoderInputRNN, self).__init__(hidden_size)
        self.model_type = 'word_vector_based'

        # define word vector embedding
        self.word_vector = word_vector

        # define layers
        self.linear = nn.Linear(self.word_vector.vector_size, hidden_size)

        # define empty word vector (oov)
        self.empty_vector = np.array([0. for _ in range(self.word_vector.vector_size)])

    def loadAttributes(self, attr_dict) :
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
        if torch.cuda.is_available():
            input = input.cuda()

        # Feed forward to linear unit
        input = self.linear(input)

        # Feed forward to gru unit
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if torch.cuda.is_available():
            return result.cuda()
        else:
            return result

    def saveState(self, filepath):
        torch.save(self.state_dict(), filepath)

    def getAttrDict(self):
        return {
            'model_type' : self.model_type,
            'hidden_size' : self.hidden_size,
            'state_dict' : self.getCpuStateDict(),
        }
'''

# Decoder
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, lang, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()

        # define parameters
        self.hidden_size = hidden_size
        self.output_size = lang.n_words
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.lang = lang

        # define layers
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def loadAttributes(self, attr_dict) :
        self.hidden_size = attr_dict['hidden_size']
        self.dropout_p = attr_dict['dropout_p']
        self.max_length = attr_dict['max_length']
        self.lang.load_dict(attr_dict['lang'])
        self.output_size = self.lang.n_words
        self.load_state_dict(attr_dict['state_dict'])

    # Feed forward method
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if torch.cuda.is_available():
            return result.cuda()
        else:
            return result

    def getCpuStateDict(self) :
        state_dict = self.state_dict()
        if torch.cuda.is_available() :
            for key in state_dict :
                state_dict[key] = state_dict[key].cpu()
        return state_dict

    def getAttrDict(self):
        return {
            'hidden_size' : self.hidden_size,
            'dropout_p' : self.dropout_p,
            'max_length' : self.max_length,
            'lang' : self.lang.getAttrDict(),
            'state_dict' : self.getCpuStateDict(),
        }

