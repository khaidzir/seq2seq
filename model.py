import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from gensim.models import KeyedVectors

OOV_INDEX = -1  # Word index to represent oov word

# Encoder word based
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()

        # define parameters
        self.hidden_size = hidden_size  # Dimension of embedding word vector

        # define layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        # define empty word vector (oov)
        self.empty_vector = np.array([0. for _ in range(hidden_size)])
        self.empty_vector = Variable(torch.Tensor(self.empty_vector)).view(1, 1, -1)
        if torch.cuda.is_available() :
            self.empty_vector = self.empty_vector.cuda()

    # Feed forward method, input is index of word
    def forward(self, input, hidden):
        # If word is not oov, take embedding vector of it
        if input.data[0] != OOV_INDEX :
            embedded = self.embedding(input).view(1, 1, -1)
        else :
            # Word is oov, take [0, 0, 0, ...] as embedding vectors
            embedded = self.empty_vector
        output = embedded
        # Forward to GRU unit
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if torch.cuda.is_available():
            return result.cuda()
        else:
            return result

    def saveState(self, filepath):
        torch.save(self.state_dict(), filepath)

    def loadState(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
        # self.load_state_dict(torch.load(filepath))

# Encoder using pre trained word embedding
class EncoderEmbeddingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, word_vector):
        super(EncoderEmbeddingRNN, self).__init__()

        # define parameters
        self.hidden_size = hidden_size  # Dimension of embedding word vector

        # define layers
        self.gru = nn.GRU(hidden_size, hidden_size)

        # define word vector embedding
        self.word_vector = word_vector

        # define empty word vector (oov)
        self.empty_vector = np.array([0. for _ in range(hidden_size)])

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

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if torch.cuda.is_available():
            return result.cuda()
        else:
            return result

    def saveState(self, filepath):
        torch.save(self.state_dict(), filepath)

    def loadState(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
        # self.load_state_dict(torch.load(filepath))

# Encoder using pre-trained embedding as input
class EncoderEmbeddingInputRNN(nn.Module):
    def __init__(self, input_size, hidden_size, word_vector):
        super(EncoderEmbeddingInputRNN, self).__init__()

        # define parameters
        self.hidden_size = hidden_size  # Dimension of embedding word vector

        # define word vector embedding
        self.word_vector = word_vector

        # define layers
        self.linear = nn.Linear(self.word_vector.vector_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        # define empty word vector (oov)
        self.empty_vector = np.array([0. for _ in range(self.word_vector.vector_size)])

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

    def loadState(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
        # self.load_state_dict(torch.load(filepath))

# Decoder
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()

        # define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # define layers
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

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

    def saveState(self, filepath):
        torch.save(self.state_dict(), filepath)

    def loadState(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
        # self.load_state_dict(torch.load(filepath))
