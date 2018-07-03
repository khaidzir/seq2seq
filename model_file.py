import torch
from model2 import WordEncoderRNN, PreTrainedEmbeddingEncoderRNN, EmbeddingEncoderInputRNN, AttnDecoderRNN
from lang import Lang
from gensim.models import KeyedVectors

def model_save(model, filepath) :
    attr_dict = model.getAttrDict()
    torch.save(attr_dict, filepath)

def encoder_load(file_path, word_vector=None) :
    attr_dict = torch.load(file_path)
    encoder_type = attr_dict['model_type']
    encoder = None
    if encoder_type == "word_based" :
        encoder = WordEncoderRNN(attr_dict['hidden_size'], Lang(attr_dict['lang']))
        encoder.loadAttributes(attr_dict)
    elif encoder_type == "pre_trained_embedding" :
        encoder = WordEncoderRNN(word_vector)
        encoder.loadAttributes(attr_dict)
    elif encoder_type == "word_vector_based" :
        encoder = EmbeddingEncoderInputRNN(attr_dict['hidden_size'])
        encoder.loadAttributes(attr_dict)
    return encoder

def decoder_load(file_path) :
    attr_dict = torch.load(file_path)
    hidden_size = attr_dict['hidden_size']
    lang = Lang(attr_dict['lang'])
    dropout_p = attr_dict['dropout_p']
    max_length = attr_dict['max_length']
    decoder = AttnDecoderRNN(hidden_size, lang, dropout_p, max_length)
    decoder.loadAttributes(attr_dict)
    return decoder
