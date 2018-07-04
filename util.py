import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import params
import unicodedata
import re
from io import open
from simple_wordvector import SimpleWordVector

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
numbers_ext = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '.']
NUM_TOKEN = "<NUM>"

# --- NORMALIZATION: TO CLEAN UNICODE CHARACTERS ----
def unicodeToAscii(sentence):
    return ''.join(
        c for c in unicodedata.normalize('NFD', sentence)
        if unicodedata.category(c) != 'Mn'
        )
def normalizeString(sentence):
    sentence = unicodeToAscii(sentence.lower().strip())
    sentence = re.sub(r"([,.;?!'\"\-()<>[\]/\\&$%*@~+=])", r" \1 ", sentence)
    sentence = re.sub(r"[^a-zA-Z0-9,.;?!'\"\-()<>[\]/\\&$%*@~+=]+", r" ", sentence)
    return sentence

def is_numeric(word) :
    if word[0] not in numbers :
        return False
    for c in word :
        if c not in numbers_ext :
            return False
    return True

def filter_numeric(words) :
    sentence = ''
    for word in words :
        if is_numeric(word) :
            sentence += NUM_TOKEN
        else :
            sentence += word
        sentence += ' '
    return sentence[:-1]

def normalize_no_punc(sentence) :
    sentence = unicodeToAscii(sentence.lower().strip())
    sentence = re.sub(r"([;?!'\"\-()<>[\]/\\&$%*@~+=])", r" ", sentence)
    sentence = filter_numeric(sentence.split())
    sentence = re.sub(r"[^a-z<>A-Z0-9]+", r" ", sentence)
    return sentence

# Get list of word indexes representation from sentence (list of words)
def get_sequence_index(sentence, word2index) :
    arr = [word2index[word] if word in word2index else params.OOV_INDEX for word in sentence]
    retval = Variable(torch.LongTensor(arr))
    if params.USE_CUDA :
        return retval.cuda()
    return retval

# Get wordvector from text file
def load_wordvector_text(word_vector_file):
    
    model = SimpleWordVector()
    with open(word_vector_file,'r') as f :
        setsize = False
        for line in f :
            split = line.split()
            word = split[0]
            wv = np.array( [float(val) for val in split[1:]] )
            model[word] = wv
            if not setsize :
                model.vector_size = len(wv)
                setsize = True
    return model

'''
glovefile = '/home/prosa/Downloads/glove.6B/glove.6B.50d.txt'
model = load_wordvector_text(glovefile)
word = input('kata : ')
while word != '<end>' :
    if word in model :
        print(model[word])
    else :
        print('euweuh!')
    word = input('kata : ')
'''