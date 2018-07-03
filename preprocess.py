from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import os

from lang import Lang

import util

# --- NORMALIZATION: TO CLEAN UNICODE CHARACTERS ----

MAX_LEN = 80

def unicodeToAscii(sentence):
    return ''.join(
        c for c in unicodedata.normalize('NFD', sentence)
        if unicodedata.category(c) != 'Mn'
        )

def normalizeString(sentence):
    # sentence = unicodeToAscii(sentence.lower().strip())
    # sentence = re.sub(r"([,.;?!'\"\-()<>[\]/\\&$%*@~+=])", r" \1 ", sentence)
    # sentence = re.sub(r"[^a-zA-Z0-9,.;?!'\"\-()<>[\]/\\&$%*@~+=]+", r" ", sentence)
    return sentence

def filterPair(pair):
    return True
    # return len(pair[0].split()) < MAX_LEN and len(pair[1].split()) < MAX_LEN

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# main function, readLang + prepareData
def readLang(filepath, reverse=False):
    print("reading lines...")

    # read the file and split into lines
    lines = open(filepath, encoding='utf-8').read().strip().split('\n')

    # getting the language names from filename
    # filename = os.path.splitext(os.path.basename(filepath))[0]
    # lang = filename.split('-')

    # split every line into pairs and normalize
    # pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # For chatbot, normalize only input side
    pairs = []
    for l in lines :
        split = l.split('\t')
        pairs.append([split[0], split[1]])
        # pairs.append( [util.normalize_no_punc(split[0]), split[1]] )
        # pairs.append( [util.normalize_no_punc(split[0]), split[2]] )

    # reverse pairs if needed, make lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        source_lang = Lang()
        target_lang = Lang()
    else:
        source_lang = Lang()
        target_lang = Lang()

    return source_lang, target_lang, pairs


def prepareData(filepath, reverse=False):

    # reading input file, obtaib lang names, initiate lang instances
    source_lang, target_lang, pairs = readLang(filepath, reverse)
    print('read {} sentence pairs.'.format(len(pairs)))

    # dummy filtering process, fastening the dev check
    # pairs = filterPairs(pairs)
    # print('reduced to {} sentence pairs.'.format(len(pairs)))

    # mapping vocabs/words in each languages into indexes
    print('counting words...')
    for pair in pairs:
        source_lang.addSentence(pair[0])
        target_lang.addSentence(pair[1])

    # for checking / logging purpose
    print('counted words:')
    print('> {0}: {1}'.format('source', source_lang.n_words))
    print('> {0}: {1}'.format('target', target_lang.n_words))

    return source_lang, target_lang, pairs

def buildPairs(filepath) :
    # read the file and split into lines
    lines = open(filepath, encoding='utf-8').read().strip().split('\n')

    # split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    print('read {} sentence pairs.'.format(len(pairs)))

    # dummy filtering process, fastening the dev check
    pairs = filterPairs(pairs)
    print('reduced to {} sentence pairs.'.format(len(pairs)))

    return pairs