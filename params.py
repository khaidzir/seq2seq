import torch

OOV_INDEX = -1
USE_CUDA = torch.cuda.is_available()
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
SEEDER = 88888888
CHAR_LENGTH = 50
# WORD_VECTORS_FILE = 'word_vector/prosa-w2v/prosa.vec'

# FASTTEXT CBOW
# WORD_VECTORS_FILE = '/home/prosa/Works/Text/tools/fastText-0.1.0/data/preprocessed_codot_article.bin'
# WORD_VECTORS_FILE = '/home/prosa/Works/Text/tools/fastText-0.1.0/data/preprocessed_twitter_cbow.bin'
# WORD_VECTORS_FILE = '/home/prosa/Works/Text/tools/fastText-0.1.0/data/codot_twitter_cbow.bin'

# WORD2VEC CBOW
WORD_VECTORS_FILE = '/home/prosa/Works/Text/word_embedding/word2vec/cbow/codot_combine_twitter_cbow.vec'
# WORD_VECTORS_FILE = '/home/prosa/Works/Text/word_embedding/word2vec/cbow/codot_cbow.vec'
# WORD_VECTORS_FILE = '/home/prosa/Works/Text/word_embedding/word2vec/cbow/twitter_cbow.vec'

# WORD2VEC SKIPGRAM
# WORD_VECTORS_FILE = '/home/prosa/Works/Text/word_embedding/word2vec/skipgram/codot_combine_twitter_sgram.vec'
# WORD_VECTORS_FILE = '/home/prosa/Works/Text/word_embedding/word2vec/skipgram/codot_sgram.vec'
# WORD_VECTORS_FILE = '/home/prosa/Works/Text/word_embedding/word2vec/skipgram/twitter_sgram.vec'

# WORD_VECTORS_FILE = '/home/prosa/Works/Text/word_embedding/en/GoogleNews-vectors-negative300.bin'
