class Lang:
    def __init__(self):
        self.word2index = {'<SOS>': 0, '<EOS>': 1}
        self.word2count = {}
        self.index2word = {0: '<SOS>', 1: '<EOS>'}
        self.n_words = 2  # for indexing purpose? is it needed?

    def load_dict(self, attrDict) :
        self.word2index = attrDict['word2index']
        self.word2count = attrDict['word2count']
        self.index2word = attrDict['index2word']
        self.n_words = attrDict['n_words']

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def getAttrDict(self) :
        return {
            'word2index' : self.word2index,
            'word2count' : self.word2count,
            'index2word' : self.index2word,
            'n_words' : self.n_words
        }