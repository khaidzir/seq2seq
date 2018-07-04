class SimpleWordVector() :

    def __init__(self) :
        self.word_dict = {}
        self.vector_size = 0

    def __getitem__(self, key) :
        return self.word_dict[key]

    def __setitem__(self, key, value) :
        self.word_dict[key] = value

    def __contains__(self, item) :
        return item in self.word_dict
    