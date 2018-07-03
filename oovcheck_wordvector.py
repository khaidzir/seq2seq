import params
from gensim.models import KeyedVectors

filetrain = "/home/prosa/Works/Text/korpus/chatbot_dataset/plain/preprocessed/split-augmented/perfile/train-nontask.aug"
word_vectors = KeyedVectors.load(params.WORD_VECTORS_FILE)

count = 0
n_oov = 0

with open(filetrain) as f :
    for line in f :
        line = line.strip().split('\t')
        for word in line[0].split() :
            count += 1
            if word not in word_vectors :
                n_oov += 1

print("Total oov : %d\n"%(n_oov))
print("Total kata : %d\n"%(count))

