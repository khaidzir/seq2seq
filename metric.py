from util import normalize_no_punc

def print_conf_matrix(writer, matrix) :
    for i in range(len(matrix)) :
        for j in range(len(matrix[i])) :
            writer.write("%d\t"%(matrix[i][j]))
        writer.write("\n")

def print_prec_rec(writer, vector) :
    for i in range(len(vector)) :
        writer.write("%d : %.4f\n"%(i,vector[i]))

def get_stats(conf_mat) :
    # Get n data
    n_data = sum([sum(row) for row in conf_mat])
    n_class = len(conf_mat)

    # Calculate accuracy
    acc = 0
    for i in range(n_class) : 
        acc += conf_mat[i][i]
    if n_data == 0 :
        acc = 0
    else :
        acc /= n_data

    # Calculate precision
    prec = [0 for _ in range(n_class)]
    for i in range(n_class) :
        div_prec = 0
        for j in range(n_class) :
            div_prec += conf_mat[i][j]
        if div_prec == 0 :
            prec[i] = 0
        else :
            prec[i] = conf_mat[i][i]/div_prec

    # Calculate recall
    rec = [0 for _ in range(n_class)]
    for i in range(n_class) :
        div_rec = 0
        for j in range(n_class) :
            div_rec += conf_mat[j][i]
        if div_rec == 0 :
            rec[i] = 0
        else :
            rec[i] = conf_mat[i][i]/div_rec

    return acc,prec,rec

def equal_sentence(sent1, sent2) :
    if len(sent1) != len(sent2) :
        return False
    for i in range(len(sent1)) :
        if sent1[i] != sent2[i] :
            return False
    return True

idx2class = {
    0: 'Mohon maaf jika aku ada salah. Bantu aku perbaiki kesalahanku ya.',
    1: 'Senang bisa membantu kamu. Silakan jangan ragu untuk tanya aku lagi jika kamu ingin mencari berita.',
    2: 'Selamat siang',
    3: 'Terima kasih feedbacknya, kami Beritagar berkomitmen untuk selalu meningkatkan kualitas berita kami',
    4: 'Informasi lebih lanjut dapat dilihat di tombol paling kanan atas atau kontak redaksi@beritagar.id',
    5: 'Hi, bot beritagar disini siap membantu kamu mencari berita-berita yang paling oke hanya dengan cukup tanyakan topik berita yang ingin kamu lihat.',
    6: 'Selamat malam',
    7: 'ga bisa, semoga kelak di masa depan aku bisa ya',
    8: 'Maaf aku tidak tahu. Bot Beritagar hanya melayani pencarian berita',
    9: 'Silakan',
    10: 'Baik, bot beritagar siap membantu kamu mencari berita',
    11: 'Maaf aku tidak bisa. Bot Beritagar hanya melayani pencarian berita',
    12: 'Aku tidak mengerti yang kamu ucapkan. Coba lagi ya.',
    13: 'Selamat sore',
    14: 'Selamat pagi',
    15: 'Baca berita aja. Bot beritagar disini siap membantu kamu mencari berita-berita yang paling oke hanya dengan cukup tanyakan topik berita yang ingin kamu lihat',
    16: 'Terima kasih',
    17: 'Ada, silakan ketik kata kunci dari berita yang diinginkan',
    18: 'Maaf aku tidak mengerti. Bot Beritagar hanya melayani pencarian berita'
}

class2idx = {
    'Mohon maaf jika aku ada salah. Bantu aku perbaiki kesalahanku ya.' : 0,
    'Senang bisa membantu kamu. Silakan jangan ragu untuk tanya aku lagi jika kamu ingin mencari berita.' : 1,
    'Selamat siang' : 2,
    'Terima kasih feedbacknya, kami Beritagar berkomitmen untuk selalu meningkatkan kualitas berita kami' : 3,
    'Informasi lebih lanjut dapat dilihat di tombol paling kanan atas atau kontak redaksi@beritagar.id' : 4,
    'Hi, bot beritagar disini siap membantu kamu mencari berita-berita yang paling oke hanya dengan cukup tanyakan topik berita yang ingin kamu lihat.' : 5,
    'Selamat malam' : 6,
    'ga bisa, semoga kelak di masa depan aku bisa ya' : 7,
    'Maaf aku tidak tahu. Bot Beritagar hanya melayani pencarian berita' : 8,
    'Silakan' : 9,
    'Baik, bot beritagar siap membantu kamu mencari berita' : 10,
    'Maaf aku tidak bisa. Bot Beritagar hanya melayani pencarian berita' : 11,
    'Aku tidak mengerti yang kamu ucapkan. Coba lagi ya.' : 12,
    'Selamat sore' : 13,
    'Selamat pagi' : 14,
    'Baca berita aja. Bot beritagar disini siap membantu kamu mencari berita-berita yang paling oke hanya dengan cukup tanyakan topik berita yang ingin kamu lihat' : 15,
    'Terima kasih' : 16,
    'Ada, silakan ketik kata kunci dari berita yang diinginkan' : 17,
    'Maaf aku tidak mengerti. Bot Beritagar hanya melayani pencarian berita' : 18
}

conf_matrix = [ [0 for _ in range(len(idx2class))] for c in range(len(idx2class)) ]

hyp_file = 'test/chatbot_new/word2vec/skipgram/combined_sgram/nontask_charembed'
ref_file = '/home/prosa/Works/Text/korpus/chatbot_dataset/plain/preprocessed/split/test-nontask.csv'
fout = 'test/chatbot_new/word2vec/skipgram/combined_sgram/nontask_charembed_confmat'

refs = []
hyps = []
with open(ref_file) as f :
    for line in f :
        line = line.strip()
        split = line.split('\t')
        refs.append(normalize_no_punc(split[2]).split())

with open(hyp_file) as f :
    for line in f :
        line = line.strip()
        hyps.append(normalize_no_punc(line).split())
hyps = hyps[:-1]

assert(len(refs) == len(hyps))

# for i in range(len(hyps)) :
#     conf_matrix[class2idx[hyps[i]]][class2idx[refs[i]]] += 1

# acc,prec,recall = get_stats(conf_matrix)
# fout = open(fout, 'w')
# fout.write(str(idx2class))
# fout.write("\n\nAccuracy : %.4f"%(acc))
# fout.write("\n\nPrecision :\n")
# print_prec_rec(fout, prec)
# fout.write("\n\nRecall :\n")
# print_prec_rec(fout, recall)
# fout.write("\n\nConfusion matrix :\n")
# print_conf_matrix(fout, conf_matrix)
# fout.close()

hit = 0
for i in range(len(refs)) :
    if equal_sentence(refs[i], hyps[i]) :
        hit += 1
print("Akurasi : %.4f\n"%(hit/len(refs)))
