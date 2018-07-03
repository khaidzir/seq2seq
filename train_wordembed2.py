import time
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from utility.timehelper import timeSince

SOS_token = 0
EOS_token = 1

use_cuda = torch.cuda.is_available()


class Trainer:
    def __init__(self, source_lang, target_lang, pairs, teacher_forcing_r=0.5):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.pairs = pairs
        self.teacher_forcing_r = teacher_forcing_r

    # Function for transforming sentence to sequence of indexes (based on dict)
    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def variableFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        var = Variable(torch.LongTensor(indexes).view(-1, 1))
        if use_cuda:
            return var.cuda()
        else:
            return var

    def variablesFromPair(self, pair):
        source_var = pair[0].split()
        target_var = self.variableFromSentence(self.target_lang, pair[1])
        return source_var, target_var

    # Function to train data in general
    def trainOneStep(self, source_var, target_var, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion,
                     max_len=10):

        # encoder training side
        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        source_len = len(source_var)
        target_len = target_var.size()[0]

        encoder_outputs = Variable(torch.zeros(max_len, encoder.hidden_size))
        if use_cuda:
            encoder_outputs = encoder_outputs.cuda()

        loss = 0

        for en_idx in range(source_len):
            encoder_output, encoder_hidden = encoder(source_var[en_idx],
                                                     encoder_hidden)
            encoder_outputs[en_idx] = encoder_output[0][0]

        # decoder training side
        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        # probabilistic step, set teacher forcing ration to 0 to disable
        if random.random() < self.teacher_forcing_r:
            # use teacher forcing, feed target from corpus as the next input
            for de_idx in range(target_len):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_var[de_idx])
                decoder_input = target_var[de_idx]
        else:
            # without forcing, use its own prediction as the next input
            for de_idx in range(target_len):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                if use_cuda:
                    decoder_input = decoder_input.cuda()

                loss += criterion(decoder_output, target_var[de_idx])
                if ni == EOS_token:
                    break

        # back propagation, optimization
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0] / target_len

    # main function, iterating the training process
    def train(self, encoder, decoder, n_iter, learning_rate=0.01,
              print_every=1000, max_len=10):

        start = time.time()
        print_loss_total = 0

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        training_pairs = []
        for i in range(n_iter):
            training_pairs.append(
                self.variablesFromPair(random.choice(self.pairs)))

        for iter in range(1, n_iter+1):
            training_pair = training_pairs[iter-1]
            source_var = training_pair[0]
            target_var = training_pair[1]

            loss = self.trainOneStep(source_var, target_var, encoder, decoder,
                                     encoder_optimizer, decoder_optimizer,
                                     criterion, max_len=max_len)
            print_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('{0} ({1} {2}%%) {3:0.4f}'.format(
                    timeSince(start, iter/n_iter), iter, iter/n_iter*100,
                    print_loss_avg))

    # evaluation section
    def evaluate(self, encoder, decoder, sentence, max_len=10):
        source_var = sentence.split()
        source_len = len(source_var)
        encoder_hidden = encoder.initHidden()

        encoder.train(False)
        decoder.train(False)

        encoder_outputs = Variable(torch.zeros(max_len, encoder.hidden_size))
        if use_cuda:
            encoder_outputs = encoder_outputs.cuda()

        for en_idx in range(source_len):
            encoder_output, encoder_hidden = encoder(source_var[en_idx],
                                                     encoder_hidden)
            encoder_outputs[en_idx] = \
                encoder_outputs[en_idx] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_len, max_len)

        for de_idx in range(max_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[de_idx] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(self.target_lang.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))
            if use_cuda:
                decoder_input = decoder_input.cuda()

        encoder.train(True)
        decoder.train(True)

        return decoded_words, decoder_attentions[:de_idx+1]

    def evaluateRandomly(self, encoder, decoder, n=10, max_len=10):
        for i in range(n):
            pair = random.choice(self.pairs)
            print('> {}'.format(pair[0]))
            print('= {}'.format(pair[1]))
            output_words, attentions = self.evaluate(encoder, decoder, pair[0], max_len=max_len)
            output_sentence = ' '.join(output_words)
            print('< {}'.format(output_sentence))
            print('')

    def evaluateAll(self, encoder, decoder, max_len=10):
        references = []
        outputs = []
        for i in range(10):
            pair = random.choice(self.pairs)
            references.append(pair[1])
            output_words, attentions = self.evaluate(encoder, decoder, pair[0], max_len=max_len)
            output_sentence = ' '.join(output_words)
            outputs.append(output_sentence)

        with open('result/reference.txt', 'w', encoding='utf-8') as f:
            for reference in references:
                f.write('{}\n'.format(reference))
        f.close()

        with open('result/output.txt', 'w', encoding='utf-8') as f:
            for output in outputs:
                f.write('{}\n'.format(output))
        f.close()

    def evaluateFromTest(self, test_pairs, encoder, decoder, max_len=10):
        references = []
        outputs = []
        for pair in test_pairs:
            references.append(pair[1])
            output_words, attentions = self.evaluate(encoder, decoder, pair[0], max_len=max_len)
            output_sentence = ' '.join(output_words)
            outputs.append(output_sentence)

        with open('result/reference.txt', 'w', encoding='utf-8') as f:
            for reference in references:
                f.write('{}\n'.format(reference))
        f.close()

        with open('result/output.txt', 'w', encoding='utf-8') as f:
            for output in outputs:
                f.write('{}\n'.format(output))
        f.close()

