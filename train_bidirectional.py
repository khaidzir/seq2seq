import time
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from utility.timehelper import timeSince
import params
import util

class Trainer:
    def __init__(self, pairs, encoder, decoder, teacher_forcing_r=0.5):
        assert encoder.max_length == decoder.max_length
        self.pairs = pairs
        self.teacher_forcing_r = teacher_forcing_r
        self.processed_pairs = None
        self.encoder = encoder
        self.decoder = decoder

    # Process input sentence so can be processed by encoder's forward
    def process_input(self, sentence) :
        input = sentence.split()
        if self.encoder.model_type == 'word_based' :
            return util.get_sequence_index(input, self.encoder.lang.word2index)
        elif self.encoder.model_type == 'pre_trained_embedding' :
            return input

    # Process pair so can be processed by encoder's forward
    def process_pair(self, pair) :
        if self.encoder.model_type == 'word_based' :
            return self.process_word_based_pair(pair)
        elif self.encoder.model_type == 'pre_trained_embedding' :
            return self.process_pre_trained_embedding_pair(pair)

    # Process pairs to index of words
    def process_word_based_pair(self, pair) :
        source = pair[0].split()
        target = pair[1].split()
        target.append(params.EOS_TOKEN)
        return [util.get_sequence_index(source, self.encoder.lang.word2index), util.get_sequence_index(target, self.decoder.lang.word2index)]

    def process_pre_trained_embedding_pair(self, pair) :
        source = pair[0].split()
        target = pair[1].split()
        target.append(params.EOS_TOKEN)
        target = util.get_sequence_index(target, self.decoder.lang.word2index)
        return [source, target]

    # Function to train data in general
    def trainOneStep(self, source_var, target_var, encoder_optimizer, decoder_optimizer, criterion):

        # encoder training side
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        source_len = len(source_var)
        target_len = len(target_var)

        encoder_outputs,encoder_hidden,projected_hidden = self.encoder(source_var)

        loss = 0

        # decoder training side
        decoder_input = Variable(torch.LongTensor([self.decoder.lang.word2index[params.SOS_TOKEN]]))
        if params.USE_CUDA :
            decoder_input = decoder_input.cuda()
        decoder_hidden = projected_hidden

        # probabilistic step, set teacher forcing ration to 0 to disable
        if random.random() < self.teacher_forcing_r:
            # use teacher forcing, feed target from corpus as the next input
            for de_idx in range(target_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                target = torch.tensor([target_var[de_idx]])
                if params.USE_CUDA :
                    target = target.cuda()
                loss += criterion(decoder_output, target)
                decoder_input = target_var[de_idx]
        else:
            # without forcing, use its own prediction as the next input
            for de_idx in range(target_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(topi)
                if de_idx >= len(target_var) :
                    print( str(len(target_var)) + ' - ' + str(de_idx) )
                target = torch.tensor([target_var[de_idx]])
                if params.USE_CUDA :
                    target = target.cuda()
                loss += criterion(decoder_output, target)
                if ni == self.decoder.lang.word2index[params.EOS_TOKEN]:
                    break

        # back propagation, optimization
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        # return loss.data[0] / target_len
        return loss.item() / target_len

    # main function, iterating the training process
    def train(self, n_iter, learning_rate=0.01, print_every=1000, epoch=1):

        start = time.time()
        print_loss_total = 0

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        self.processed_pairs = []
        for pair in self.pairs :
            self.processed_pairs.append(self.process_pair(pair))
        training_pairs = self.processed_pairs
        assert len(training_pairs) == n_iter
        # training_pairs = []
        # for i in range(n_iter):
        #     training_pairs.append(random.choice(self.processed_pairs))

        for ep in range(epoch) :
            for iter in range(1, n_iter+1):
                training_pair = training_pairs[iter-1]
                source_var = training_pair[0]
                target_var = training_pair[1]

                loss = self.trainOneStep(source_var, target_var,
                                        encoder_optimizer, decoder_optimizer,
                                        criterion)
                print_loss_total += loss

                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('{0} ({1} {2}%) {3:0.4f}'.format(
                        timeSince(start, iter/n_iter), iter, iter/n_iter*100,
                        print_loss_avg))

    # Function to train data in general
    def trainOneStepV2(self, source_var, target_var, criterion):

        # encoder training side
        source_len = len(source_var)
        target_len = len(target_var)

        encoder_outputs,encoder_hidden,projected_hidden = self.encoder(source_var)

        loss = 0

        # decoder training side
        decoder_input = Variable(torch.LongTensor([self.decoder.lang.word2index[params.SOS_TOKEN]]))
        if params.USE_CUDA :
            decoder_input = decoder_input.cuda()
        decoder_hidden = projected_hidden

        # probabilistic step, set teacher forcing ration to 0 to disable
        if random.random() < self.teacher_forcing_r:
            # use teacher forcing, feed target from corpus as the next input
            for de_idx in range(target_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                target = torch.tensor([target_var[de_idx]])
                if params.USE_CUDA :
                    target = target.cuda()
                loss += criterion(decoder_output, target)
                decoder_input = target_var[de_idx]
        else:
            # without forcing, use its own prediction as the next input
            for de_idx in range(target_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(topi)
                if de_idx >= len(target_var) :
                    print( str(len(target_var)) + ' - ' + str(de_idx) )
                target = torch.tensor([target_var[de_idx]])
                if params.USE_CUDA :
                    target = target.cuda()
                loss += criterion(decoder_output, target)
                if ni == self.decoder.lang.word2index[params.EOS_TOKEN]:
                    break

        # return loss.data[0] / target_len
        # return loss.item() / target_len
        return loss

    # main function, iterating the training process
    def train_batch(self, learning_rate=0.01, print_every=1000, epoch=1, batch_size=1):

        start = time.time()
        print_loss_total = 0

        # encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        # decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)

        encoder_optimizer = optim.RMSprop(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.RMSprop(self.decoder.parameters(), lr=learning_rate)

        criterion = nn.NLLLoss()

        self.processed_pairs = []
        for pair in self.pairs :
            self.processed_pairs.append(self.process_pair(pair))
        training_pairs = self.processed_pairs

        n_training = len(training_pairs)
        n_total = n_training * epoch
        progress = 0
        # Epoch loop
        for ep in range(epoch) :
            print("Epoch - %d/%d"%(ep+1, epoch))
            permutation = torch.randperm(n_training)

            # Train set loop
            for i in range(0, n_training, batch_size):

                # Batch data
                endidx = i+batch_size if i+batch_size<n_training else n_training
                indices = permutation[i:endidx]

                # Zero gradient
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss = 0

                # Batch loop
                for iter in range(0, len(indices)) :
                    training_pair = training_pairs[indices[iter]]
                    source_var = training_pair[0]
                    target_var = training_pair[1]

                    curr_loss = self.trainOneStepV2(source_var, target_var,
                                            criterion)
                    print_loss_total = curr_loss.item() / len(target_var)
                    loss += curr_loss
                    progress += 1

                    if progress % print_every == 0:
                        print_loss_avg = print_loss_total / print_every
                        print_loss_total = 0
                        print('{0} ({1} {2}%) {3:0.4f}'.format(
                            timeSince(start, progress/n_total), progress, progress/n_total*100,
                            print_loss_avg))

                # back propagation, optimization
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

    # evaluation section
    def evaluate(self, sentence):
        assert self.encoder.max_length == self.decoder.max_length

        # Convert sentence (words) to list of word index
        source_var = self.process_input(sentence)
        source_len = len(source_var)

        # Set training mode to false
        self.encoder.train(False)
        self.decoder.train(False)

        # Forward to encoder
        encoder_outputs,encoder_hidden, projected_hidden = self.encoder(source_var)

        # SOS_TOKEN as first input word to decoder
        decoder_input = Variable(torch.LongTensor([self.decoder.lang.word2index[params.SOS_TOKEN]]))
        if params.USE_CUDA :
            decoder_input = decoder_input.cuda()
        
        # Encoder projected hidden state as initial decoder hidden state
        decoder_hidden = projected_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(self.decoder.max_length, self.decoder.max_length)

        # Decoding iteration, stop until found EOS_TOKEN or max_length is reached
        for de_idx in range(self.decoder.max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[de_idx] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0].item()

            if ni == self.decoder.lang.word2index[params.EOS_TOKEN]:
                decoded_words.append(params.EOS_TOKEN)
                break
            else:
                decoded_words.append(self.decoder.lang.index2word[ni])

            decoder_input = Variable(topi)

        # Set back training mode
        self.encoder.train(True)
        self.decoder.train(True)

        return decoded_words, decoder_attentions[:de_idx+1]

    def evaluate_beam_search(self, sentence, beam_width):
        assert self.encoder.max_length == self.decoder.max_length

        # Convert sentence (list of words) to list of word index
        source_var = self.process_input(sentence)
        source_len = len(source_var)

        # Set training mode to false
        self.encoder.train(False)
        self.decoder.train(False)

        # Forward to encoder
        encoder_outputs,encoder_hidden = self.encoder(source_var)

        # SOS_TOKEN as first input word to decoder
        first_input = [self.decoder.lang.word2index[params.SOS_TOKEN]]
        
        # Encoder hidden state as initial decoder hidden state
        decoder_hidden = encoder_hidden

        global_list = []
        current_beam_width = beam_width
        maintained_list = [ (decoder_hidden, first_input, 0) ]
        while(current_beam_width > 0) :
            temp_list = []
            for item in maintained_list :
                temp_list += self.beam_search_one_step(encoder_outputs, item[0], item[1], item[2], current_beam_width)

            temp_list.sort(key=lambda tup: tup[2], reverse=True)
            temp_list = temp_list[:current_beam_width]

            maintained_list = []
            for item in temp_list :
                if item[1][-1] == self.decoder.lang.word2index[params.EOS_TOKEN] or len(item[1]) == self.decoder.max_length :
                    global_list.append(item)
                    current_beam_width -= 1
                else :
                    maintained_list.append(item)

        # Set back training mode
        self.encoder.train(True)
        self.decoder.train(True)

        global_list.sort(key=lambda tup: tup[2], reverse=True)
        return [ self.to_seq_words(item[1]) for item in global_list[:beam_width] ]

    # Beam search for one step
    # Return value is array of tuples (decoder_hidden_state, seq_word_idx, score)
    def beam_search_one_step(self, encoder_outputs, hidden_state, seq_word_idx, score, beam_width) :
        decoder_input = Variable(torch.LongTensor( [seq_word_idx[-1]] ))
        if params.USE_CUDA :
            decoder_input = decoder_input.cuda()
        decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, hidden_state, encoder_outputs)
        topv, topi = decoder_output.data.topk(beam_width)
        retval = []
        for i in range(beam_width) :
            new_seq_word_idx = seq_word_idx + [topi[0][i]]
            retval.append( (decoder_hidden, new_seq_word_idx, score+topv[0][i]) )
        return retval

    # Convert sequence of word index to sequence of words
    def to_seq_words(self, seq_word_idx) :
        return [ self.decoder.lang.index2word[idx] for idx in seq_word_idx ]
        
    def evaluateRandomly(self, n=10):
        for i in range(n):
            pair = random.choice(self.pairs)
            print('> {}'.format(pair[0]))
            print('= {}'.format(pair[1]))
            output_words, attentions = self.evaluate(pair[0])
            output_sentence = ' '.join(output_words)
            print('< {}'.format(output_sentence))
            print('')

    def evaluateTrainSet(self) :
        for pair in self.pairs :
            print('> {}'.format(pair[0]))
            print('= {}'.format(pair[1]))
            output_words, attentions = self.evaluate(pair[0])
            output_sentence = ' '.join(output_words)
            print('< {}'.format(output_sentence))
            print('')

    def evaluateAll(self):
        references = []
        outputs = []
        for i in range(10):
            pair = random.choice(self.pairs)
            references.append(pair[1])
            output_words, attentions = self.evaluate(encoder, decoder, pair[0])
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

    def evaluateFromTest(self, test_pairs):
        references = []
        outputs = []
        for pair in test_pairs:
            references.append(pair[1])
            output_words, attentions = self.evaluate(encoder, decoder, pair[0])
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

