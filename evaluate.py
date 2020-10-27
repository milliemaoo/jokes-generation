import torch
import random
from data import loadPreparedData,Dictionary
from config import SOS_token,EOS_token,MAXLEN
from utils import sentence2index,load_embeddings
from model import *
from collections import OrderedDict

MAX_LENGTH = MAXLEN
indexesFromSentence = sentence2index

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

"""
some refer to：https://github.com/ywk991112/pytorch-chatbot/blob/master/evaluate.py
"""
def normalizeString(s):
    # lower
    s = unicodeToAscii(s.lower().strip())
    # special rules to deal with . ! ?
    s = re.sub(r"([.!?])", r" \1", s)
    # filter special symbol with whitespaces
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # multiple whitespaces to single whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s


class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        # return mean of sentence_score
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                terminates.append(([voc.index2word[idx.item()] for idx in self.sentence_idxes] + ['<EOS>'],
                                   self.avgScore())) # tuple(word_list, score_float
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<EOS>')
            else:
                words.append(voc.index2word[self.sentence_idxes[i].item()])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<EOS>')
        return (words, self.avgScore())

def beam_decode(decoder, decoder_hidden, encoder_outputs, keyword, voc, beam_size, max_length=MAX_LENGTH):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for i in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = torch.LongTensor([[sentence.last_idx]])
            decoder_input = decoder_input.to(device)
            decoder_hidden = sentence.decoder_hidden
            decoder_output, decoder_hidden, _ = decoder(encoder_outputs,decoder_input, keyword, decoder_hidden)
            topv, topi = decoder_output.topk(beam_size)
            term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size, voc)
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)
        next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []
    terminal_sentences += [sentence.toWordScore(voc) for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)
    n = min(len(terminal_sentences), 15)
    return terminal_sentences[:n]

def decode(decoder,decoder_hidden,encoder_outputs,keyword,voc,max_length=MAX_LENGTH):

    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_input = decoder_input.to(device)

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length) #TODO: or (MAX_LEN+1, MAX_LEN+1)

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            encoder_outputs,decoder_input,keyword,decoder_hidden
        )
        _, topi = decoder_output.topk(3)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(voc.index2word[ni.item()])

        decoder_input = torch.LongTensor([[ni]])
        decoder_input = decoder_input.to(device)

    return decoded_words, decoder_attentions[:di + 1]


def evaluate(encoder,decoder,src_voc,tag_voc,sentence,keyword,beam_size,max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(src_voc, sentence)] 
    lengths = [len(indexes) for indexes in indexes_batch]
    keyword_batch = [tag_voc.word2index[keyword]]
    input_batch = torch.LongTensor(indexes_batch)
    input_batch = input_batch.to(device)
    keyword_batch = torch.LongTensor(keyword_batch)
    keyword_batch = keyword_batch.to(device)
    encoder_outputs, (encoder_h,encoder_c) = encoder(input_batch,lengths,None)
    decoder_hidden = (encoder_h[:decoder.n_layers],encoder_c[:decoder.n_layers])

    if beam_size == 1:
        return decode(decoder, decoder_hidden, encoder_outputs, keyword_batch, tag_voc)
    else:
        return beam_decode(decoder, decoder_hidden, encoder_outputs, keyword_batch, tag_voc, beam_size)

# def evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, n=10):
#     for _ in range(n):
#         pair = random.choice(pairs)
#         print("=============================================================")
#         if reverse:
#             print('>', " ".join(reversed(pair[0].split())))
#         else:
#             print('>', pair[0])
#         if beam_size == 1:
#             output_words, _ = evaluate(encoder, decoder, voc, pair[0], beam_size)
#             output_sentence = ' '.join(output_words)
#             print('<', output_sentence)
#         else:
#             output_words_list = evaluate(encoder, decoder, voc, pair[0], beam_size)
#             for output_words, score in output_words_list:
#                 output_sentence = ' '.join(output_words)
#                 print("{:.3f} < {}".format(score, output_sentence))

def evaluateInput(encoder,decoder,src_voc,tag_voc,beam_size):
    question = ''
    while(1):
        try:
            question = input('q: ')
            keyword = input('k: ')
            if question == 'q': break
            if beam_size == 1:
                output_words, _ = evaluate(encoder,decoder,src_voc,tag_voc,question,keyword,beam_size)
                output_sentence = ' '.join(output_words)
                print('a:', output_sentence)
            else:
                output_words_list = evaluate(encoder,decoder,src_voc,tag_voc,question,keyword,beam_size)
                for output_words, score in output_words_list:
                    output_sentence = ' '.join(output_words)
                    print("{:.3f} < {}".format(score, output_sentence))
        except KeyError:
            print("Incorrect spelling.")


def runTest(n_layers,input_size,hidden_size,modelFile,beam_size,corpus_name):
    checkpoint = None

    # load word parameter 
    checkpoint = torch.load(modelFile,map_location='cpu')
    src_voc = checkpoint['src_dict']
    tag_voc = checkpoint['tag_dict']
    source_embedding = nn.Embedding(src_voc.n_words,input_size)
    target_embedding = nn.Embedding(tag_voc.n_words,input_size)
    source_embedding.load_state_dict(checkpoint['src_embedding'])
    target_embedding.load_state_dict(checkpoint['tag_embedding'])

    #build
    encoder = EncoderRNN(source_embedding,input_size,hidden_size,n_layers,bidirectional=True,dropout=0.1,rnn_type='LSTM')
    decoder = DecoderRNN(target_embedding,input_size,hidden_size,tag_voc.n_words,n_layers,rnn_type='LSTM',use_ATTN=True,dropout=0.1)

    # load model
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.train(False)
    decoder.train(False)
    # to GPU
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    torch.set_grad_enabled(False)
    #evaluate
    evaluateInput(encoder,decoder,src_voc,tag_voc,beam_size)