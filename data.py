import torch
import re
import os
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from config import SOS_token,EOS_token,PAD_token,MAXLEN

MIN_COUNT = 1
# dictionary.
class Dictionary:
	def __init__(self,name):
		self.name = name
		self.word2index = {"SOS":0,"EOS":1,"PAD":2}
		self.index2word = {0:"SOS",1:"EOS",2:"PAD"}
		self.word2count = {}
		self.n_words = 3 #SOS,EOS,PAD

	def addSentence(self,sentence):
		for word in word_tokenize(sentence):
			self.addWord(word)

	def addWord(self,word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words  = self.n_words+1
		else:
			self.word2count[word] = self.word2count[word]+1


# judge the length 
def judgeLen(que,ans):
	if len(que.split(" ")) < MAXLEN and len(ans.split(" "))<MAXLEN:
		return True
	return False

# filter sequences according to maximum sequence length
def filterMaxLenPairs(question,answer,keywords):
	pairs_new = [(que,ans,key) for que,ans,key in zip(question,answer,keywords) if judgeLen(que,ans)]
	return pairs_new


def prepareData(fileq,filea,filek,corpus_name):
	voc_src = Dictionary('src_word')
	voc_tag = Dictionary('tag_word')
	#loading..
	question = [line.decode('utf-8').strip() for line in open(fileq, "rb").readlines()]
	answer = [line.decode('utf-8').strip() for line in open(filea, "rb").readlines()]
	keywords = [line.decode('utf-8').strip() for line in open(filek, "rb").readlines()]
	# filter the long sentences.
	pairs = filterMaxLenPairs(question,answer,keywords)
	# build dictionary
	for pair in pairs:
		voc_src.addSentence(pair[0])
		voc_tag.addSentence(pair[1])
		voc_tag.addWord(pair[2])
	print("source vocabulary Size:", voc_src.n_words)
	print("target vocabulary Size:", voc_tag.n_words)
	print("INFO:End Build vocabulary!")

	if not os.path.exists(corpus_name):
		os.makedirs(corpus_name)

	# save
	torch.save(voc_src,os.path.join(corpus_name,'{!s}.tar'.format('source_vocabulary')))
	torch.save(voc_tag,os.path.join(corpus_name,'{!s}.tar'.format('target_vocabulary')))
	torch.save(pairs,os.path.join(corpus_name,'{!s}.tar'.format('training_pairs')))

	return voc_src,voc_tag,pairs

# load prepared data.
def loadPreparedData(fileq,filea,filek,corpus_name):
	print("INFO:Training_data have not prepared,Start prepare training data and vocabulary...")
	voc_src,voc_tag,pairs = prepareData(fileq,filea,filek,corpus_name)

	print("INFO:End prepare training data and vocabulary!")

	return voc_src,voc_tag,pairs
