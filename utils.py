import os
import random
import itertools
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from config import SOS_token,EOS_token,PAD_token,REVERSE
from data import *
from gensim.models import KeyedVectors

# Mask pad
def binaryMatrix(output_batch):
	matrix=[]
	for i,seq in enumerate(output_batch):
		matrix.append([])
		for word in seq:
			if word == PAD_token:
				matrix[i].append(0)
			else:
				matrix[i].append(1)
	return matrix				

# sentence index, add EOS
def sentence2index(voc,sentence):
	return [voc.word2index[word] for word in word_tokenize(sentence)] + [EOS_token];

#keywords index.
def Word2index(voc,keywordlist):
	keywords_tensor = [voc.word2index[word] for word in keywordlist]
	return keywords_tensor

# pad
def padding(l,fillValue=PAD_token):
	padded_input = list(itertools.zip_longest(*l,fillvalue=fillValue))
	return padded_input

# input data 
def prepareInput(voc,input_data):

	#index
	input_indexes = [sentence2index(voc,sentence) for sentence in input_data]
	# record length
	input_len = [len(word_tokenize(sentence_indexes)) for sentence_indexes in input_data]
	input_len = torch.LongTensor(input_len)
	# padding
	padded_input = padding(input_indexes)
	# tensor
	input_tensor = torch.LongTensor(padded_input)
	input_tensor = input_tensor.transpose(0,1)
	return input_tensor,input_len

# output data
def prepareOutput(voc,output_data):

	# index
	output_indexes = [sentence2index(voc,sentence) for sentence in output_data]
	# record length
	output_maxlen = max([len(word_tokenize(sentence_indexes)) for sentence_indexes in output_data])
	# padding
	padded_output = padding(output_indexes)
	# record pad
	mask = binaryMatrix(padded_output)
	mask = torch.ByteTensor(mask)
	mask = mask.transpose(0,1)
	# Tensor
	output_tensor = torch.LongTensor(padded_output)
	output_tensor = output_tensor.transpose(0,1)

	return output_tensor,mask,output_maxlen

# prepare batch input tensor
def batch2TrainData(voc_src,voc_tag,pair_batch):

	# pair sort
	pair_batch.sort(key=lambda x:len(x[0].split(" ")),reverse = True)
	input_data = []
	output_data = []
	keywords = []

	# split
	for i in pair_batch:
		input_data.append(i[0])
		output_data.append(i[1])
		keywords.append(i[2])

	# prepare for input,output,keywords
	input_tensor,input_len = prepareInput(voc_src,input_data)
	output_tensor,mask,max_target_len = prepareOutput(voc_tag,output_data)
	keywords_tensor = Word2index(voc_tag,keywords)
	keywords_tensor = torch.LongTensor(keywords_tensor)

	return input_tensor,input_len,output_tensor,max_target_len,mask,keywords_tensor

def loadTraingingData(voc_src,voc_tag,pairs,batch_size,n_iterations):

	try:
		print("INFO:Start loading all_training_batches...")
		pair_batches = torch.load(os.path.join('training_batch',
						'{}_{}_{}.tar'.format(n_iterations,\
							'all_training_batches',batch_size)))
	except FileNotFoundError:

		print("INFO:All_traning_batches have not prepared! Start preparing training_batches...")
	# 准备n_iterations个训练mini-batch数据，每个大小为batch_size
	# 且迭代次数是n_iterations，因而每次迭代是一个mini-batch
		pair_batches = [batch2TrainData(voc_src,voc_tag,[random.choice(pairs) for _ in range(batch_size)])
							 for _ in range(n_iterations)]
		torch.save(pair_batches,os.path.join('training_batch',
					'{}_{}_{}.tar'.format(n_iterations,\
						'all_training_batches',batch_size)))
	return pair_batches

def init_embedding(input_embedding):
	"""
	Initialize embedding tensor with values from the uniform distribution.

	:param input_embedding: embedding tensor
	:return:
	"""
	bias = np.sqrt(3.0 / input_embedding.size(1))
	nn.init.uniform_(input_embedding, -bias, bias)


def load_embeddings(emb_file,voc,input_size):
	"""
	Load pre-trained embeddings for words in the voc, it's convenient for training
	"""
	# model = KeyedVectors.load_word2vec_format(emb_file)
	# print("word2vec load succeed")
	# vocabulary_vector = {}
	# for key in voc.word2index:
	# 	if key in model:
	# 		vocabulary_vector[key] = model[key]
	# pd.DataFrame(vocabulary_vector).to_csv("./embeddings/glove_300_tag_vocabulary.csv")
	# print("wordvectormatrix save succeed")

	vocabulary_vector = dict(pd.read_csv(emb_file,index_col=0))
	embeddings = torch.FloatTensor(len(voc.word2index), input_size)
	init_embedding(embeddings)
	# Read embedding file
	print("\nLoading embeddings...")
	
	for key,value in vocabulary_vector.items():
		embeddings[voc.word2index[key]] = torch.FloatTensor(value)

	# Sanity check
	assert embeddings.size(0) == len(voc.word2index)


	print("\nDone.\n Embedding vocabulary: %d\n" % (len(voc.word2index)))

	return embeddings


