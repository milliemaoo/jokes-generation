import spacy
import re
from collections import Counter,defaultdict
import pandas as pd
import csv
import pickle
import numpy as np

#you can refer to this to obtain keywords list.
#you'd better clean your data first.:)

'''
build vocab and noun 
'''
def bulid_voc_ntable(datafile):
	nlp = spacy.load('en')
	sentences = [line.decode('utf-8').strip() for line in open(datafile,"rb").readlines()]
	vocab = Counter()
	noun = Counter()
	for sentence in sentences:
		pos = nlp(sentence)
		for ele in pos:
			vocab[str(ele)] += 1
			if ele.tag_ == 'NN':
				 noun[str(ele)] += 1
	noun = dict(noun)
	vocab = dict(vocab)
	dict_noun = noun.items()
	dict_noun = sorted(dict_noun,key=lambda x:x[1],reverse=True)
	dict_vocab = vocab.items()
	dict_vocab = sorted(dict_vocab,key=lambda x:x[1],reverse=True)
	f = open('nouns.txt','w',encoding='utf-8')
	f1 = open('vocab.txt','w',encoding='utf-8')
	for ele in dict_noun:
		if len(str(ele))>1:
			f.write(ele[0]+'\n')
	f.close()
	for ele in dict_vocab:
		f1.write(ele[0]+"\n")
	f1.close()

'''
build table
load your data,vocab and noun 
'''
def build_table(question,answer,vocab,nouns) 
	answers = []
	questions = []
	for line in open(question,"rb").readlines():
		questions.append(line.decode('utf-8').strip())
	for line in open(answer,"rb").readlines():
		answers.append(line.decode('utf-8').strip())

	tmp = []
	for w in vocab:
		for w_ in nouns:
			tmp.append(w+'|'+w_)
	cotable = dict([(ele,1) for ele in tmp])

	for qws,aws in zip(questions,answers):
		qws = qws.strip().split()
		aws = aws.strip().split()
		for ind,w in enumerate(aws):
			for w_ in qws:
				try:
					cotable[w_+'|'+w] += 1
				except:
					pass
	ntable = defaultdict(int)
	qtable = defaultdict(int)
	for k,ele in cotable.items():
		ntable[k.split('|')[1]] += ele
		qtable[k.split('|')[0]] += ele
	cotable = dict([(k,1.*v/ntable[k.split('|')[1]]) for k,v in cotable.items()])
	total = sum(qtable.values())
	qtable = dict([(k,1.*v/total) for k,v in qtable.items()])
	f = open('co_table.pkl','wb+')
	pickle.dump(cotable,f)
	f.close()
	# f = open('n_table.pkl','wb+')
	# pickle.dump(ntable,f)
	# f.close()
	f = open('q_table.pkl','wb+')
	pickle.dump(qtable,f)
	f.close()

# calculate PMI
def sentencePMI(sentence,cotable,qtable,nouns):
	ws = sentence.split()
	def pmi(wq):
		v = np.zeros(len(nouns))
		for ind,ele in enumerate(nouns):
			try:
				v[ind] = np.log((cotable[wq+'|'+ele])/(qtable[wq]))
			except:
				pass
		#print('middle result:%s,pmi:%f'%(nouns[np.argmax(v)],np.max(v)))
		return v
	vs = np.zeros(len(nouns))
	for w in ws:
		vs += pmi(w)
	return nouns[np.argmax(vs)]



if __name__ == '__main__':

# test_pmi
	datafile = "./dataset.txt"   # your train dataset.
	question = "./question.txt"
	answer = "./answer.txt"
	bulid_voc_ntable(datafile) 
	vocab = [ele.decode('utf-8').strip() for ele in open("vocab_5000.txt","rb").readlines()]
	nouns = [ele.decode('utf-8').strip() for ele in open("nouns_2500.txt","rb").readlines()]
	print('build table')
	build_table(question,answer,vocab,nouns)

	f = open('co_table.pkl','rb')
	cotable = pickle.load(f)
	f.close()
	print('cotable loaded')
	f = open('q_table.pkl','rb')
	qtable = pickle.load(f)
	f.close()
	print('qtable loaded')

# an example
	sentence = 'how do you eat an elephant'
	print(sentencePMI(sentence,cotable,qtable,nouns))







