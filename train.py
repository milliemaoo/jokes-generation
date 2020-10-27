import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import logging
import random
import math
import os
from tqdm import tqdm
from model import EncoderRNN,DecoderRNN
from config import SOS_token,EOS_token,PAD_token,MAXLEN,teacher_forcing_ratio

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# one iteration
def train_iteration(input_batch_tensor,input_len,target_batch_tensor,max_target_len,mask,keywords_batch_tensor,encoder,decoder,
	encoder_optimizer,decoder_optimizer,batch_size,use_ATTN,max_length=MAXLEN):
	# Gradient reset
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_batch_tensor = input_batch_tensor.to(device)
	target_batch_tensor = target_batch_tensor.to(device)
	keywords_batch_tensor = keywords_batch_tensor.to(device)
	mask = mask.to(device)

	loss = 0
	print_losses = []
	n_totals = 0

	encoder_outputs,(encoder_h,encoder_c) = encoder(input_batch_tensor,input_len,None)
	# convert input
	decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)])
	decoder_input = decoder_input.to(device)
	decoder_hidden = (encoder_h[:decoder.n_layers],encoder_c[:decoder.n_layers]) # because of the bidirection

	#use_teach_forcing = True if random.random() < teacher_forcing_ratio else False
	use_teach_forcing = False

	# use teach forcing
	if use_teach_forcing:
		for t in range(max_target_len):
			if use_ATTN:
				decoder_output,decoder_hidden,_ = decoder(encoder_outputs,decoder_input,keywords_batch_tensor,decoder_hidden)
			else:
				decoder_output,decoder_hidden = decoder(encoder_outputs,decoder_input,keywords_batch_tensor,decoder_hidden)
			decoder_input = target_batch_tensor[:,t].view(1,-1)
			loss += F.cross_entropy(decoder_output,target_batch_tensor[:,t], ignore_index=EOS_token)					
	# without forcing
	else:
		for t in range(max_target_len):
			if use_ATTN:					
				decoder_output,decoder_hidden,_ = decoder(encoder_outputs,decoder_input,keywords_batch_tensor,decoder_hidden)
			else:
				decoder_output,decoder_hidden = decoder(encoder_outputs,decoder_input,keywords_batch_tensor,decoder_hidden)					
			_,topi = decoder_output.topk(1)					
			decoder_input = torch.LongTensor([[topi[i][0]] for i in range(batch_size)])
			decoder_input = decoder_input.to(device)
			loss += F.cross_entropy(decoder_output, target_batch_tensor[:,t], ignore_index=EOS_token)

	loss.backward()

	# gradient clip
	gradient_clip = 50.0
	_ = torch.nn.utils.clip_grad_norm_(encoder.parameters(),gradient_clip)
	_ = torch.nn.utils.clip_grad_norm_(decoder.parameters(),gradient_clip)

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item()/max_target_len



def train(load_pretrain,src_voc,tag_voc,pair_batches,n_iteration,learning_rate,batch_size,n_layers,input_size,hidden_size,print_every,
			save_every,dropout,src_embeddings,tag_embeddings,rnn_type='LSTM',bidirectional=True,use_ATTN=True,decoder_learning_ratio=1.0):
	"""
		src_voc:source vocabulary
		tag_voc:target vocabulary
	"""
	checkpoint = None
	# build the model
	source_embedding = nn.Embedding(src_voc.n_words,input_size)
	source_embedding.weight = nn.Parameter(src_embeddings)
	target_embedding = nn.Embedding(tag_voc.n_words,input_size)
	target_embedding.weight = nn.Parameter(tag_embeddings)

	#whether fine-tune embedding
	for p in source_embedding.parameters():
		p.requires_grad = False

	for p in target_embedding.parameters():
		p.requires_grad = False

	print("INFO:Building Encoder and Decoder ... ")
	encoder = EncoderRNN(source_embedding,input_size,hidden_size,n_layers,bidirectional=bidirectional,dropout=dropout,rnn_type=rnn_type)
	decoder = DecoderRNN(target_embedding,input_size,hidden_size,tag_voc.n_words,n_layers,rnn_type=rnn_type,use_ATTN=use_ATTN,dropout=dropout)
	
	#load pre-training 
	if load_pretrain!=None:
		checkpoint = torch.load(load_pretrain)
		encoder.load_state_dict(checkpoint['en'])
		decoder.load_state_dict(checkpoint['de'])
	# train on GPU
	encoder = encoder.to(device)
	decoder = decoder.to(device)

	print("INFO:Building optimizers ...")
	encoder_optimizer = optim.Adam(encoder.parameters(),lr=learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(),lr=learning_rate*decoder_learning_ratio)

	# load pre-training
	if load_pretrain!=None:
		encoder_optimizer.load_state_dict(checkpoint['en_opt'])
		decoder_optimizer.load_state_dict(checkpoint['de_opt'])


	print("INFO:Initializing...")

	start_iteration = 1
	perplexity = []
	print_loss = 0
	# logger
	logger = logging.getLogger()
	logger.setLevel(level = logging.INFO)
	handler = logging.FileHandler("log.txt")
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	if load_pretrain!=None:
		start_iteration = checkpoint['iteration'] + 1
		perplexity = checkpoint['plt']

	# iteration
	for iteration in tqdm(range(start_iteration,n_iteration+1)):
		#
		training_batch = pair_batches[iteration-1]
		input_batch_tensor,input_len,output_batch_tensor,max_target_len,mask,keywords_batch_tensor = training_batch

		loss = train_iteration(input_batch_tensor,input_len,output_batch_tensor,max_target_len,mask,keywords_batch_tensor,encoder,
			decoder,encoder_optimizer,decoder_optimizer,batch_size,use_ATTN=use_ATTN)

		print_loss += loss
		perplexity.append(loss)

		if iteration % print_every == 0:
			print_loss_average = math.exp(print_loss/print_every)
			print('INFO:loss %d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_average))
			logger.info('INFO:loss %d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_average))
			print_loss = 0.0

		# save model
		if (iteration % save_every == 0):
			directory = os.path.join('model_param','{}_{}_{}'.format(n_layers,n_layers,hidden_size))
			if not os.path.exists(directory):
				os.makedirs(directory)
			torch.save(
				{
					'iteration':iteration,
					'encoder':encoder.state_dict(),
					'decoder':decoder.state_dict(),
					'encoder_optim':encoder_optimizer.state_dict(),
					'decoder_optim':decoder_optimizer.state_dict(),
					'loss':loss,
					'plt':perplexity
				},os.path.join(directory,'{}_{}.tar'.format(n_iteration,'seq2seq_bidir_model'))
				)
