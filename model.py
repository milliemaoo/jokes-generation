import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Attention
class AttentionModel(nn.Module):

	def __init__(self,hidden_size):
		super(AttentionModel,self).__init__()
		self.hidden_size = hidden_size
		self.attn = nn.Linear(2*self.hidden_size,self.hidden_size)
		self.v = nn.Parameter(torch.FloatTensor(1,self.hidden_size))

	# encoder_outputs from the output of Encoder，last_output from the previous output of Decoder
	def forward(self,encoder_outputs,rnn_hidden):

		encoder_seq_len = encoder_outputs.size(1)
		batch_size = rnn_hidden.size(0)

		energy = torch.zeros(batch_size,encoder_seq_len)
		energy = energy.to(device)

		# calculate
		for b in range(batch_size):
			for i in range(encoder_seq_len):
				# 512 + 512 = 1024
				tmp = torch.cat((rnn_hidden[b],encoder_outputs[b,i].unsqueeze(0)),1)
				#print('rnn_hidden_Size:',rnn_hidden[:b].size())
				#print('encoder_outputs_size:',encoder_outputs[i,b].size())
				#print('tmp_size:',tmp.size())
				tmp = self.attn(tmp) #512
				energy[b,i] = self.v.squeeze(0).dot(tmp.squeeze(0))

		attns = F.softmax(energy,dim=1).unsqueeze(1)
		return attns 

# EncoderRNN
class EncoderRNN(nn.Module):
	def __init__(self,embedding,input_size,hidden_size,n_layers=1,bidirectional=True,dropout=0.1,rnn_type='LSTM'):
		super(EncoderRNN, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.bidirectional = bidirectional
		self.dropout = dropout
		self.rnn_type = rnn_type
		self.embedding = embedding 
		# build RNN
		if self.rnn_type == 'LSTM':
			self.lstm = nn.LSTM(input_size,hidden_size,n_layers,bidirectional=self.bidirectional,
				dropout=(0 if n_layers==1 else dropout),batch_first=True)
		elif self.rnn_type == 'GRU':
			self.gru = nn.GRU(hidden_size,hidden_size,n_layers,
				dropout=(0 if self.n_layers==1 else dropout),bidirectional=self.bidirectional,batch_first=True)

	def forward(self,input_seq,input_lengths,hidden=None):

		# Sort by decreasing true word sequence length (when training!)
		input_lengths, word_sort_ind = input_lengths.sort(dim=0, descending=True)
		input_seq = input_seq[word_sort_ind]

		embedded = self.embedding(input_seq)
		# packed_sequence
		packed = torch.nn.utils.rnn.pack_padded_sequence(embedded,input_lengths,batch_first=True)
		# LSTM or GRU
		if self.rnn_type == 'LSTM':
			output,(h,c) = self.lstm(packed,None)
			output,_ = torch.nn.utils.rnn.pad_packed_sequence(output,batch_first=True) #[batch_size,seq_length,hidden_size*n_directions]
			output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:] #for bidirectional = True
			return output,(h,c)

		elif self.rnn_type == 'GRU':
			output,hidden = self.gru(packed,hidden)
			output,_ = torch.nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
			return output,hidden


# DecoderRNN
class DecoderRNN(nn.Module):
	def __init__(self,embedding,input_size,hidden_size,output_size,n_layers=4,rnn_type='LSTM',use_ATTN=True,dropout = 0.1):
		super(DecoderRNN,self).__init__()
		self.use_ATTN = use_ATTN
		self.embedding = embedding
		self.rnn_type = rnn_type
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		# RNN
		if self.rnn_type == 'LSTM':
			self.lstm = nn.LSTM(2*self.input_size,self.hidden_size,
				self.n_layers,dropout=(0 if n_layers == 1 else self.dropout),batch_first=True)
		elif self.rnn_type == 'GRU':
			self.gru = nn.GRU(2*self.input_size,self.hidden_size,
				self.n_layers,dropout=(0 if n_layers == 1 else self.dropout),batch_first=True)

		if self.use_ATTN == True:
			self.attn_layer = AttentionModel(hidden_size)
			self.attn_linear = nn.Linear(self.hidden_size*2+self.input_size,self.hidden_size)
		# without attention
		self.outLayer = nn.Linear(self.hidden_size,self.output_size)

	# input_seq: SOS
	# pay attention to last_hidden. GRU:Tensor，LSTM:(h0,c0)
	def forward(self,encoder_outputs,input_seq,keyword,last_hidden):

		SOS_embed = self.embedding(input_seq)
		key_embed = self.embedding(keyword)
		if SOS_embed.size(1) != 1:
			raise ValueError("Decoder Start seq_len should be 1 !")
		input_embed = torch.cat((SOS_embed,key_embed.unsqueeze(1)),-1)
		#print(last_hidden.size())
		# LSTM
		if self.rnn_type == 'LSTM':
			outputs,(h,c) = self.lstm(input_embed,last_hidden) #pay attention to last_hidden
			#print("LSTM_outpus size:",outputs.size())
			if self.use_ATTN == True:
				attn_weights = self.attn_layer(encoder_outputs,outputs)
				# [batch_size,1,hidden_size]
				context = attn_weights.bmm(encoder_outputs)
				outputs = outputs.squeeze(1) #[batch_size,hidden_size]
				context = context.squeeze(1) #[batch_size,hidden_size]
				# concat
				tmp = torch.cat((outputs,context,key_embed.squeeze(1)),1) # [batch_size,hidden_size*2]
				# output
				outputs_tmp = self.attn_linear(tmp) # [batch_size,hidden_size]
				outputs_tmp = torch.tanh(outputs_tmp)
				outputs_final = self.outLayer(outputs_tmp) # [batch_size,output_size]
				#print("Output：",outputs_final.size())
				return outputs_final,(h,c),attn_weights
			else:
				outputs = outputs.squeeze(1)
				outputs = self.outLayer(outputs) 
				return outputs,(h,c)
		# GRU
		else:
			outputs,h = self.gru(input_embed,last_hidden)
			if self.use_ATTN == True:
				attn_weights = self.attn_layer(encoder_outputs,outputs)
				# [batch_size,1,hidden_size]
				context = attn_weights.bmm(encoder_outputs)
				outputs = outputs.squeeze(1) #[batch_size,hidden_size]
				context = context.squeeze(1) #[batch_size,hidden_size]
				# concat
				tmp = torch.cat((outputs,context,key_embed.squeeze(1)),1) # [batch_size,hidden_size*2]
				# output
				outputs_tmp = self.attn_linear(tmp) # [batch_size,hidden_size]
				outputs_tmp = torch.tanh(outputs_tmp)
				outputs_final = self.outLayer(outputs_tmp) # [batch_size,output_size]
				return outputs_final,h,attn_weights

			else:
				outputs = outputs.squeeze(1)
				outputs = self.outLayer(outputs) # to target output size
				return outputs,h