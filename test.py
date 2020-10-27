# test
import torch
from evaluate import runTest

if __name__ == '__main__':
	#load path
	modelFile = "./model_param/4_4_512/30000_seq2seq_bidir_model_test.tar"
	corpus_name = "test_data"
	# The test parameters are consistent with the training model
	n_layers = 4
	input_size = 300
	hidden_size = 512
	beam_size = 4 
	runTest(n_layers,input_size,hidden_size,modelFile,beam_size,corpus_name)