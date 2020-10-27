import torch
from data import loadPreparedData
from utils import loadTraingingData,load_embeddings
from train import train

if __name__ == '__main__':
# some hp 
    fileq = "question.txt" # corpus path
    filea = "answer.txt"
    filek = "cue_train.txt"
    corpus_name = "train_data"  #define the corpus name
    src_emb_file = "./embeddings/glove_300_src_vocabulary.csv"
    tag_emb_file = './embeddings/glove_300_tag_vocabulary.csv'
    batch_size = 64
    n_iteration = 30000
    learning_rate = 0.0001
    n_layers = 4
    input_size = 300
    hidden_size = 512
    print_every = 5
    save_every = 20
    bidir = True
    dropout = 0.1
    workers = 1  #number of workers for loading data in the DataLoader
    voc_src,voc_tag,pairs = loadPreparedData(fileq,filea,filek,corpus_name)
    train_batches = loadTraingingData(voc_src,voc_tag,pairs,batch_size,n_iteration)
    src_embeddings = load_embeddings(src_emb_file,voc_src,input_size)
    tag_embeddings = load_embeddings(tag_emb_file,voc_tag,input_size)

    print("INFO:Start ...")
    # add your training model path,if new = None
    #load_pretrain = '1000_seq2seq_bidir_model.tar'
    load_pretrain = None
    # train
    train(load_pretrain,voc_src,voc_tag,train_batches,n_iteration,learning_rate,batch_size,n_layers,input_size,hidden_size,print_every,
        save_every,dropout,src_embeddings,tag_embeddings,bidirectional=bidir,use_ATTN=True,decoder_learning_ratio=1.0)
    print("INFO:End ...")