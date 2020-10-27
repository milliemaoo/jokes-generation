# QA jokes generation
This is a seq2seq question-answer jokes generation inspired by [seq2bf](http://cn.arxiv.org/pdf/1607.00970). It is able to predict a keyword representing the gist of the reply from the questions and then incorporate this additional information into the generation to get a humorous reply.

## Corpus
You can get the corpus from [QA jokes](https://www.kaggle.com/bfinan/jokes-question-and-answer)  
Prepare your query, reply, and cue words file, with one instance per line.

## Usage
 * You can train and inference by start_train.py and start_test.py, change the argument values in your own need.

 * Note in the test, the post is entered by terminal, each post need a keyword which is decided by yourself, you can also refer to toolkits.py to automatically select a keyword based on post sentence.

## Example

    q: What will happen if you went inside a black hole?    
    k: world
    a: I don't know either. It must be out of this world.  

## References

[Seq2BF for Dialogue Generation](https://github.com/MaZhiyuanBUAA/Seq2BFforDialogueGeneration)

[seq2seqchatbot](https://github.com/ywk991112/pytorch-chatbot)

It's a preliminary version now.
