# QA jokes generation
This is a seq2seq question-answer jokes generation inspired by [seq2bf](http://cn.arxiv.org/pdf/1607.00970). It is able to predict a keyword representing the gist of the reply from the questions and then incorporate this additional information into the generation to get a humorous reply.

## Corpus
You can get the corpus from [QA jokes](https://www.kaggle.com/bfinan/jokes-question-and-answer)  
Prepare your query, reply, and cue words file; one instance per line.

## Usage
Refer to toolkits.py to get your own keywords for test data.

You can train and inference by train.py and test.py, change the argument values in your own need.

## Example

    q: What will happen if you went inside a black hole?  
    a: I don't know either. It must be out of this world.    
    k: world

## References

[Seq2BF for Dialogue Generation](https://github.com/MaZhiyuanBUAA/Seq2BFforDialogueGeneration)

[seq2seqchatbot](https://github.com/fancyerii/blog-codes/tree/master/chatbot)
