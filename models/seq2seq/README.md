# Seq2Seq Notes

- The Seq2Seq model is made of 2 modules:
    * Encoder: 
	+ Given N sentences of shape (N, seq_length), compress the input 
	  sequence into (SEQ_LENGTH, N, HIDDEN_SIZE)
	+ Intuition: Compress each word from each sentence
    * Decoder: 
	+ Given an input token of shape (N, ), predict the probability of 
          the next word (N, VOCAB_SIZE)
	+ Intuition: Given the summary of each sentence 

Steps of Encoder:
0. Input: (N, seq_length)
1. Embedding: (seq_length, N, hidden_size)
2. RNN/LSTM/GRU:
   - out: (seq_length, N, hidden_size)
   - hidden/cell: (num_layers, N, hidden_size)

Steps of Decoder
0. Input: (N, ) => (1, N)
1. Embedding: (1, N, embedding_size) 
2. RNN/LSTM/GRU: (1, N, hidden_size) => (N, hidden_size)
3. FC: (N, VOCAB_SIZE)

