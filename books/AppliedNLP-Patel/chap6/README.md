# Chap 6 - Recurrent Neural Networks and Other Sequence Models

[Code](https://github.com/nlpbook/nlpbook/blob/main/ch06.ipynb)

## Contents

- [o] Sequence Modeling using
    - [X] Reccurent Neural Neworks (RNNs)
	- [X] Dummy RNN (no embedding)
	- [X] RNN Embedding flatten (?)
	- [X] Bidirectionals RNNs
    - [X] Long-Short Term Memory (LSTMs)
	- [X] Simple LSTM
	- [X] Bidirectional LSTM
    - [ ] Gated Recurrent Units (GRUs)
- [ ] More Models
    - [ ] AWD-LSTMs
    - [ ] QRNNs
    - [ ] SHA-RNNs
- [ ] Training


**To Research**
- [ ] Why is there a difference between RNNEmbeding and RNNEmbeddingFlatten?
- [ ] Should we use `reshape()` or `view()` to reshape? what dimension 
      should middle layer have?
- [ ] Why do we want to make RNN/LSTM bidirectional?
- [ ] In LSTM, why do we want `x, _ = self.lstm(x)[1]` instead of `x, _ = self.lstm(x)`


**Notes**

- The input size for RNN, LSTM and GRU are: 
    * `nn.RNN()` : `[1 x context_size x embed_dim]`
    * `nn.LSTM()` :
    * `nn.GRU()` :
- The output size of the last `nn.Linear()` layer should depend on the task 
  we want to accomplish:
    - predict next word: the last layer need to output the probability for 
      each word in the vocab, so we need `nn.Linear( .. , vocab_size )`
    - predict context: the last layer need to output the indices for the 
      dictionary, so we need `nn.Linear( .. , context_dim )`


## Ressources


