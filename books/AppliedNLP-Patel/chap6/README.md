# Chap 6 - Recurrent Neural Networks and Other Sequence Models

[Code](https://github.com/nlpbook/nlpbook/blob/main/ch06.ipynb)

## Contents

- [o] Sequence Modeling using
    - [X] Dummy RNN (no embedding)
    - [X] Reccurent Neural Neworks (RNNs)
    - [X] RNN Embedding flatten (?)
    - [X] Bidirectionals RNNs
    - [ ] Long-Short Term Memory (LSTMs)
    - [ ] Gated Recurrent Units (GRUs)
- [ ] More Models
    - [ ] AWD-LSTMs
    - [ ] QRNNs
    - [ ] SHA-RNNs


**To Research**
- [ ] Why is there a difference between RNNEmbeding and RNNEmbeddingFlatten?
- [ ] Should we use `reshape()` or `view()` to reshape? what dimension 
      should middle layer have?


**Notes**

- The output size of the last `nn.Linear()` layer should depend on the task 
  we want to accomplish:
    - predict next word: the last layer need to output the probability for 
      each word in the vocab, so we need `nn.Linear( .. , vocab_size )`
    - predict context: the last layer need to output the indices for the 
      dictionary, so we need `nn.Linear( .. , context_dim )`


## Ressources


