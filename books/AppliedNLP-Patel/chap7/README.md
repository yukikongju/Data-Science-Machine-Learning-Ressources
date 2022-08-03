# Chap 7 -


## Contents

- [ ] Transformers with PyTorch
    - [ ] Multihead Attention
    - [ ] Masking Generation
- [ ] Scaled dot product



**To Investigate**
- [ ] What does a `multi-head` attention do?
    - [ ] What is `dot product attention`?
    - [ ] What is `Scaled Dot Product Attention`?
    - [ ] What is `Multi-Head Self-Attention`?
- [ ] How does the transformer separate inputs into queries, keys and values?
- [ ] How to configure attention span for each head?



**Notes**
- At the moment, transformers are mostly used for NLP and not for CV because 
  it takes `O(n^2)` to compute images
- Unlike RNN which runs sequentially, transformers process the data parallelly ie 
  each head process the input for their respective attention span and combine 
  their finding after
    




## Ressources

- [ ] [Transformer, Multi-head attention and masking on pytorch by wook](https://sungwookyoo.github.io/tips/study/Multihead_Attention/)
- [ ] [Transformers and multihead attention by UVADLC](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)


