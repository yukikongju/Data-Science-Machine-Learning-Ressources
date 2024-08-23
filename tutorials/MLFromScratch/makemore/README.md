# makemore

Building the following languages models:
- [ ] Bigrams
- [ ] MLP
- [ ] RNN
- [ ] Transformers

**Prerequisites**

- Torch Generator
- Tensor Broadcasting
- Negative log Likelihood vs normalized neg log likelihood vs pytorch cross-entropy
- model smoothing to avoid zerero division
- one-hot encoding
- learning rate optimization + lr decay

Running the notebook: `python3 -m notebook`

## Bag of words

Idea:


Steps:
- [X] Probability dict of Bigrams as Tensor
- [X] Using multinomial to sample from the model
- [X] Basic Neural Neto with one layer

- Resource: [Andrej Kaparthy - The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&ab_channel=AndrejKarpathy)
- [code](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb)

## MLP

Idea:


Steps:
- [ ] Build dataset with block_size (number of letter used as context)
- [ ] Initialize embedding and its parameters
- [ ] Train Model with gradient descent + learning rate optimization (lr decay)
- [ ] visualize embedding

- Resource: [Andrej Kaparthy - ](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4&ab_channel=AndrejKarpathy)

## WaveNet


- Resource: [Andrej Kaparthy - makemore pt5](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)


# Resources


