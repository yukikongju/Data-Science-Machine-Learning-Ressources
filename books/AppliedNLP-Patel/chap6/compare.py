import torch

from torch import optim, nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


from rnn import BidirectionalRNN, DummyRNN, RNNEmbedding, RNNEmbeddingFlatten
from lstm import SimpleLSTM, BidirectionalLSTM
from gru import SimpleGRU, BidirectionalGRU



### Step 1: Create dummy data


# generated from https://www.poem-generator.org.uk/

sentence = """Whose ant is that? I think I know.
Its owner is quite sad though.
It really is a tale of woe,
I watch her frown. I cry hello.

She gives her ant a shake,
And sobs until the tears make.
The only other sound's the break,
Of distant waves and birds awake.

The ant is goofy, humped and deep,
But she has promises to keep,
Until then she shall not sleep.
She lies in bed with ducts that weep.

She rises from her bitter bed,
With thoughts of sadness in her head,
She idolises being dead.
Facing the day with never ending dread.
With thanks to the poet, Robert Frost, for the underlying structure.""".split()

def get_ngrams(sentence, n):
    n_grams = [ (sentence[i-n: i], sentence[i]) for i in range(n, len(sentence)) ]
    return n_grams
    
CONTEXT_SIZE = 3
n_grams = get_ngrams(sentence, CONTEXT_SIZE)
vocab = set(sentence)
word_idx = { word: idx for idx, word in enumerate(vocab) }

#  print(vocab)
#  print(n_grams)

### Step 2: separate into train and test data

def get_index_from_tokens(tokens):
    """ 
    tokens = ['her', 'bitter', 'bed']
    """
    return [word_idx.get(token) for token in tokens]
    

X = [ get_index_from_tokens(ngram[0]) for ngram in n_grams ]
y = [ word_idx.get(ngram[1]) for ngram in n_grams ]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)


### Initialize model, loss_fn, optimizer

model = DummyRNN(CONTEXT_SIZE, CONTEXT_SIZE, len(vocab))
optimizer = optim.SGD(model.parameters(), lr=1e-3)
loss_fn = nn.NLLLoss()
n_epochs = 10 


def training(n_epochs, model, loss_fn, optimizer, x_train, y_train):
    """
    
    """
    losses = []
    for epoch in range(n_epochs):
        for tokens, label in zip(x_train, y_train):
            t = torch.tensor(tokens).unsqueeze(1)   # FIXME
            print(t.shape)
            out = model(t)
            loss = loss_fn(out, torch.tensor(['label']))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.append(loss)
        print(f"Epoch: {epoch}, Loss: {loss}")

    return losses

training(n_epochs, model, loss_fn, optimizer, x_train, y_train)


def testing(model, x_test, y_test):
    with torch.no_grad():
        y_pred = []
        for token, label in zip(x_test, y_test):
            out = model(token)
            _, label_pred = torch.max(out, dim=1)
            y_pred.append(label_pred)

        print(classification_report(y_test, y_pred))


    





