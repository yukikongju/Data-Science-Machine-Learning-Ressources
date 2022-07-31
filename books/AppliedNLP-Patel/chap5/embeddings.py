import torch
import torch.nn.functional as F

from torch import nn, optim
from tqdm import tqdm

#  from nltk.tokenize import word_tokenize

#  https://www.youtube.com/watch?v=mCvW_qNm7rY&t=763s
#  https://github.com/johnsanterre/Teaching/blob/main/embedding_progression/1_embeddings_orginal.py

    
def get_ngrams(sentence, n):
    """ 
    Function that retrieves n-grams: map consecutive n words to next word

    Parameters
    ----------
    sentence: list of str
        sentence broken down by words (ie ['hello', 'world'])
    n: int
        num of grams
    
    Returns
    -------
    n_grams: list of tuples
        

    Examples
    --------
    >>> sentence = ['When', 'forty', 'winters', 'shall', 'besiege']
    >>> get_ngrams(sentence, 2)
    >>> [(['When', 'forty'], 'winters'), (['forty', 'winters'], 'shall'), (['winters', 'shall'], 'besiege')]
    """
    n_grams = [
        ( sentence[i - n: i ], sentence[i] )
        for i in range(n, len(sentence))
    ]
    return n_grams
    

class NGramsLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramsLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        #  x = self.embeddings(x).reshape(x.shape[0], -1) # why works with view, but not with reshape()?
        x = self.embeddings(x).view((1, -1))
        x = F.relu(self.linear1(x))
        x = F.log_softmax(x, dim=1)
        return x


def training(n_epochs, model, loss_fn, optimizer, n_grams, word_idx):
    """
    Get Loss History

    Parameters
    ----------
    n_epochs: int
    model: nn.Model
    optimizer: torch.optim
    n_grams: list of tuples
    word_idx: dict
        dictionary that maps word to its index (lookup table)
    """
    losses = []
    for epoch in range(n_epochs):
        for context, target in n_grams:
            context_idx = torch.tensor([word_idx[word] for word in context]) # dtype
            log_probs = model(context_idx)
            loss = loss_fn(log_probs, torch.tensor([word_idx[target]]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
    return losses


CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()


n_grams = get_ngrams(sentence, CONTEXT_SIZE)
vocab = set(sentence)
word_idx = { word: i for i, word in enumerate(vocab) }


n_epochs = 10
model = NGramsLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=1e-4)
loss_fn = nn.NLLLoss()

losses = training(n_epochs, model, loss_fn, optimizer, n_grams, word_idx)
print(losses)
    
        


