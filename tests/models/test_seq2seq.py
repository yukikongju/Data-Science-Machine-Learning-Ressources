import torch
import pytest

from models.seq2seq.encoder import Encoder
from models.seq2seq.decoder import Decoder


@pytest.fixture
def sequences() -> torch.Tensor:
    VOCAB_SIZE, SEQ_LENGTH = 1000, 25
    N = 5
    return torch.randint(0, VOCAB_SIZE, size=(N, SEQ_LENGTH))

@pytest.fixture
def input_token() -> torch.Tensor:
    N, VOCAB_SIZE, SEQ_LENGTH = 5, 1000, 25
    return torch.randint(0, VOCAB_SIZE, size=(N, ))


def test_encoder(sequences: torch.Tensor):
    N, SEQ_LENGTH = sequences.size()
    VOCAB_SIZE = 1000
    EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS = 10, 4, 2
    inputs = sequences.transpose(1, 0)
    encoder = Encoder(input_size=VOCAB_SIZE, embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, p=0.4)
    out, hidden, cell = encoder(inputs)
    assert hidden.size() == cell.size() == (NUM_LAYERS, N, HIDDEN_SIZE)
    assert out.size() == (SEQ_LENGTH, N, HIDDEN_SIZE)

def test_decoder(input_token: torch.Tensor):
    N = input_token.size()[0]
    EMBEDDING_SIZE, NUM_LAYERS, HIDDEN_SIZE = 10, 2, 4
    VOCAB_SIZE = 1000

    hidden_in = torch.randn(NUM_LAYERS, N, HIDDEN_SIZE)
    cell_in = torch.randn(NUM_LAYERS, N, HIDDEN_SIZE)

    decoder = Decoder(input_size=VOCAB_SIZE, embedding_size=EMBEDDING_SIZE, 
                      hidden_size=HIDDEN_SIZE, output_size=VOCAB_SIZE, 
                      num_layers=NUM_LAYERS, p=0.1)
    output = decoder(input_token, hidden_in, cell_in)
    assert output.size() == (N, VOCAB_SIZE)


