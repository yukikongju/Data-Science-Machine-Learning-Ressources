import torch
import pytest

from models.seq2seq.encoder import Encoder


@pytest.fixture
def sequences() -> torch.Tensor:
    VOCAB_SIZE, SEQ_LENGTH = 1000, 25
    N = 5
    return torch.randint(0, VOCAB_SIZE, size=(N, SEQ_LENGTH))

def test_encoder(sequences: torch.Tensor):
    N, SEQ_LENGTH = sequences.size()
    VOCAB_SIZE = 1000
    EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS = 10, 4, 2
    inputs = sequences.transpose(1, 0)
    encoder = Encoder(input_size=VOCAB_SIZE, embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, p=0.4)
    out, hidden, cell = encoder(inputs)
    assert hidden.size() == cell.size() == (NUM_LAYERS, N, HIDDEN_SIZE)
    assert out.size() == (SEQ_LENGTH, N, HIDDEN_SIZE)


