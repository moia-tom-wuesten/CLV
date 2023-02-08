import torch
import torch.nn as nn
from src.CLV.pytorch_model.base_model import LSTMModel


def test_LSTMModel_init():
    max_weeks = 52
    max_trans = 12
    stateful = False
    hidden_size = 32

    model = LSTMModel(max_weeks, max_trans, stateful, hidden_size)

    assert model.max_weeks == max_weeks
    assert model.max_trans == max_trans
    assert model.stateful == stateful
    assert model.hidden_size == hidden_size
    assert isinstance(model.embdedding_week, nn.Embedding)
    assert isinstance(model.embedding_transaction, nn.Embedding)
    assert isinstance(model.lstm, nn.LSTM)
    assert isinstance(model.output_layer, nn.Linear)
    assert isinstance(model.softmax_layer, nn.Softmax)


def test_LSTMModel_forward_output_shape():
    model = LSTMModel(52, 12, False, 20)
    x = torch.rand(10, 5, 2).long()
    output = model(x)
    assert output.shape == (10, 5, 12)


def test_LSTMModel_hidden_state_reset():
    model = LSTMModel(52, 12, False, 20)
    x = torch.rand(10, 5, 2).long()

    model.reset_hidden_state(x)
    assert torch.all(torch.eq(model.hidden[0], torch.zeros(1, 10, 20)))
    assert torch.all(torch.eq(model.hidden[1], torch.zeros(1, 10, 20)))
