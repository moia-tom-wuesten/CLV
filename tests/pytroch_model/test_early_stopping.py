import pytest
from src.CLV.pytorch_model.abstract_lstm import EarlyStopping


@pytest.fixture
def early_stopping():
    early_stopp = EarlyStopping(tolerance=5, min_delta=0)
    return early_stopp


def test_early_stopping_init(early_stopping):
    """
    Test the default values of an EarlyStopping instance.

    This test creates an instance of the EarlyStopping class with a tolerance of 5 and min_delta of 0.
    It verifies that the initial values of the tolerance, min_delta, counter, and early_stop attributes are 5, 0, 0, and False, respectively.
    """
    assert early_stopping.tolerance == 5
    assert early_stopping.min_delta == 0
    assert early_stopping.counter == 0
    assert early_stopping.early_stop == False


def test_early_stopping_counter_up(early_stopping):
    """
    Test that the counter increases when there is improvement in the validation loss.

    This test creates an instance of the EarlyStopping class with a tolerance of 5 and min_delta of 0.
    It then calls the EarlyStopping instance with a train loss of 1 and validation loss of 2.
    The test verifies that the counter attribute is equal to 1 and the early_stop attribute is False, indicating that the counter increased but early stopping was not triggered.
    """
    early_stopping(train_loss=1, validation_loss=2)
    assert early_stopping.counter == 1
    assert early_stopping.early_stop == False


def test_early_stopping_no_counter_up(early_stopping):
    """
    Test that the counter does not increase when there is no improvement in the validation loss.

    This test creates an instance of the EarlyStopping class with a tolerance of 5 and min_delta of 0.
    It then calls the EarlyStopping instance with a train loss of 2 and validation loss of 1.
    The test verifies that the counter attribute is equal to 0 and the early_stop attribute is False, indicating that the counter did not increase and early stopping was not triggered.
    """
    early_stopping(train_loss=2, validation_loss=1)
    assert early_stopping.counter == 0
    assert early_stopping.early_stop == False


def test_early_stopping_trigger_early_stopping(early_stopping):
    """
    Test that early stopping is triggered when there is enough improvement in the validation loss.

    This test creates an instance of the EarlyStopping class with a tolerance of 5 and min_delta of 0.
    It then calls the EarlyStopping instance with a train loss and validation loss that increases by 1 each time.
    The test verifies that the counter attribute is equal to 5 and the early_stop attribute is True, indicating that early stopping has been triggered.
    """
    for i in range(5):
        early_stopping(train_loss=i, validation_loss=i + 1)
    assert early_stopping.counter == 5
    assert early_stopping.early_stop == True
