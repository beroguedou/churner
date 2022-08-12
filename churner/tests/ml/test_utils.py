import pytest
from churner.ml.utils import toto

def test_one_epoch_validation():
    assert toto() == 'yes'