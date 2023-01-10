import torch

from src.models.model import MyAwesomeModel


def test_model():

    model = MyAwesomeModel()

    input = torch.randn(1, 1, 28, 28)
    output = model(input)

    assert input.shape == (1, 1, 28, 28) and output.shape == (
        1,
        10,
    ), "Input shape or output shape wrong"
