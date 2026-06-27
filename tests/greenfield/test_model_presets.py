"""Tests for E11 model size presets."""

from greenfield.train.model_presets import NL_E10A, NL_E11A, RENDERER_E11B
from greenfield.train.nl_parser_model import NlParserModel
from greenfield.train.nl_transformer import count_parameters
from greenfield.renderer.transformer_renderer import TransformerRendererModel


def test_e10a_preset_under_3m():
    model = NlParserModel(**NL_E10A.as_dict())
    params = count_parameters(model)
    assert params < 3_000_000


def test_e11a_preset_near_50m():
    model = NlParserModel(**NL_E11A.as_dict())
    params = count_parameters(model)
    assert 45_000_000 <= params <= 55_000_000


def test_e11b_renderer_near_50m():
    model = TransformerRendererModel(**RENDERER_E11B.as_dict())
    params = count_parameters(model)
    assert 45_000_000 <= params <= 55_000_000
