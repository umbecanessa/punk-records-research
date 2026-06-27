"""Model size presets for the greenfield NL + renderer stack."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NlParserPreset:
    name: str
    d_model: int
    nhead: int
    num_layers: int
    out_dim: int
    dropout: float = 0.1

    def as_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "out_dim": self.out_dim,
            "dropout": self.dropout,
        }


@dataclass(frozen=True)
class RendererPreset:
    name: str
    d_model: int
    nhead: int
    num_layers: int
    dropout: float = 0.1

    def as_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


# E10.1 — ~2M params, 96-char window (production v1.1)
NL_E10A = NlParserPreset("e10a", d_model=192, nhead=6, num_layers=4, out_dim=128)

# E11a — ~50M params, open phrasing + messy NL
NL_E11A = NlParserPreset("e11a", d_model=640, nhead=8, num_layers=10, out_dim=256)

# E9b — ~830K params
RENDERER_E9B = RendererPreset("e9b", d_model=128, nhead=4, num_layers=4)

# E11b — ~49M params, multi-template renderer phrasing
RENDERER_E11B = RendererPreset("e11b", d_model=640, nhead=8, num_layers=10)

NL_PRESETS = {p.name: p for p in (NL_E10A, NL_E11A)}
RENDERER_PRESETS = {p.name: p for p in (RENDERER_E9B, RENDERER_E11B)}
