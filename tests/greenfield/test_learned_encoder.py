"""End-to-end learned encoder smoke test."""

from __future__ import annotations

import random

import torch

from greenfield.episodes import CurriculumStage, generate_script
from greenfield.learned_encoder import LearnedEncoder
from greenfield.runner import load_policy, run_episode
from greenfield.simulator import sample_world
from greenfield.train.dataset import OpcodeDataset
from greenfield.train.features import FEATURE_DIM
from greenfield.train.model import EventEncoderModel
from greenfield.types import Policy


def test_opcode_dataset_build():
    policy = Policy()
    ds = OpcodeDataset(size=100, seed=0, stages=[CurriculumStage.A], policy=policy)
    x, y = ds[0]
    assert x.shape[0] == FEATURE_DIM
    assert y.ndim == 0


def test_learned_encoder_quick_train_and_eval():
    device = torch.device("cpu")
    policy = Policy()
    ds = OpcodeDataset(size=2000, seed=1, stages=[CurriculumStage.A, CurriculumStage.B], policy=policy)
    model = EventEncoderModel(hidden=64)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(5):
        for i in range(0, 2000, 64):
            batch_x = []
            batch_y = []
            for j in range(i, min(i + 64, 2000)):
                x, y = ds[j]
                batch_x.append(x)
                batch_y.append(y)
            bx = torch.stack(batch_x)
            by = torch.stack(batch_y)
            logits = model(bx)
            loss = torch.nn.functional.cross_entropy(logits, by)
            optim.zero_grad()
            loss.backward()
            optim.step()

    enc = LearnedEncoder(model, device=device, stage="A")
    world = sample_world(random.Random(0), num_facts=1)
    script = generate_script(world, stage=CurriculumStage.A)
    _, metrics = run_episode(
        world=world,
        script=script,
        policy=policy,
        encoder=enc,
        stage="A",
    )
    assert metrics.query_accuracy == 1.0
