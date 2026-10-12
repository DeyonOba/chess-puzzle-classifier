"""Instantiate callbacks and loggers from Hydra config dicts (template-style)."""

from __future__ import annotations

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig


def instantiate_callbacks(cfg: DictConfig | None) -> list[Callback]:
    out: list[Callback] = []
    if not cfg:
        return out
    for _, cb in cfg.items():
        if isinstance(cb, DictConfig) and "_target_" in cb:
            out.append(hydra.utils.instantiate(cb))
    return out


def instantiate_loggers(cfg: DictConfig | None) -> list[Logger]:
    out: list[Logger] = []
    if not cfg:
        return out
    for _, lg in cfg.items():
        if isinstance(lg, DictConfig) and "_target_" in lg:
            out.append(hydra.utils.instantiate(lg))
    return out
