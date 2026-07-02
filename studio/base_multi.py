# studio/base_multi.py
"""Cirugía de pesos mono -> multi-hablante para VITS (Piper)."""
from __future__ import annotations

import inspect


def hparams_multi(hp_mono: dict, num_speakers: int, gin_channels: int = 256) -> dict:
    """hparams del base mono, filtrados a args de VitsModel, con speakers/gin fijados."""
    from piper.train.vits.lightning import VitsModel
    validos = set(inspect.signature(VitsModel.__init__).parameters) - {"self", "kwargs"}
    hp = {k: v for k, v in hp_mono.items() if k in validos}
    hp["num_speakers"] = int(num_speakers)
    hp["gin_channels"] = int(gin_channels)
    return hp


def fusionar_pesos(mono_sd: dict, multi_sd: dict):
    """Parte del state_dict multi (init) y copia de mono toda clave con misma forma.

    Devuelve (merged, n_copiadas, n_nuevas). n_nuevas = claves de multi que quedaron
    con su valor inicial (p.ej. emb_g y las capas cond de condicionamiento por hablante).
    """
    merged = dict(multi_sd)
    copiadas = 0
    for k, v in mono_sd.items():
        if k in merged and hasattr(v, "shape") and merged[k].shape == v.shape:
            merged[k] = v
            copiadas += 1
    nuevas = len(multi_sd) - copiadas
    return merged, copiadas, nuevas
