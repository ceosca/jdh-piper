# studio/sherpa_export.py
"""Empaqueta una voz Piper (.onnx + .onnx.json) para sherpa-onnx.

Produce una carpeta autocontenida <voz>/ con el .onnx (metadatos embebidos),
tokens.txt y espeak-ng-data — lista para sherpa-onnx (CLI/Android/navegador).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def tokens_txt(phoneme_id_map: dict) -> str:
    """Contenido de tokens.txt: por símbolo, su PRIMER id -> línea '<símbolo> <id>'."""
    lineas = [f"{s} {ids[0]}" for s, ids in phoneme_id_map.items()]
    return "\n".join(lineas) + "\n"


def meta_data(config: dict) -> dict:
    """Metadatos que sherpa-onnx lee del .onnx (defaults ante config incompleto)."""
    espeak = config.get("espeak") or {}
    lang = config.get("language") or {}
    audio = config.get("audio") or {}
    return {
        "model_type": "vits",
        "comment": "piper",
        "language": lang.get("name_english", "Spanish"),
        "voice": espeak.get("voice", "es"),
        "has_espeak": 1,
        "n_speakers": config.get("num_speakers", 1),
        "sample_rate": audio.get("sample_rate", 22050),
    }
