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


def espeak_data_dir(env_root: Path) -> Path | None:
    """Ubica el espeak-ng-data que trae piper en el env. None si no está."""
    p = Path(env_root) / "Lib" / "site-packages" / "piper" / "espeak-ng-data"
    return p if p.is_dir() else None


def _add_meta_data(onnx_path: Path, meta: dict) -> None:
    """Embebe metadatos en el .onnx (in-place)."""
    import onnx
    model = onnx.load(str(onnx_path))
    for k, v in meta.items():
        m = model.metadata_props.add()
        m.key = k
        m.value = str(v)
    onnx.save(model, str(onnx_path))


def _leeme(voz: str) -> str:
    return (
        "Voz Piper empaquetada para sherpa-onnx.\n\n"
        "Probar (con sherpa-onnx instalado, desde esta carpeta):\n\n"
        "  sherpa-onnx-offline-tts \\\n"
        f"    --vits-model={voz}.onnx \\\n"
        "    --vits-tokens=tokens.txt \\\n"
        "    --vits-data-dir=espeak-ng-data \\\n"
        "    --output-filename=prueba.wav \\\n"
        "    \"Hola, esto es una prueba.\"\n"
    )


def empaquetar(onnx: Path, config_json: Path, out_dir: Path, espeak_dir: Path,
               add_meta=_add_meta_data) -> Path:
    """Arma la carpeta autocontenida para sherpa-onnx. Devuelve out_dir.

    `add_meta` es inyectable (para tests): por defecto embebe metadatos con onnx."""
    onnx, config_json = Path(onnx), Path(config_json)
    out_dir, espeak_dir = Path(out_dir), Path(espeak_dir)
    config = json.loads(config_json.read_text(encoding="utf-8"))
    if "phoneme_id_map" not in config:
        raise ValueError("El config no tiene 'phoneme_id_map'.")
    voz = onnx.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_onnx = out_dir / f"{voz}.onnx"
    shutil.copyfile(onnx, dst_onnx)
    (out_dir / "tokens.txt").write_text(tokens_txt(config["phoneme_id_map"]),
                                        encoding="utf-8")
    add_meta(dst_onnx, meta_data(config))
    dst_espeak = out_dir / "espeak-ng-data"
    if dst_espeak.exists():
        shutil.rmtree(dst_espeak)
    shutil.copytree(espeak_dir, dst_espeak)
    (out_dir / "LEEME.txt").write_text(_leeme(voz), encoding="utf-8")
    return out_dir
