# -*- coding: utf-8 -*-
"""Regenera el checkpoint base saneado (base_ckpt/silvio_base_clean.ckpt).

El saneado se deriva del crudo (es_MX/ald/medium): filtra los hyper_parameters a
los args de VitsModel (descarta claves de Trainer viejo que la CLI rechaza), quita
loops/callbacks y resetea epoch/global_step a 0 (para fine-tunear desde época 0).
Conserva optimizer_states + lr_schedulers (LR ~5.7e-5, ideal para fine-tune suave).
SOLO toca metadata; los pesos (state_dict) quedan idénticos — no afecta la calidad.

Es un artefacto DERIVADO y gitignoreado. `asegurar_base()` lo regenera solo si falta
(lo llaman entrenar.py / entrenar_base.py). Uso manual: env\\python.exe preparar_base.py
"""
import inspect
import os
import pathlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CRUDO = ROOT / "base_ckpt" / "es" / "es_MX" / "ald" / "medium" / "epoch=9999-step=1753600.ckpt"
SALIDA = ROOT / "base_ckpt" / "silvio_base_clean.ckpt"


def regenerar(salida: Path = SALIDA) -> Path:
    if not CRUDO.exists():
        raise SystemExit(f"Falta el checkpoint crudo: {CRUDO}\n"
                         "Bajalo de rhasspy/piper-checkpoints (es_MX/ald/medium).")
    pathlib.PosixPath = pathlib.WindowsPath  # el crudo se guardó en Linux
    import torch
    from piper.train.vits.lightning import VitsModel
    valid = set(inspect.signature(VitsModel.__init__).parameters) - {"self", "kwargs"}
    ck = torch.load(str(CRUDO), map_location="cpu", weights_only=False)
    ck["hyper_parameters"] = {k: v for k, v in ck["hyper_parameters"].items() if k in valid}
    ck.pop("loops", None)
    ck.pop("callbacks", None)
    ck["epoch"] = 0
    ck["global_step"] = 0
    torch.save(ck, str(salida))
    print(f"Base saneado listo: {salida} ({os.path.getsize(salida) / 1e6:.0f} MB)")
    return salida


def asegurar_base(path) -> None:
    """Si falta el base saneado estándar, lo regenera del crudo (automático)."""
    p = Path(path)
    if p.exists():
        return
    if p.name == SALIDA.name and CRUDO.exists():
        print(f"[base] falta {p.name}; regenerando del crudo…", flush=True)
        regenerar(SALIDA)


def main() -> None:
    regenerar(SALIDA)


if __name__ == "__main__":
    main()
