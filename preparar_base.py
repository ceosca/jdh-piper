# -*- coding: utf-8 -*-
"""Regenera el checkpoint base saneado (base_ckpt/silvio_base_clean.ckpt).

El saneado se deriva del crudo (es_MX/ald/medium): filtra los hyper_parameters a
los args de VitsModel (descarta claves de Trainer viejo que la CLI rechaza), quita
loops/callbacks y resetea epoch/global_step a 0 (para fine-tunear desde época 0).
Conserva optimizer_states + lr_schedulers (LR ~5.7e-5, ideal para fine-tune suave).

Es un artefacto DERIVADO y gitignoreado: si desaparece, corré este script.
Uso:  env\\python.exe preparar_base.py
"""
import inspect
import os
import pathlib
from pathlib import Path

pathlib.PosixPath = pathlib.WindowsPath
import torch  # noqa: E402

from piper.train.vits.lightning import VitsModel  # noqa: E402

ROOT = Path(__file__).resolve().parent
CRUDO = ROOT / "base_ckpt" / "es" / "es_MX" / "ald" / "medium" / "epoch=9999-step=1753600.ckpt"
SALIDA = ROOT / "base_ckpt" / "silvio_base_clean.ckpt"


def main() -> None:
    if not CRUDO.exists():
        raise SystemExit(f"Falta el checkpoint crudo: {CRUDO}\n"
                         "Bajalo de rhasspy/piper-checkpoints (es_MX/ald/medium).")
    valid = set(inspect.signature(VitsModel.__init__).parameters) - {"self", "kwargs"}
    ck = torch.load(str(CRUDO), map_location="cpu", weights_only=False)
    ck["hyper_parameters"] = {k: v for k, v in ck["hyper_parameters"].items() if k in valid}
    ck.pop("loops", None)
    ck.pop("callbacks", None)
    ck["epoch"] = 0
    ck["global_step"] = 0
    torch.save(ck, str(SALIDA))
    print(f"Base saneado listo: {SALIDA} ({os.path.getsize(SALIDA) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
