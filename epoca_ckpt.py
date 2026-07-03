# -*- coding: utf-8 -*-
"""Imprime la época del checkpoint más nuevo de una corrida.

Fallback para el botón "¿Cómo va?" cuando todavía no hay epoch.txt ni checkpoint
periódico (ej. una corrida que arrancó antes de tener el escritor de época). Se
corre como subproceso desde la GUI para no importar torch en el proceso de la UI.
Uso: env\\python.exe epoca_ckpt.py training/<voz>
"""
import glob
import os
import pathlib
import sys

pathlib.PosixPath = pathlib.WindowsPath
import torch  # noqa: E402


def main() -> None:
    rd = sys.argv[1] if len(sys.argv) > 1 else "."
    fs = glob.glob(os.path.join(rd, "ckpts", "*.ckpt"))
    if not fs:
        return
    f = max(fs, key=os.path.getmtime)
    try:
        ep = torch.load(f, map_location="cpu", weights_only=False).get("epoch")
        if ep is not None:
            print(int(ep))
    except Exception:
        pass


if __name__ == "__main__":
    main()
