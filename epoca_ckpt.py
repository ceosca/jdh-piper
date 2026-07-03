# -*- coding: utf-8 -*-
"""Imprime época actual + mejor punto de una corrida, leyendo los checkpoints.

Fallback para el botón "¿Cómo va?" cuando una corrida no tiene epoch.txt/mejor.txt
(ej. arrancó antes del escritor de progreso). Se corre como subproceso desde la GUI
para no importar torch en el proceso de la UI. Imprime dos líneas:
    epoca <N>
    mejor <época> <val_mel>
Uso: env\\python.exe epoca_ckpt.py training/<voz>
"""
import glob
import os
import pathlib
import sys

pathlib.PosixPath = pathlib.WindowsPath
import torch  # noqa: E402


def _epoca_del_mas_nuevo(ckdir):
    fs = glob.glob(os.path.join(ckdir, "*.ckpt"))
    if not fs:
        return None
    f = max(fs, key=os.path.getmtime)
    try:
        return torch.load(f, map_location="cpu", weights_only=False).get("epoch")
    except Exception:
        return None


def _mejor(ckdir):
    fs = glob.glob(os.path.join(ckdir, "*-best.ckpt"))
    if not fs:
        return None
    try:
        ck = torch.load(fs[0], map_location="cpu", weights_only=False)
        score = None
        for v in ck.get("callbacks", {}).values():
            if isinstance(v, dict) and v.get("best_model_score") is not None:
                score = float(v["best_model_score"])
        return ck.get("epoch"), score
    except Exception:
        return None


def main() -> None:
    rd = sys.argv[1] if len(sys.argv) > 1 else "."
    ckdir = os.path.join(rd, "ckpts")
    ep = _epoca_del_mas_nuevo(ckdir)
    if ep is not None:
        print(f"epoca {int(ep)}")
    mj = _mejor(ckdir)
    if mj and mj[0] is not None:
        v = f"{mj[1]:.3f}" if mj[1] is not None else "?"
        print(f"mejor {int(mj[0])} {v}")


if __name__ == "__main__":
    main()
